import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RuleAttentionNetwork(nn.Module):
    def __init__(self, student_emb_dim, num_rules, num_labels, dense_dropout=0.3, device="cuda", seed=42):
        super(RuleAttentionNetwork, self).__init__()
        self.num_labels = num_labels
        self.device = device

        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Dense layers for student embeddings
        self.dense = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(student_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dense_dropout)
        )

        # Embedding Layers
        self.rule_embed = nn.Embedding(num_rules + 1, 128, padding_idx=0)
        self.rule_bias = nn.Embedding(num_rules + 1, 1, padding_idx=0)

        # Initialization
        nn.init.xavier_uniform_(self.rule_embed.weight) # Xavier uniform initializer
        nn.init.uniform_(self.rule_bias.weight)

    def forward(self, student_embeddings, rule_ids, rule_preds_onehot):
        # Process student embeddings
        x_hidden = self.dense(student_embeddings)  # batch_size x 128
        
        # Get rule embeddings and biases
        rule_embeddings = self.rule_embed(rule_ids)  # batch_size x max_rule_seq_length x 128
        rule_biases = self.rule_bias(rule_ids).squeeze(-1)  # batch_size x max_rule_seq_length

        # Compute attention scores
        att_scores = torch.matmul(x_hidden.unsqueeze(2), rule_embeddings.transpose(2, 3)).squeeze(2)
        # att_scores = torch.bmm(x_hidden.unsqueeze(1), rule_embeddings.transpose(1, 2)).squeeze(1)
        att_scores += rule_biases
        att_sigmoid_proba = torch.sigmoid(att_scores)
        
        # Compute raw outputs
        outputs = torch.matmul(att_sigmoid_proba.float().unsqueeze(2), rule_preds_onehot.float()).squeeze(2)
        # att_scores = torch.bmm(x_hidden.unsqueeze(1), rule_embeddings.transpose(1, 2)).squeeze(1)

        # Normalize Outputs with random rule and L1 normalization
        outputs = normalize_with_random_rule(outputs, att_sigmoid_proba, rule_preds_onehot)
        outputs = l1_normalize(outputs, self.num_labels)

        return outputs, att_sigmoid_proba


class RAN:
    """
    Rule Attention Network
      * Input: text embedding x, array of rule predictions
      * Output: aggregate label
    """

    def __init__(self, args, num_rules, num_labels, logger=None, name='ran'):
        self.args = args
        self.name = name
        self.logger = logger
        self.manual_seed = args.seed
        torch.manual_seed(self.manual_seed)
        self.model_dir = args.logdir
        self.sup_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.unsup_batch_size = args.unsup_batch_size
        self.sup_epochs = args.num_epochs
        self.unsup_epochs = args.num_unsup_epochs
        self.num_labels = num_labels
        self.num_rules = num_rules
        self.device = args.device
        self.use_gpu = self.device=="cuda"

        # Using Student as an extra rule
        self.num_rules += 1
        self.student_rule_id = self.num_rules
        self.hard_student_rule = args.hard_student_rule
        self.trained = False
        self.xdim = None
        self.ignore_student = False

    def postprocess_rule_preds(self, rule_pred, student_pred=None):
        N, M = rule_pred.shape[0], rule_pred.shape[1]
        rule_mask = (rule_pred != -1).astype(int) # Have applied rules
        fired_rule_ids = [[(np.nonzero(x)[0] + 1).tolist() for x in line] for line in rule_mask]
        non_zero_rule_pred = []
        for i in range(N):
            preds_i = []
            for j, fired_rules in enumerate(fired_rule_ids[i]):
                preds_i_j = [rule_pred[i, j, id-1] for id in fired_rules]
                preds_i_j = preds_i_j + [self.num_labels] * (self.num_rules-1 - len(preds_i_j))
                preds_i.append(preds_i_j)
            non_zero_rule_pred.append(preds_i)
        
        one_hot_rule_pred = np.eye(self.num_labels + 1)[np.array(non_zero_rule_pred)]
        one_hot_rule_pred = one_hot_rule_pred[:, :, :, :-1]
        fired_rule_ids = [[x + [0] * (self.num_rules-1 - len(x)) for x in sent] for sent in fired_rule_ids]
        fired_rule_ids = np.array(fired_rule_ids)

        if student_pred is not None:
            mask_one = np.ones((N, M, 1))
            if student_pred.ndim > 3:
                student_pred = np.squeeze(student_pred, axis=None)
            if self.hard_student_rule:
                # Convert Student's soft probabilities to hard labels
                student_pred = np.eye(self.num_labels)[np.argmax(student_pred, axis=-1)]
            student_pred = student_pred[..., np.newaxis, :]  # Add axis=2
            one_hot_rule_pred = np.concatenate([student_pred, one_hot_rule_pred], axis=2)
            rule_mask = np.concatenate([mask_one, rule_mask], axis=2)
            if not self.ignore_student:
                student_rule_id = np.ones((N, M, 1)) * self.student_rule_id
            else:
                student_rule_id = np.zeros((N, M, 1))
            fired_rule_ids = np.concatenate([student_rule_id, fired_rule_ids], axis=-1)

        return rule_mask, fired_rule_ids, one_hot_rule_pred

    def init_model(self):
        self.model = RuleAttentionNetwork(self.xdim,
                                          num_rules=self.num_rules,
                                          num_labels=self.num_labels,
                                          device=self.device)
        if self.use_gpu:
            self.model = self.model.to(self.device)

    def epoch_run(self, data, mode, loss_fn, optimizer=None, scheduler=None):
        # mode in ["sup_train", "unsup_train", "dev"]
        total_loss = 0
        num_batch = len(data['id'])
        for i in range(num_batch):
            x = torch.tensor(data['features'][i])
            student_pred = data['proba'][i]
            rule_pred = data['weak_labels'][i]
            rule_one_hot, fired_rule_ids, rule_pred = self.postprocess_rule_preds(rule_pred, student_pred)
            fired_rule_ids = torch.LongTensor(fired_rule_ids)
            rule_pred = torch.LongTensor(rule_pred)
            if self.use_gpu:
                x = x.cuda()
                fired_rule_ids = fired_rule_ids.cuda()
                rule_pred = rule_pred.cuda()
            if mode != 'unsup_train':
                y = torch.tensor(data['output_ids'][i])
                if self.use_gpu:
                    y = y.cuda()
            if mode in ['sup_train', 'unsup_train']:
                optimizer.zero_grad()
                outputs, _ = self.model(x, fired_rule_ids, rule_pred)
                loss = loss_fn(outputs) if mode=='unsup_train' else loss_fn(outputs.view(-1, self.num_labels), y.view(-1))
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
            else:
                outputs_dev, _ = self.model(x, fired_rule_ids, rule_pred)
                dev_loss = loss_fn(outputs_dev.view(-1, self.num_labels), y.view(-1))
                total_loss += dev_loss.detach()
        if mode == 'unsup_train':
            scheduler.step()
        if mode == 'sup_train':
            scheduler.step(total_loss/num_batch)
        
        return total_loss/num_batch

    def train(self, train_data, dev_data=None, unsup_data=None):
        assert unsup_data is not None, "For SSL RAN you need to also provide unlabeled data... "

        if not self.trained:
            self.init_model()
        
        self.logger.info("\n\n\t\t*** Training RAN ***")

        loss_fn = MinEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        scheduler = create_learning_rate_scheduler(optimizer,
                                                   max_learn_rate=1e-2,
                                                   end_learn_rate=1e-5,
                                                   warmup_epoch_count=2,
                                                   total_epoch_count=self.sup_epochs)

        # Training loop for unsupervised data
        self.model.train()
        unsup_losses = []
        for epoch in range(self.sup_epochs):
            unsup_loss = self.epoch_run(unsup_data, mode="unsup_train", loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
            unsup_losses.append(unsup_loss.item())
            self.logger.info("Unsupervised trainning: EPOCH {}/{} - LOSS: {}".format(epoch+1, self.sup_epochs, unsup_loss))

        # Reinitialize loss function and optimizer for supervised training
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.num_labels)
        optimizer = optim.Adam(self.model.parameters())
        scheduler = create_learning_rate_scheduler(optimizer,
                                                   max_learn_rate=1e-2,
                                                   end_learn_rate=1e-5,
                                                   warmup_epoch_count=2,
                                                   total_epoch_count=self.sup_epochs)
        
        # Training loop for supervised training
        self.trained = True
        train_losses = []
        dev_losses = []
        best_val_loss = np.inf
        for epoch in range(self.sup_epochs):
            self.model.train()
            train_loss = self.epoch_run(train_data, mode="sup_train", loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
            train_losses.append(train_loss.item())
            self.model.eval()
            with torch.no_grad():
                dev_loss = self.epoch_run(dev_data, mode="dev", loss_fn=loss_fn)
                dev_losses.append(dev_loss.item())
                if dev_loss < best_val_loss:
                    best_val_loss = dev_loss
                    best_model = self.model
            self.logger.info("Supervised trainning: EPOCH {}/{} - TRAIN LOSS: {} - VAL LOSS: {} - BEST VAL LOSS: {}".format(epoch+1, self.sup_epochs, train_loss, dev_loss, best_val_loss))
        self.model = best_model
        
        return {
            'unsup_loss': unsup_losses,
            'sup_loss': train_losses,
            'dev_loss': dev_losses,
            'best_dev_loss': best_val_loss.item(),
        }

    def predict_ran(self, dataset, batch_size=128, inference_mode=False):
        y_preds = []
        att_scores = []
        soft_probas = []
        rule_masks = []

        label = False
        if 'output_ids' in dataset:
            label=True
        num_batch = len(dataset['id'])

        self.model.eval()
        with torch.no_grad():
            for i in range(num_batch):
                x_batch = torch.tensor(dataset['features'][i])
                rule_pred_batch = dataset['weak_labels'][i]
                student_pred_batch = dataset['proba'][i]
                if student_pred_batch is None:
                    random_pred = (rule_pred_batch != -1).sum(axis=-1) == 0 # no rules apply on a word -> True
                else:
                    random_pred = np.zeros((rule_pred_batch.shape[0], rule_pred_batch.shape[1]), dtype=bool)
                random_pred = torch.tensor(random_pred)
                if self.use_gpu:
                    random_pred = random_pred.cuda()
                rule_mask, fired_rule_ids, rule_pred_one_hot = self.postprocess_rule_preds(rule_pred_batch, student_pred_batch)
                fired_rule_ids = torch.LongTensor(fired_rule_ids)
                rule_pred_one_hot = torch.LongTensor(rule_pred_one_hot)
                if self.use_gpu:
                    x_batch = x_batch.cuda()
                    fired_rule_ids = fired_rule_ids.cuda()
                    rule_pred_one_hot = rule_pred_one_hot.cuda()
            
                y_pred, att_score = self.model(x_batch, fired_rule_ids, rule_pred_one_hot)

                preds = torch.argmax(y_pred, dim=-1, keepdim=False)
                max_proba = torch.max(y_pred, dim=-1)[0]
                confidence_thres = 0.5
                ignore_pred = max_proba < confidence_thres
                random_pred[ignore_pred] = True
                preds[random_pred] = -1
                if not inference_mode:
                    soft_proba = y_pred

                y_preds.append(preds.detach().cpu().numpy())
                if not inference_mode:
                    att_scores.append(att_score.detach().cpu().numpy())
                    soft_probas.append(soft_proba.detach().cpu().numpy())
                    rule_masks.append(rule_mask)

        if inference_mode:
            return {
                'id': dataset['id'],
                'input_ids': dataset['input_ids'],
                'align_index': dataset['align_index'],
                'preds': y_preds,
                'is_nsw': dataset['is_nsw']
            }

        return {
            'id': dataset['id'],
            'input_ids': dataset['input_ids'],
            'output_ids': dataset['output_ids'] if label else None,
            'align_index': dataset['align_index'],
            'preds': y_preds,
            'is_nsw': dataset['is_nsw'],
            'proba': soft_probas,
            "att_scores": att_scores,
            "rule_mask": rule_masks,
        }

    def load(self, savefile):
        self.logger.info("loading rule attention network from {}".format(savefile))
        self.model.load_state_dict(torch.load(savefile))

    def save(self, savefile):
        self.logger.info("Saving rule attention network at {}".format(savefile))
        torch.save(self.model.state_dict(), savefile)
        return


def MinEntropyLoss():
    def loss(y_prob):
        per_example_loss = -y_prob * torch.log(y_prob)
        return torch.mean(per_example_loss)
    return loss

def create_learning_rate_scheduler(optimizer,
                                   max_learn_rate=1e-2,
                                   end_learn_rate=1e-5,
                                   warmup_epoch_count=3,
                                   total_epoch_count=10):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                        total_epoch_count - warmup_epoch_count + 1))
        return float(res)

    learning_rate_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler, verbose=1)

    return learning_rate_scheduler

def l1_normalize(x, num_labels):
    x = x + 1e-05  # Avoid stability issues
    l1_norm = torch.sum(x, dim=-1, keepdim=True).detach()
    l1_norm = torch.repeat_interleave(l1_norm, repeats=num_labels, dim=-1)  # Equivalent to tf.keras.backend.repeat_elements()

    return x / l1_norm

def normalize_with_random_rule(output, att_sigmoid_proba, rule_preds_onehot):
    device=output.device
    num_labels = rule_preds_onehot.shape[-1]
    sum_prob = torch.sum(rule_preds_onehot, dim=-1).detach()
    rule_mask = (sum_prob > 0).float()
    num_rules = torch.sum(sum_prob, dim=-1).float()
    masked_att_proba = att_sigmoid_proba * rule_mask
    sum_masked_att_proba = torch.sum(masked_att_proba, dim=-1)
    uniform_rule_att_proba = num_rules - sum_masked_att_proba
    uniform_vec = torch.ones((uniform_rule_att_proba.shape[0], uniform_rule_att_proba.shape[1], num_labels)) / num_labels
    uniform_vec = uniform_vec.to(device)
    uniform_pred = torch.repeat_interleave(uniform_rule_att_proba.unsqueeze(-1), repeats=num_labels, dim=-1) * uniform_vec
    output_with_uniform_rule = output + uniform_pred
    return output_with_uniform_rule