import os
import torch
from normalizer.model_construction.bartpho import get_bartpho_normalizer
from normalizer.model_construction.phobert import get_phobert_normalizer
from normalizer.model_construction.visobert import get_visobert_normalizer
from normalizer.trainer_methods import train, predict, inference

class Trainer:
    def __init__(self, args, tokenizer, logger=None):
        self.args = args
        self.name = args.student_name
        self.logger = logger
        self.tokenizer = tokenizer
        self.manual_seed = args.seed
        self.model_dir = args.logdir
        self.sup_batch_size = args.train_batch_size
        self.unsup_batch_size = args.unsup_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.sup_epochs = args.num_epochs
        self.unsup_epochs = args.num_unsup_epochs
        self.device = args.device
        self.use_gpu = self.device=='cuda'
        if self.name == 'bartpho':
            self.model = get_bartpho_normalizer(
                len(self.tokenizer), 
                mask_n_predictor=args.append_n_mask, 
                nsw_detector=args.nsw_detect
            )
        elif self.name == 'phobert':
            self.model = get_phobert_normalizer(
                len(self.tokenizer), 
                mask_n_predictor=args.append_n_mask, 
                nsw_detector=args.nsw_detect
            )
        elif self.name == 'visobert':
            self.model = get_visobert_normalizer(
                len(self.tokenizer), 
                mask_n_predictor=args.append_n_mask, 
                nsw_detector=args.nsw_detect
            )
        if self.use_gpu:
            self.model = self.model.to(self.device)
        self.fine_tuning_strategy = args.fine_tuning_strategy
        self.learning_rate = args.learning_rate
        self.loss_weights = args.loss_weights
        self.soft_labels = args.soft_labels
        self.append_n_mask = args.append_n_mask
        self.nsw_detect = args.nsw_detect
        self.topk = args.topk

    def train(self, train_data, dev_data=None): 
        losses, self.model = train(
            logger=self.logger,
            name=self.name,
            tokenizer=self.tokenizer,
            model=self.model,
            mode="train",
            n_epochs=self.sup_epochs,
            batch_size=self.sup_batch_size,
            fine_tuning_strategy=self.fine_tuning_strategy,
            learning_rate=self.learning_rate,
            append_n_mask=self.append_n_mask,
            nsw_detect=self.nsw_detect,
            soft_labels=self.soft_labels,
            loss_weights=self.loss_weights,
            manual_seed=self.manual_seed,
            use_gpu=self.use_gpu,
            train_data=train_data, 
            dev_data=dev_data
        )
        return losses

    def finetune(self, train_data, dev_data=None):
        # Similar to training but with smaller learning rate
        losses, self.model = train(
            logger=self.logger,
            name=self.name,
            tokenizer=self.tokenizer,
            model=self.model,
            mode="finetune",
            n_epochs=self.sup_epochs,
            batch_size=self.sup_batch_size,
            fine_tuning_strategy=self.fine_tuning_strategy,
            learning_rate=self.learning_rate,
            append_n_mask=self.append_n_mask,
            nsw_detect=self.nsw_detect,
            soft_labels=self.soft_labels,
            loss_weights=self.loss_weights,
            manual_seed=self.manual_seed,
            use_gpu=self.use_gpu,
            train_data=train_data, 
            dev_data=dev_data
        )
        return losses

    def train_pseudo(self, train_data, dev_data=None):
        losses, self.model = train(
            logger=self.logger,
            name=self.name,
            tokenizer=self.tokenizer,
            model=self.model,
            mode="train_pseudo",
            n_epochs=self.unsup_epochs,
            batch_size=self.unsup_batch_size,
            fine_tuning_strategy=self.fine_tuning_strategy,
            learning_rate=self.learning_rate,
            append_n_mask=self.append_n_mask,
            nsw_detect=self.nsw_detect,
            soft_labels=self.soft_labels,
            loss_weights=self.loss_weights,
            manual_seed=self.manual_seed,
            use_gpu=self.use_gpu,
            train_data=train_data, 
            dev_data=dev_data
        )
        return losses

    def predict(self, data, inference_mode, dataIter=None):
        res = predict(
            model=self.model,
            batch_size=self.eval_batch_size,
            use_gpu=self.use_gpu,
            nsw_detect=self.nsw_detect,
            data=data, 
            inference_mode=inference_mode, 
            dataIter=dataIter
        )
        return res

    def inference(self, user_input):
        output = inference(
            tokenizer=self.tokenizer,
            model=self.model,
            topk=self.topk,
            use_gpu=self.use_gpu,
            user_input=user_input
        )
        return output

    def load(self, savefolder):
        model_file = os.path.join(savefolder, "final_model.pt")
        self.logger.info("Loading student from {}".format(model_file))
        self.model.load_state_dict(torch.load(model_file))
        return

    def save(self, savefolder):
        model_file = os.path.join(savefolder, "final_model.pt")
        self.logger.info("Saving model at {}".format(model_file))
        torch.save(self.model.state_dict(), model_file)
        return