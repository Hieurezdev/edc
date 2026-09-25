from project_variables import MASK_TOKEN, PAD_TOKEN, NUM_LABELS_N_MASKS
from normalizer.trainer_tools import get_label_n_masks, apply_fine_tuning_strategy
from utils import gen_dataIter, add_special_token
import torch
import numpy as np

def epoch_run(
    tokenizer, model,
    use_gpu, append_n_mask, nsw_detect,
    soft_labels, loss_weights,
    batchIter, mode, epoch, n_epochs, 
    optimizer=None
):

    mask_id = tokenizer.convert_tokens_to_ids([MASK_TOKEN])[0]
    pad_id = tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]

    labels_n_mask_prediction = None
    standard_labels = None
    loss = 0
    loss_norm = 0
    loss_n_masks_pred = 0
    loss_nsw_detection = 0
    num_batch = 0 if mode != 'train_pseudo' else len(batchIter['id'])
    if mode != 'train_pseudo':
        while True:
            try:
                batch = batchIter.__next__()
                num_batch += 1
                input_ids = batch['input_ids']
                input_tokens_tensor = torch.LongTensor(input_ids)
                output_ids = batch['output_ids']
                output_tokens_tensor = torch.LongTensor(output_ids)

                input_mask = torch.ones_like(input_tokens_tensor)
                if use_gpu:
                    input_tokens_tensor = input_tokens_tensor.cuda()
                    output_tokens_tensor = output_tokens_tensor.cuda()
                    input_mask = input_mask.cuda()

                if append_n_mask:
                    labels_n_mask_prediction = get_label_n_masks(
                        input_tokens_tensor == mask_id,
                        NUM_LABELS_N_MASKS
                    )
                    assert (((input_tokens_tensor == mask_id).nonzero() == (labels_n_mask_prediction == -1).nonzero())).all()
                    # Assigning padded input to label -1 for loss ignore
                    labels_n_mask_prediction[input_tokens_tensor == pad_id] = -1

                feeding_the_model_with_label = output_tokens_tensor.clone()

                if nsw_detect:
                    standard_labels = (input_tokens_tensor != output_tokens_tensor).long()

                    # Masking
                if optimizer is not None:
                    portion_mask = min((1- (epoch + 1) / n_epochs), 0.6)
                    mask_normed = np.random.random() < portion_mask
                    if mask_normed:
                        feeding_the_model_with_label[input_tokens_tensor == output_tokens_tensor] = -1
                        if np.random.random() < 0.5:
                            input_tokens_tensor[input_tokens_tensor != output_tokens_tensor] = mask_id

                loss_dic, logits, _ = model(
                    input_tokens_tensor, input_mask,
                    labels=feeding_the_model_with_label,
                    labels_n_masks=labels_n_mask_prediction,
                    standard_labels=standard_labels
                )
                _loss = loss_dic["loss"]
                loss_norm += loss_dic["loss_norm"].detach()
                if append_n_mask:
                    if not isinstance(loss_dic["loss_n_masks_pred"], int):
                        loss_n_masks_pred += loss_dic["loss_n_masks_pred"].detach()
                if nsw_detect:
                    if not isinstance(loss_dic["loss_nsw_detection"], int):
                        loss_nsw_detection += loss_dic["loss_nsw_detection"].detach()

                    # Training :
                loss += _loss.detach()
                if optimizer is not None:
                    _loss.backward()
                    for opti in optimizer:
                        opti.step()
                        opti.zero_grad()
                    #print("Training data optimizing")
            except StopIteration:
                print("BREAKING ITERATION")
                break
    else:
        for i in range(num_batch):
            input_ids = batchIter['input_ids'][i]
            input_tokens_tensor = torch.LongTensor(input_ids)
            output_tokens_tensor = torch.tensor(batchIter['proba'][i]) if soft_labels else torch.LongTensor(batchIter['labels'][i])
            input_mask = torch.ones_like(input_tokens_tensor)
            if use_gpu:
                input_tokens_tensor = input_tokens_tensor.cuda()
                output_tokens_tensor = output_tokens_tensor.cuda()
                input_mask = input_mask.cuda()

            if append_n_mask:
                labels_n_mask_prediction = get_label_n_masks(input_tokens_tensor == mask_id, NUM_LABELS_N_MASKS)
                assert (((input_tokens_tensor == mask_id).nonzero() == (labels_n_mask_prediction == -1).nonzero())).all()
                # Assigning padded input to label -1 for loss ignore
                labels_n_mask_prediction[input_tokens_tensor == pad_id] = -1

            if nsw_detect:
                if soft_labels:
                    standard_labels = (input_tokens_tensor != output_tokens_tensor.argmax(dim=-1)).long()
                else:
                    standard_labels = (input_tokens_tensor != output_tokens_tensor).long()

            feeding_the_model_with_label = output_tokens_tensor.clone()

            sample_weights = None
            if loss_weights:
                sample_weights = batchIter['weight'][i]
                sample_weights = torch.tensor(sample_weights)
                if use_gpu:
                    sample_weights = sample_weights.cuda()

            loss_dic, logits, _ = model(
                input_tokens_tensor, input_mask,
                labels=feeding_the_model_with_label,
                labels_n_masks=labels_n_mask_prediction,
                standard_labels=standard_labels,
                sample_weights=sample_weights,
                soft_labels=soft_labels
            )
            _loss = loss_dic["loss"]
            loss_norm += loss_dic["loss_norm"].detach()
            if append_n_mask:
                if not isinstance(loss_dic["loss_n_masks_pred"], int):
                    loss_n_masks_pred += loss_dic["loss_n_masks_pred"].detach()
            if nsw_detect:
                if not isinstance(loss_dic["loss_nsw_detection"], int):
                    loss_nsw_detection += loss_dic["loss_nsw_detection"].detach()

            # Training :
            loss += _loss.detach()
            _loss.backward()
            for opti in optimizer:
                opti.step()
                opti.zero_grad()
            #print("Training data optimizing")

    return loss/num_batch

def train(
    logger, name, tokenizer, model, mode,
    n_epochs, batch_size, fine_tuning_strategy, learning_rate,
    append_n_mask, nsw_detect, soft_labels, loss_weights,
    manual_seed, use_gpu,
    train_data, dev_data
): 
    if mode != "train_pseudo":
        train_sent_len_ls = list(set(train_data['sent_len']))
    dev_sent_len_ls = list(set(dev_data['sent_len']))
    best_val_loss = np.inf
    train_losses = []
    dev_losses = []

    for epoch in range(n_epochs):
        if mode != "train_pseudo":
            trainIter = gen_dataIter(
                train_data, batch_size, train_sent_len_ls, 
                shuffle=True, seed=manual_seed
            )
        devIter = gen_dataIter(dev_data, batch_size, dev_sent_len_ls)

        optimizer = apply_fine_tuning_strategy(
            name, fine_tuning_strategy, model, 
            learning_rate, (0.9, 0.99), 
            append_n_mask, nsw_detect
        )

        model.train()
        loss_train = epoch_run(
            tokenizer=tokenizer,
            model=model,
            use_gpu=use_gpu,
            append_n_mask=append_n_mask,
            nsw_detect=nsw_detect,
            soft_labels=soft_labels,
            loss_weights=loss_weights,
            batchIter=train_data if mode == "train_pseudo" else trainIter,
            mode=mode,
            epoch=epoch+1,
            n_epochs=n_epochs,
            optimizer=optimizer,
        )
        train_losses.append(loss_train.item())

        model.eval()
        with torch.no_grad():
            loss_dev = epoch_run(
                tokenizer=tokenizer,
                model=model,
                use_gpu=use_gpu,
                append_n_mask=append_n_mask,
                nsw_detect=nsw_detect,
                soft_labels=soft_labels,
                loss_weights=loss_weights,
                batchIter=devIter,
                mode="dev",
                epoch=epoch+1,
                n_epochs=n_epochs,
                optimizer=None,
            )
            dev_losses.append(loss_dev.item())
            if loss_dev < best_val_loss:
                best_model = model
                best_val_loss = loss_dev
                train_loss = loss_train
        
        logger.info("EPOCH {}/{}: train_loss: {} - val_loss: {} - best_val_loss: {}".format(
            epoch+1, n_epochs, loss_train, loss_dev, best_val_loss)
        )
    
    model = best_model
    losses = {
        'train_loss': train_losses,
        'dev_loss': dev_losses,
        'best_dev_loss': best_val_loss.item()
    }
    return losses, model

def predict(
    model, batch_size,
    use_gpu, nsw_detect,
    data, inference_mode, dataIter
):
    label = False
    if data is not None:
        len_ls = list(set(data['sent_len']))
        dataIter = gen_dataIter(data, batch_size=batch_size, len_list=len_ls)
    preds = []
    probas = []
    features = []
    rule_pred = []
    sent_ids = []
    input_ids = []
    output_ids = []
    align_index = []
    norm_or_not = []

    while True:
        try:
            with torch.no_grad():
                batch = dataIter.__next__()
                input_tokens_tensor = batch['input_ids']
                input_tokens_tensor = torch.LongTensor(input_tokens_tensor)
                input_mask = torch.ones_like(input_tokens_tensor)
                if use_gpu:
                    input_tokens_tensor = input_tokens_tensor.cuda()
                    input_mask = input_mask.cuda()
                model.eval()
                _, logits, feature = model(input_tokens_tensor, input_mask)

                pred = torch.argmax(logits["logits_norm"], dim=-1) # [num_words]
                if not inference_mode:
                    proba = torch.softmax(logits["logits_norm"], dim=-1) # [num_words, num_labels]

                preds.append(pred.detach().cpu().numpy())
                if not inference_mode:
                    probas.append(proba.detach().cpu().numpy())
                    features.append(feature.detach().cpu().numpy())
                    rule_pred.append(np.array(batch['weak_labels']))
                    if 'output_ids' in batch:
                        label = True
                        output_ids.append(np.array(batch['output_ids']))
                if nsw_detect:
                    norm_or_not_pred = torch.argmax(logits["logits_nsw_detection"], dim=-1)
                sent_ids.append(batch['id'])
                input_ids.append(np.array(batch['input_ids']))
                align_index.append(batch['align_index'])
                norm_or_not.append(norm_or_not_pred)
        except StopIteration:
            print("BREAKING DATA ITERATION")
            break
        
    if inference_mode:
        return {
            "id": sent_ids,
            "input_ids": input_ids,
            "align_index": align_index,
            "preds":  preds,
            "is_nsw": norm_or_not,
        }

    return {
        "id": sent_ids,
        "input_ids": input_ids,
        "output_ids": output_ids if label else None,
        "align_index": align_index,
        "preds":  preds,
        "proba": probas,
        "features": features,
        "weak_labels": rule_pred,
        "is_nsw": norm_or_not,
    }

def inference(
    tokenizer, model, topk, use_gpu, user_input
):
    inputs = tokenizer(user_input, return_tensors="pt")
    source_tokens = tokenizer.tokenize(user_input)
    add_special_token(source_tokens)
    
    model.eval()
    with torch.no_grad():
        if use_gpu:
            input_tokens_tensor = inputs.input_ids.cuda()
            input_mask = inputs.attention_mask.cuda()
        _, logits, feature = model(input_tokens_tensor, input_mask)
        # torch.argsort: Returns the indices that sort a tensor along a given dimension in ascending order by value.
        # sorted, indices = torch.sort(x)
        # if topk == 1:
        #     # pred = torch.argmax(logits["logits_norm"], dim=-1).squeeze()
        #     proba = torch.softmax(logits["logits_norm"], dim=-1)
        #     proba, pred = torch.sort(proba, descending=True)
        # else:
            # pred = torch.argsort(logits["logits_norm"], dim=-1, descending=True)[:, :, :topk].squeeze(0)
        proba = torch.softmax(logits["logits_norm"], dim=-1)
        proba, pred = torch.sort(proba, descending=True)
        pred = pred[:, :, :topk].squeeze()
        proba = proba[:, :, :topk].squeeze()
        is_nsw = torch.argmax(logits["logits_nsw_detection"], dim=-1).squeeze()

    return {
        'source_tokens': source_tokens,
        'pred': pred.cpu().tolist(),
        'proba': proba.cpu().tolist(),
        'is_nsw': is_nsw.cpu().tolist()
    }