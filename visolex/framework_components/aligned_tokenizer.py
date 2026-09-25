from utils import run_strip_accents
from project_variables import MASK_TOKEN, NULL_STR

def remove_diacritics(sent_list, rm_accent_ratio):
    new_sent_list = []
    for sent in sent_list:
        word_list = []
        for word in sent:
            new_word = run_strip_accents(word, rm_accent_ratio)
            word_list.append(new_word)
        new_sent_list.append(word_list)
    return new_sent_list

def aligned_tokenize(data, tokenizer, method, lower_case, rm_accent_ratio, strip_accents):
    tokenized_src_ls = []
    tokenized_tgt_ls = [] if method != "unlabeled" else None
    aligned_idx_ls = []
    weak_label_ls = []
    sent_len_ls = []

    ids = data['id'].tolist()
    inputs = remove_diacritics(
        data['input'].tolist(), rm_accent_ratio
    ) if strip_accents else data['input'].tolist()
    outputs = data['output'].tolist() if method != "unlabeled" else None
    weak_rules = [col for col in data.columns if col.startswith("rule")]
    weak_labels = data[weak_rules].values
    num_sents = len(data)
    num_rules = weak_labels.shape[1]

    for i in range(num_sents):
        tokenized_src = []
        tokenized_tgt = []
        aligned_idx = []
        tokenized_rule_preds = []

        input_seq = inputs[i]
        output_seq = outputs[i] if method != "unlabeled" else None
        rule_preds = weak_labels[i]
        if lower_case:
            input_seq = [token.lower() for token in input_seq]
            output_seq = [token.lower() for token in output_seq] if method != "unlabeled" else None
            for j, preds in enumerate(rule_preds):
                rule_preds[j] = [token.lower() for token in preds]
        aligned = 0
        for idx, source_token in enumerate(input_seq):
            target_token = output_seq[idx] if method != "unlabeled" else None
            rule_pred = [pred[idx] for pred in rule_preds]
            tokenized_source = tokenizer.tokenize(source_token)
            tokenized_target = tokenizer.tokenize(target_token) if method != "unlabeled" else None
 
            if method != "unlabeled":
                len_diff = len(tokenized_source) - len(tokenized_target)
                if len_diff < 0:
                    tokenized_source.extend([MASK_TOKEN]*abs(len_diff))
                elif len_diff > 0:
                    tokenized_target.extend([NULL_STR]*len_diff)
                
            tokenized_rule_pred = [tokenizer.tokenize(token) for token in rule_pred]
            for j, pred in enumerate(tokenized_rule_pred):
                if len(pred) < len(tokenized_source):
                    tokenized_rule_pred[j].extend([NULL_STR]*(len(tokenized_source)-len(pred)))
                elif len(pred) > len(tokenized_source):
                    tokenized_rule_pred[j] = pred[:len(tokenized_source)]

            tokenized_src.extend(tokenized_source)
            if method != "unlabeled":
                tokenized_tgt.extend(tokenized_target)
            aligned_idx.extend([aligned]*len(tokenized_source))
            aligned += 1

            if not tokenized_rule_preds:
                tokenized_rule_preds = tokenized_rule_pred
            else:
                for j in range(num_rules):
                    tokenized_rule_preds[j].extend(tokenized_rule_pred[j])

        add_special_token(tokenized_src)
        if method != "unlabeled":
            add_special_token(tokenized_tgt)
        for j in range(len(tokenized_rule_preds)):
            add_special_token(tokenized_rule_preds[j])

        sent_len = len(tokenized_src)
            
        tokenized_src_ls.append(tokenized_src)
        if method != "unlabeled":
            tokenized_tgt_ls.append(tokenized_tgt)
        aligned_idx_ls.append(aligned_idx)
        weak_label_ls.append(tokenized_rule_preds)
        sent_len_ls.append(sent_len)

    # Converts tokens to ids
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_src_ls]
    output_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_tgt_ls] if method != "unlabeled" else None
    for i, weak_labels in enumerate(weak_label_ls):
        weak_label_ls[i] = [tokenizer.convert_tokens_to_ids(sent) for sent in weak_labels]

    reshaped_weak_label = []
    for i, sent in enumerate(weak_label_ls):
        num_rules = len(sent)
        num_words = len(sent[0])
        new_sent = []
        for j in range(num_words):
            new_sent.append([rule[j] for rule in sent])
        reshaped_weak_label.append(new_sent)
    reshaped_weak_label
          
    if method == "unlabeled":
        preprocessed_dataset = {
            'id': ids,
            'input': inputs,
            'input_ids': input_ids,
            'align_index': aligned_idx_ls,
            'weak_labels': reshaped_weak_label,
            'sent_len': sent_len_ls,
        }
    else:
        preprocessed_dataset = {
            'id': ids,
            'input': inputs,
            'output': outputs,
            'input_ids': input_ids,
            'output_ids': output_ids,
            'align_index': aligned_idx_ls,
            'weak_labels': reshaped_weak_label,
            'sent_len': sent_len_ls,
        }
    return preprocessed_dataset