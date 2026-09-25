from project_variables import AVAILABLE_FINE_TUNING_STRATEGY
import torch

def get_optimizer(parameters, lr, optimizer="adam", betas=None):
    if betas is None:
        # betas = (0.9, 0.9)
        print("DEFAULT betas:", betas)
    opt = torch.optim.Adam(parameters, lr=lr, eps=1e-9)#, betas=betas)
    return opt

def apply_fine_tuning_strategy(model_name, fine_tuning_strategy, model, lr_init, betas=None, append_n_mask=False, nsw_detect=False):
    assert fine_tuning_strategy in AVAILABLE_FINE_TUNING_STRATEGY, "{} not in {}".format(fine_tuning_strategy, AVAILABLE_FINE_TUNING_STRATEGY)
    if fine_tuning_strategy == "standard":
        assert isinstance(lr_init, float), "{} lr : type {}".format(lr_init, type(lr_init))
        optimizer = [get_optimizer(model.parameters(), lr=lr_init, betas=betas)]
        print("TRAINING : fine tuning strategy {} : learning rate constant {} betas {}".format(fine_tuning_strategy, lr_init, betas))

    # {"roberta.embeddings": 5e-5, "roberta.encoder": 2e-5, "roberta.pooler": 1e-5, "cls": 1e-5}
    else: # fine_tuning_strategy == "flexible_lr"
        lr_init_mapping = {
            'bartpho': {
                "bart.shared": 5e-5, "bart.encoder": 2e-5, "bart.decoder": 1e-5, "cls": 1e-5
                },
            'visobert': {
                "visobert.embeddings": 5e-5, "visobert.encoder": 2e-5, "visobert.pooler": 1e-5, "cls": 1e-5
                },
            'phobert': {
                "phobert.embeddings": 5e-5, "phobert.encoder": 2e-5, "phobert.pooler": 1e-5, "cls": 1e-5
                },
        }
        lr_init = lr_init_mapping[model_name]
        if append_n_mask:
            lr_init["mask_n_predictor"] = 1e-5
        if nsw_detect:
            lr_init["nsw_detector"] = 1e-5

        assert isinstance(lr_init, dict), "lr_init should be dict in {}".format(fine_tuning_strategy)
        optimizer = []
        #print([a for a, _ in model.named_parameters()])
        n_all_layers = len([a for a, _ in model.named_parameters()])
        n_optim_layer = 0
        for pref, lr in lr_init.items():
            param_group = [param for name, param in model.named_parameters() if name.startswith(pref)]
            n_optim_layer += len(param_group)
            optimizer.append(get_optimizer(param_group, lr=lr, betas=betas))
        assert n_all_layers == n_optim_layer, "ERROR : You are missing some layers in the optimization n_all {} n_optim {} ".format(n_all_layers, n_optim_layer)
    return optimizer

def get_label_n_masks(input, num_labels_n_masks):
    output = torch.empty_like(input).long()
    for ind_sent in range(input.size(0)):
        count = 0
        for ind_word in range(input.size(1)):
            if input[ind_sent, ind_word] == 1:
                output[ind_sent, ind_word] = -1
                if count == 0:
                    ind_multi_bpe = ind_word - 1
                count += 1
            elif input[ind_sent, ind_word] == 0:
                if ind_word > 0 and input[ind_sent, ind_word -1] == 1:
                    output[ind_sent, ind_multi_bpe] = min(count, num_labels_n_masks-1)
                    count = 0
                output[ind_sent, ind_word] = 0
    return output