import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import XLMRobertaConfig, AutoModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings
from project_variables import NUM_LABELS_N_MASKS
from normalizer.model_construction.nsw_detector import BinaryPredictor

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

try:
    # apex.normalization.fused_layer_norm.FusedLayerNorm is optimized for better performance on GPU architectures compared to nn.LayerNorm
    from apex.normalization.fused_layer_norm import FusedLayerNorm as ViSoBertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class ViSoBertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(ViSoBertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class ViSoBertLMHead(nn.Module):
    def __init__(self, config, visobert_model_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = ViSoBertLayerNorm(config.hidden_size, eps=1e-12)

        num_labels = visobert_model_embedding_weights.size(0)
        self.decoder = nn.Linear(visobert_model_embedding_weights.size(1), num_labels, bias=False)
        self.decoder.weight = visobert_model_embedding_weights
        self.decoder.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class ViSoBertMaskNPredictionHead(nn.Module):
    def __init__(self, config):
        super(ViSoBertMaskNPredictionHead, self).__init__()
        self.mask_predictor_dense = nn.Linear(config.hidden_size, 50)
        self.mask_predictor_proj = nn.Linear(50, NUM_LABELS_N_MASKS)
        self.activation = gelu

    def forward(self, sequence_output):
        mask_predictor_state = self.activation(self.mask_predictor_dense(sequence_output))
        prediction_scores = self.mask_predictor_proj(mask_predictor_state)
        return prediction_scores
    
class ViSoBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super(ViSoBertForMaskedLM, self).__init__()
        self.config = config
        self.visobert = AutoModel.from_pretrained('uitnlp/visobert')
        self.visobert.resize_token_embeddings(self.config.vocab_size)
        self.visobert.config.vocab_size = self.config.vocab_size
        self.visobert.config.max_position_embeddings = self.config.max_position_embeddings
        self.visobert.config.mask_n_predictor = self.config.mask_n_predictor
        self.visobert.config.nsw_detector = self.config.nsw_detector
        self.visobert.embeddings = XLMRobertaEmbeddings(self.visobert.config)
        self.cls = ViSoBertLMHead(self.config, self.visobert.embeddings.word_embeddings.weight)
        self.mask_n_predictor = ViSoBertMaskNPredictionHead(config) if config.mask_n_predictor else None
        self.nsw_detector = BinaryPredictor(config.hidden_size, dense_dim=100) if config.nsw_detector else None
        self.num_labels_n_mask = NUM_LABELS_N_MASKS

    def forward(self, input_ids, attention_mask=None,
                labels=None, labels_n_masks=None, standard_labels=None,
                sample_weights=None, soft_labels=False):
        
        # Extract features
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.visobert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state

        # Calculate predictions
        prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)

        loss_dict = OrderedDict([
            ("loss", None), ("loss_norm", 0), ("loss_n_masks_pred", 0), ("loss_nsw_detection", 0)
        ])
        pred_dict = OrderedDict([
            ("logits_norm", None), ("logits_n_masks_pred", None), ("logits_nsw_detection", None)
        ])

        if self.mask_n_predictor is not None:
            assert self.num_labels_n_mask > 0, "ERROR  "
            logits_n_mask_prediction = self.mask_n_predictor(sequence_output)
            pred_dict["logits_n_masks_pred"] = logits_n_mask_prediction
        # Calculate predictions for NSW detection
        if self.nsw_detector is not None:
            standard_logits = self.nsw_detector(sequence_output) 
            pred_dict["logits_nsw_detection"] = standard_logits
        pred_dict["logits_norm"] = prediction_scores

        # Calculate loss
        if labels is not None:
            if self.mask_n_predictor is not None:
                assert labels_n_masks is not None, "ERROR : you provided labels for normalization and self.mask_n_predictor : so you should provide labels_n_mask_prediction"
                if sample_weights is None:
                    loss_fct_masks_pred = CrossEntropyLoss(ignore_index=-1)
                    loss_dict["loss_n_masks_pred"] = loss_fct_masks_pred(
                        logits_n_mask_prediction.view(-1, self.num_labels_n_mask), labels_n_masks.view(-1)
                    )
                else:
                    loss_fct_masks_pred = CrossEntropyLoss(ignore_index=-1, reduce='none')
                    loss_dict["loss_n_masks_pred"] = (loss_fct_masks_pred(
                        logits_n_mask_prediction.view(-1, self.num_labels_n_mask), 
                        labels_n_masks.view(-1)
                    ) * sample_weights).mean()

            if self.nsw_detector is not None:
                assert standard_labels is not None, "ERROR : you provided labels for normalization and self.nsw_detector : so you should provide standard_labels"
                if sample_weights is None:
                    loss_fct_nsw_pred = CrossEntropyLoss(ignore_index=-1)
                    loss_dict["loss_nsw_detection"] = loss_fct_nsw_pred(
                        standard_logits.view(-1, 2), standard_labels.view(-1)
                    )
                else:
                    loss_fct_nsw_pred = CrossEntropyLoss(ignore_index=-1, reduce='none')
                    loss_dict["loss_nsw_detection"] = (loss_fct_nsw_pred(
                        standard_logits.view(-1, 2), standard_labels.view(-1)
                    ) * sample_weights).mean()

            num_labels = self.config.vocab_size + 1
            masked_lm_labels = labels.view(-1, num_labels) if soft_labels else labels.view(-1)

            if sample_weights is None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, num_labels), masked_lm_labels
                )
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-1, reduce='none')
                masked_lm_loss = (loss_fct(
                    prediction_scores.view(-1, num_labels), masked_lm_labels
                ) * sample_weights).mean()
            loss_dict["loss_norm"] = masked_lm_loss

        loss_dict["loss"] = loss_dict["loss_norm"] + loss_dict["loss_n_masks_pred"] + loss_dict["loss_nsw_detection"]

        return loss_dict, pred_dict, sequence_output


def get_visobert_normalizer(vocab_size, checkpoint_dir=None, mask_n_predictor=False, nsw_detector=False):
    config = XLMRobertaConfig()
    config.vocab_size = vocab_size
    config.max_position_embeddings = 1024
    config.mask_n_predictor = mask_n_predictor
    config.nsw_detector = nsw_detector
    model = ViSoBertForMaskedLM(config)
    #model.resize_token_embeddings(vocab_size)
    num_labels = vocab_size + 1
    if checkpoint_dir is None:
        space_vector = torch.normal(
            torch.mean(model.visobert.embeddings.word_embeddings.weight.data, dim=0), 
            std=torch.std(model.visobert.embeddings.word_embeddings.weight.data, dim=0)
        ).unsqueeze(0)
        output_layer = torch.cat((model.visobert.embeddings.word_embeddings.weight.data, space_vector), dim=0)
        model.cls.decoder = nn.Linear(model.config.hidden_size, num_labels, bias=False)
        model.cls.decoder.weight = nn.Parameter(output_layer)
        model.cls.bias = nn.Parameter(torch.zeros(num_labels))
    else:
        model.cls.decoder = nn.Linear(model.config.hidden_size, num_labels, bias=False)
        model.cls.bias = nn.Parameter(torch.zeros(num_labels))
        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))

    return model