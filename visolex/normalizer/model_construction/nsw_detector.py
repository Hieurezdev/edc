import torch.nn as nn

class BinaryPredictor(nn.Module):
    def __init__(self, input_dim, dense_dim, dense_dim_2=0, verbose=1):
        super(BinaryPredictor, self).__init__()
        self.verbose = verbose
        self.dense_2 = None
        self.dense = None
        dim_predictor = input_dim
        if dense_dim is not None:
            if dense_dim>0:
                self.dense = nn.Linear(input_dim, dense_dim)
                dim_predictor = dense_dim
                if dense_dim_2 is not None:
                    if dense_dim_2 > 0:
                        self.dense_2 = nn.Linear(dense_dim, dense_dim_2)
                        dim_predictor = dense_dim_2
        else:
            assert dense_dim_2 is None or dense_dim_2 == 0, "ERROR : dense_dim_2 cannot be not null if dense_dim is "
        self.predictor = nn.Linear(dim_predictor , out_features=2)

    def forward(self, encoder_state_projected):
        # encoder_state_projected size : [batch, ??, dim decoder (encoder state projected)]

        if self.dense is not None:
            intermediary = nn.ReLU()(self.dense(encoder_state_projected))
            if self.dense_2 is not None:
                intermediary = nn.ReLU()(self.dense_2(intermediary))
        else:
            intermediary = encoder_state_projected
            # hidden_state_normalize_not  size [batch, ???, 2]
        hidden_state_normalize_not = self.predictor(intermediary)

        return hidden_state_normalize_not