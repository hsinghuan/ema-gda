import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.head.reset_parameters()

    def forward(self, x):
        return self.head(self.encoder(x))

    def get_encoder_head(self):
        return self.encoder, self.head

    def feature(self, x):
        return self.head.feature(self.encoder(x))