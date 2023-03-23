import torch.nn as nn
import torch.nn.functional as F

class TwoLayerMLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.25):
        super(TwoLayerMLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        output = self.fc2(x1)
        return output