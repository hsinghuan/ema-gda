import random
import numpy as np
import torch
import torch.nn.functional as F

def get_device(gpuID):
    if torch.cuda.is_available():
        device = "cuda:" + str(gpuID)
    else:
        device = "cpu"
    return device


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

@torch.no_grad()
def eval(loader, encoder, head, device="cpu"):
    encoder.eval()
    head.eval()
    total_loss = 0
    total_correct = 0
    total_num = 0
    for data, y in loader:
        data, y = data.to(device), y.to(device)
        output = head(encoder(data))
        loss = F.nll_loss(F.log_softmax(output, dim=1), y)

        pred = torch.argmax(output, dim=1)
        total_correct += torch.eq(pred, y).sum().item()
        total_loss += loss.item() * data.shape[0]
        total_num += data.shape[0]

    return total_loss / total_num, total_correct / total_num


