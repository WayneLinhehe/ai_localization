import torch
from torch.autograd import Variable

x = torch.tensor([[1.0],[2.0]],)
print(x[1].item())