import torch.nn as nn
import torch.nn.functional as F
import torch

def save_model(path: str, model):
    """
    Save model to a file
    Input:
        path: path to save model to
        model: Pytorch model to save
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
      super(ValueNetwork, self).__init__()

      hidden1 = 256
      hidden2 = 128
      self.net = nn.Sequential(
          nn.Linear(input_size, hidden1),
          nn.ReLU(),
          nn.Linear(hidden1, hidden2),
          nn.ReLU(),
          nn.Linear(hidden2, 1)
      )

    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      if not isinstance(x, torch.Tensor):
          x = torch.tensor(x, dtype=torch.float32)
      was_1d = False
      if x.dim() == 1:
          x = x.unsqueeze(0)
          was_1d = True
      out = self.net(x).squeeze(-1)
      if was_1d:
          return out.squeeze(0)
      return out
