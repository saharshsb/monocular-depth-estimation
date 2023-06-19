import torch
from torch import nn

#create linear regression model class
from torch import nn
class LinearRegressionModel(nn.Module):#Almost everything in pytorch inherits from nn.Module
    def __init__(self):
      super().__init__()
      self.weights=nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
      self.bias=nn.Parameter(torch.randn(1,
                                         requires_grad=True,
                                         dtype=torch.float))
    #forward method to define the computation int the model
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias