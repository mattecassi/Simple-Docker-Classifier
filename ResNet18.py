import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):

  def __init__(self,ic, oc, k=3,default_stride=1 ):

    super(ResidualBlock, self).__init__()
    self._ic = ic
    self._oc = oc
    self._k = k
    self._default_stride = default_stride

    if self._ic != self._oc:
      # define downsample
      self._downsample = nn.Sequential(
            layer_init(nn.Conv2d(self._ic, self._oc, kernel_size=1, stride=2)),
            nn.BatchNorm2d(self._oc)
      )
      stride_first = 2
    else:
      self._downsample = nn.Identity()
      stride_first = self._default_stride

    self._conv1 = nn.Sequential(
        layer_init(nn.Conv2d(self._ic, self._oc, kernel_size=self._k, stride=stride_first, padding=1)),
        nn.BatchNorm2d(self._oc),
        nn.ReLU()
        )
    self._conv2 = nn.Sequential(
        layer_init(nn.Conv2d(self._oc, self._oc, kernel_size=self._k, stride=1, padding=1)),
        nn.BatchNorm2d(self._oc)
        )
    self._relu = nn.ReLU()

  def forward(self, original):
    out = self._conv1(original)
    
    out = self._conv2(out)
    res = self._downsample(original)
    #print(out.shape, res.shape)
    return self._relu(out + res)


class ResidualGroup(nn.Module):

  def __init__(self,ic, oc, n_times=2):
    super(ResidualGroup, self).__init__()
    self._ic = ic
    self._oc = oc
    self._blk1 = ResidualBlock(self._ic, self._oc)
    self._blk2 = ResidualBlock(self._oc, self._oc)

  
  def forward(self, x):
    out = self._blk1(x)
    out = self._blk2(out)
    return out


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)

class ResNet18(nn.Module):
  def __init__(self):
    super(ResNet18, self).__init__()
    
    self._blk0 = nn.Sequential(
        layer_init(nn.Conv2d(1,64, kernel_size=7, stride=2)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2)
    )

    self._blk1 = ResidualGroup(64,64)
    self._blk2 = ResidualGroup(64,128)
    self._blk3 = ResidualGroup(128,256)
    self._blk4 = ResidualGroup(256, 512)
    
    self._blk_output = nn.Sequential(
        nn.AvgPool2d(kernel_size=7),
        View(),
        layer_init(nn.Linear(512, 10), std=0.01),
    )
  
  def forward(self, x):
    out = self._blk0(x)
    out = self._blk1(out)
    out = self._blk2(out)
    out = self._blk3(out)
    out = self._blk4(out)
    out = self._blk_output(out)
    return out


if __name__ == "__main__":
    x = torch.randn((4,1,224,224))
    net = ResNet18()
    assert net(x).shape == torch.Size([4,10]), "Something went wrong"
    print("...END...")