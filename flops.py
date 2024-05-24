# -- coding: utf-8 --
import torch
import torchvision
from unettrans import Unettrans
from thop import profile

# Model
print('==> Building model..')

model = Unettrans()

dummy_input = torch.randn(1, 4, 160, 160)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

