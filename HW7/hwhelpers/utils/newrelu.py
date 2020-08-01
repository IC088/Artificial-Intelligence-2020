'''

Guided Backprop relu

IC
'''

import torch
import torch.nn as nn


class MyReLU(nn.Module):
    def __init__(self):
        super(MyReLU, self).__init__()
    
    def forward(self,input):
        return GuidedBackpropRelU.apply(input)

class GuidedBackpropRelU(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx,grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        grad_input[grad_input<0] = 0
        grad_input[input<0] = 0
        return grad_input
