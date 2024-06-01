import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x: (batch_size, time_steps, *input_size)
        batch_size, time_steps = x.shape[:2]

        # Reshape input to (batch_size * time_steps, *input_size)
        x_reshape = x.contiguous().view(-1, *x.shape[2:])

        # Apply the module to the reshaped input
        y = self.module(x_reshape)

        # Reshape the output back to (batch_size, time_steps, *output_size)
        y = y.contiguous().view(batch_size, time_steps, *y.shape[1:])

        return y
