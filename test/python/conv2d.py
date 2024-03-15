import json
from sys import stdin

import torch

l: list = json.loads(stdin.read())

groups: int
inChannels: int
outChannels: int
kernelSize: list[int]
inp: torch.Tensor
padding: list[int]
stride: list[int]
dilation: list[int]
padding_mode: str
for item in l:
    name: str = item['name']
    if name == 'kernel':
        data = item['data']
        kernel = torch.as_tensor(data['data']).reshape(data['size'])
    elif name == 'input':
        data = item['data']
        inp = torch.as_tensor(data['data']).reshape(data['size'])
    elif name == 'groups':
        groups = item['data']
    elif name == 'padding':
        padding = item['data']
    elif name == 'stride':
        stride = item['data']
    elif name == 'dilation':
        dilation = item['data']
    elif name == 'padding_mode':
        padding_mode = item['data']

inChannels = kernel.size(1) * groups
outChannels = kernel.size(0)
kernelSize = [kernel.size(2), kernel.size(3)]

# Create a 2D convolutional layer
conv = torch.nn.Conv2d(
    in_channels=inChannels,
    out_channels=outChannels,
    padding=padding,
    stride=stride,
    dilation=dilation,
    kernel_size=kernelSize,
    bias=False,
    groups=groups,
    padding_mode=padding_mode
)
with torch.no_grad():
    conv.weight = torch.nn.Parameter(kernel)
out = conv(inp)

print(json.dumps([{'name': 'output', 'type': 'Tensor', 'data': {'data': out.flatten().tolist(), 'size': out.size()}}]))
