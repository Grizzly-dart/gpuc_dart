from sys import stdin
import json
import torch

l: list = json.loads(stdin.read())

kernelSize: list[int]
padding: list[int]
padding_mode: str
stride: list[int]
dilation: list[int]
inp: torch.Tensor
for item in l:
    name: str = item['name']
    if name == 'kernelSize':
        kernelSize = item['data']
    elif name == 'padding':
        padding = item['data']
    elif name == 'padding_mode':
        padding_mode = item['data']
    elif name == 'stride':
        stride = item['data']
    elif name == 'dilation':
        dilation = item['data']
    elif name == 'input':
        data = item['data']
        inp = torch.as_tensor(data['data']).reshape(data['size'])

maxpool = torch.nn.MaxPool2d(
    kernelSize,
    stride=stride,
    padding=padding,
    dilation=dilation,
    return_indices=False,
    ceil_mode=False)
out = maxpool(inp)

print(json.dumps([{'name': 'output', 'type': 'Tensor', 'data': {'data': out.flatten().tolist(), 'size': out.size()}}]))