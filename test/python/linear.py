import json
from sys import stdin
import torch
from typing import List, Optional

l: List = json.loads(stdin.read())

input: torch.Tensor
weight: torch.Tensor
bias: Optional[torch.Tensor] = None

for item in l:
    if item['name'] == 'input':
        data = item['data']
        input = torch.tensor(data['data']).reshape(data['size'])
    elif item['name'] == 'weight':
        data = item['data']
        weight = torch.tensor(data['data']).reshape(data['size'])
    elif item['name'] == 'bias':
        data = item['data']
        bias = torch.tensor(data['data']).reshape(data['size'])

linear = torch.nn.Linear(weight.size(1), weight.size(0), bias = bias is not None)
with torch.no_grad():
    linear.weight = torch.nn.Parameter(weight)
    if bias is not None:
        linear.bias = torch.nn.Parameter(bias)

out = linear(input)

print(json.dumps([{'name': 'output', 'type': 'Tensor', 'data': {'data': out.flatten().tolist(), 'size': out.size()}}]))