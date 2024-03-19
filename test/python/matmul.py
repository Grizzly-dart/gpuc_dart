from sys import stdin
import torch
import json
from typing import Optional, List

l: List = json.loads(stdin.read())

inputA: torch.Tensor
inputB: torch.Tensor
inputC: Optional[torch.Tensor] = None
for item in l:
    name: str = item['name']
    if name == 'inputA':
        data = item['data']
        inputA = torch.as_tensor(data['data']).reshape(data['size'])
    elif name == 'inputB':
        data = item['data']
        inputB = torch.as_tensor(data['data']).reshape(data['size'])
    elif name == 'inputC':
        data = item['data']
        inputC = torch.as_tensor(data['data']).reshape(data['size'])


out = torch.matmul(inputA, inputB)
out = out if inputC is None else out + inputC

print(json.dumps([{'name': 'output', 'type': 'Tensor', 'data': {'data': out.flatten().tolist(), 'size': out.size()}}]))
