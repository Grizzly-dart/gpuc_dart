from sys import stdin
import torch
import json

l: list = json.loads(stdin.read())

inputA: torch.Tensor
inputB: torch.Tensor
for item in l:
    name: str = item['name']
    if name == 'inputA':
        data = item['data']
        inputA = torch.as_tensor(data['data']).reshape(data['size'])
    elif name == 'inputB':
        data = item['data']
        inputB = torch.as_tensor(data['data']).reshape(data['size'])

out = torch.matmul(inputA, inputB)

print(json.dumps([{'name': 'output', 'type': 'Tensor', 'data': {'data': out.flatten().tolist(), 'size': out.size()}}]))
