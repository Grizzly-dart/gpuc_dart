from sys import stdin
import torch
import json

l: list = json.loads(stdin.read())

input: torch.Tensor

for item in l:
    name: str = item['name']
    if name == 'input':
        data = item['data']
        input = torch.as_tensor(data['data']).reshape(data['size'])

out = input.transpose(0, 1)

print(json.dumps([{'name': 'output', 'type': 'Tensor', 'data': {'data': out.flatten().tolist(), 'size': out.size()}}]))