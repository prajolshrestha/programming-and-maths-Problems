import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(10) ])
    
    def forward(self, tensor):
        for i, l in enumerate(self.layers):
            tensor = self.layers[i // 2](tensor) + l(tensor)
        return tensor

seed = torch.manual_seed(42)
inps = torch.rand(3,)
outs = torch.tensor([1., 2., 3])

model = mlp(3,3)
print(model)

optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inps)
    loss = F.l1_loss(outputs, outs)
    print(f"loss: {loss}")
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    pred = model(inps)

print(pred)
