import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_layers):
       super().__init__()
       self.num_layers = num_layers
       h = [h_dim] * (num_layers - 1)

       self.layers = nn.ModuleList(nn.Linear(i,o) for i,o in zip([in_dim] + h, h + [out_dim]))

    def forward(self, tensor):
        for i, layer in enumerate(self.layers):
            tensor = F.relu(layer(tensor)) if i < self.num_layers -1 else layer(tensor)

        return  tensor
    
seed = torch.manual_seed(42)
inps = torch.rand(3)
print(inps)
outs = torch.tensor([1., 2.])
mlp_model = MLP(3,6,2,4)    
print(mlp_model)

optimizer = optim.SGD(mlp_model.parameters(), lr=1e-5, momentum=0.9)
optimizer = optim.Adam(mlp_model.parameters(), lr=1e-5, )
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = mlp_model(inps)
    loss = F.l1_loss(outputs, outs)
    print(f"loss: {loss}")
    loss.backward()
    optimizer.step()

mlp_model.eval()
with torch.no_grad():
    pred = mlp_model(inps)
print(pred)

