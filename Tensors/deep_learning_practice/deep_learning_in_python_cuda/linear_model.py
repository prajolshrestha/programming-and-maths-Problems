import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class linear_regression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim)

    def forward(self, tensor):
        return self.linear_layer(tensor)



##############################################################
# inputs and targets
seed = torch.manual_seed(42) 
inps = torch.rand(3)
tgts = torch.tensor([1., 2.])

# model
linear_model = linear_regression(3,2)


# optimizer
optimizer = optim.SGD(linear_model.parameters(), lr=1e-5, momentum=0.9)

logger.info("Starting training ...")

# training
epochs = 10000
for epoch in range(epochs):

    logger.debug(f"Epoch {epoch+1} started ...")

    #1. initialize gradients = 0
    optimizer.zero_grad()

    #2. Forward pass 
    outputs = linear_model(inps.unsqueeze(0))
  

    #3. compute loss
    loss = F.l1_loss(outputs.squeeze(), tgts)
    
    logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

    #4. backward pass
    loss.backward()

    #5. Update parameters
    optimizer.step()


# Evaluation
linear_model.eval()
with torch.no_grad():
    preds = linear_model(inps.unsqueeze(0))

print(preds)



