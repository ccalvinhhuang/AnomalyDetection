import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


normal_data = torch.randn(100, 10)  # 100 samples of normal data
fraudulent_data = torch.randn(20, 10)  # 20 samples of potentially fraudulent data



class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


input_dim = 10  # Dimensionality of input data
encoding_dim = 5  # Dimensionality of the encoding layer

# Initialize the autoencoder model
model = Autoencoder(input_dim, encoding_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert the data to DataLoader
normal_dataloader = DataLoader(TensorDataset(normal_data), batch_size=10, shuffle=True)
fraudulent_dataloader = DataLoader(TensorDataset(fraudulent_data), batch_size=10, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for data in normal_dataloader:
        optimizer.zero_grad()
        inputs = data[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
# Detect anomalies in both normal and potentially fraudulent data
anomalies = []

for data in normal_dataloader:
    inputs = data[0]
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    print(loss.item() > 0.9)
    if loss.item() > 0.9:  # Set an appropriate threshold
        anomalies.append(inputs)

for data in fraudulent_dataloader:
    inputs = data[0]
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    print(loss.item() > 0.9)
    if loss.item() > 0.9:
        anomalies.append(inputs)

print(len(anomalies))
