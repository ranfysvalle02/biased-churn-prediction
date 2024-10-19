import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Generate synthetic customer data for churn prediction
def generate_data(num_samples, bias_towards_churn=True):
    num_cases = np.random.poisson(10, num_samples)
    avg_response_time = np.random.gamma(5, 1, num_samples)
    num_high_priority_cases = np.random.binomial(num_cases, 0.1)
    account_age = np.random.uniform(1, 10, num_samples)
    num_products = np.random.randint(1, 5, num_samples)
    
    # Determine churn based on cases and response time
    churn = ((num_cases > 7) | (avg_response_time > 6)).astype(int)
    
    # Introduce random noise to churn labels
    noise = np.random.binomial(1, 0.1, num_samples)
    churn = np.logical_xor(churn, noise).astype(int)
    
    # Apply bias towards churn if specified
    if bias_towards_churn:
        churn = np.maximum(churn, np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]))
    else:
        churn = np.minimum(churn, np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]))
    
    return num_cases, avg_response_time, num_high_priority_cases, account_age, num_products, churn

# Load or generate data
num_samples = 10000
data = generate_data(num_samples, bias_towards_churn=False)

"""
bias_towards_churn=True
Predicted probability of churn for example 1: 0.7901
Predicted probability of churn for example 2: 0.9615
Predicted probability of churn for example 3: 0.7767
Predicted probability of churn for example 4: 0.8826
Predicted probability of churn for example 5: 0.7967
"""

"""
bias_towards_churn=False
Predicted probability of churn for example 1: 0.0242
Predicted probability of churn for example 2: 0.1502
Predicted probability of churn for example 3: 0.1188
Predicted probability of churn for example 4: 0.0319
Predicted probability of churn for example 5: 0.1013
"""

# Create PyTorch tensors from the generated data
X = torch.tensor(np.vstack(data[:-1]).T, dtype=torch.float32)
y = torch.tensor(data[-1], dtype=torch.float32)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y.unsqueeze(1), test_size=0.2, random_state=42)

class CustomerChurnPredictor(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output dimension is always 1 for binary classification
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Apply sigmoid to output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Create data loaders for training and testing
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Initialize the model with specified input and hidden dimensions
model = CustomerChurnPredictor(input_dim=5, hidden_dim=32)

# Initialize a PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=10)

# Train the model
trainer.fit(model, train_loader)

# Make predictions on new data samples
new_data_samples = torch.tensor([
    [3.0, 4.5, 1.0, 5.0, 2.0],
    [7.0, 3.2, 2.0, 2.0, 4.0],
    [1.0, 5.7, 0.0, 8.0, 1.0],
    [5.0, 4.0, 1.0, 3.0, 3.0],
    [2.0, 6.1, 0.0, 9.0, 1.0]
], dtype=torch.float32)

model.eval()
with torch.no_grad():
    predictions = model(new_data_samples)

# Print predicted probabilities of churn for each new sample
for i in range(predictions.shape[0]):
    print(f"Predicted probability of churn for example {i + 1}: {predictions[i].item():.4f}")
