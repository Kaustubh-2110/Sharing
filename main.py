import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate 500 points for class 1
mean1 = [1, 0]
cov1 = [[0.3, 0], [0, 0.3]]  # covariance matrix with variance 0.3
class1_points = np.random.multivariate_normal(mean1, cov1, 500)
class1_labels = np.ones(500)

# Generate 500 points for class 2
mean2 = [0, 1]
cov2 = [[0.3, 0], [0, 0.3]]  # covariance matrix with variance 0.3
class2_points = np.random.multivariate_normal(mean2, cov2, 500)
class2_labels = np.zeros(500)

# Combine the data
X = np.vstack((class1_points, class2_points))
y = np.hstack((class1_labels, class2_labels))

# Step 2: Split the data into training and testing sets
X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(class1_points, class1_labels, train_size=300, random_state=42)
X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(class2_points, class2_labels, train_size=300, random_state=42)

X_train = np.vstack((X_train_class1, X_train_class2))
y_train = np.hstack((y_train_class1, y_train_class2))
X_test = np.vstack((X_test_class1, X_test_class2))
y_test = np.hstack((y_test_class1, y_test_class2))

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_a = nn.Parameter(torch.randn(in_features, rank))
        self.lora_b = nn.Parameter(torch.randn(rank, out_features))

    def forward(self, x):
        lora_update = torch.matmul(self.lora_a, self.lora_b).t()
        return torch.matmul(x, (self.weight + lora_update).t())

class SimpleNNWithLoRA(nn.Module):
    def __init__(self):
        super(SimpleNNWithLoRA, self).__init__()
        self.fc1 = LoRALayer(2, 10, rank=2)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_lora = SimpleNNWithLoRA()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lora.parameters(), lr=0.001)

# Train the model with LoRA
num_epochs = 50
for epoch in range(num_epochs):
    model_lora.train()
    optimizer.zero_grad()
    outputs = model_lora(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model with LoRA
model_lora.eval()
with torch.no_grad():
    test_outputs = model_lora(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy of the model with LoRA on the test set: {accuracy * 100:.2f}%')
    report = classification_report(y_test_tensor, predicted, target_names=['Class 0', 'Class 1'])
    print(report)

# Save the classification report to a file
with open("classification_report_lora.txt", "w") as f:
    f.write("Classification Report with LoRA:\n")
    f.write(report)
    f.write("\nAccuracy: " + str(accuracy * 100) + "%")
