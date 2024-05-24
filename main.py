import numpy as np

# Example data
texts = ["I love this movie!", "This movie is terrible."]
labels = np.array([1, 0])  # 1 for positive, 0 for negative

# Simulated output from a pre-trained model (2 samples, 768 features each)
np.random.seed(0)
pretrained_output = np.random.rand(2, 768)

# Adapter layer weights (768 input features, 1 output feature)
adapter_weights = np.random.rand(768, 1)
adapter_bias = np.random.rand(1)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss
def binary_cross_entropy(preds, targets):
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

# Forward pass
def forward(pretrained_output, adapter_weights, adapter_bias):
    logits = np.dot(pretrained_output, adapter_weights) + adapter_bias
    preds = sigmoid(logits)
    return preds

# Backward pass and parameter update
def backward(pretrained_output, preds, labels, adapter_weights, adapter_bias, learning_rate=0.01):
    # Compute gradients
    d_loss = preds - labels.reshape(-1, 1)  # Gradient of the loss wrt predictions
    d_weights = np.dot(pretrained_output.T, d_loss) / len(labels)  # Gradient of the loss wrt weights
    d_bias = np.sum(d_loss) / len(labels)  # Gradient of the loss wrt bias

    # Update weights and bias
    adapter_weights -= learning_rate * d_weights
    adapter_bias -= learning_rate * d_bias

    return adapter_weights, adapter_bias

# Training loop (simplified)
epochs = 1000
for epoch in range(epochs):
    preds = forward(pretrained_output, adapter_weights, adapter_bias)
    loss = binary_cross_entropy(preds, labels)
    adapter_weights, adapter_bias = backward(pretrained_output, preds, labels, adapter_weights, adapter_bias)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example inference
test_pretrained_output = np.random.rand(1, 768)  # Simulated pre-trained output for a new sample
test_pred = forward(test_pretrained_output, adapter_weights, adapter_bias)
print(f"Predicted sentiment probability: {test_pred[0, 0]:.4f}")
