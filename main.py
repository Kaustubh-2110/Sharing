import numpy as np
import gzip
import urllib.request

# Helper function to load the MNIST dataset
def load_mnist():
    def download(filename):
        url = 'http://yann.lecun.com/exdb/mnist/' + filename
        urllib.request.urlretrieve(url, filename)
    
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 1, 28, 28) / 255.0

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    download('train-images-idx3-ubyte.gz')
    download('train-labels-idx1-ubyte.gz')
    download('t10k-images-idx3-ubyte.gz')
    download('t10k-labels-idx1-ubyte.gz')
    
    train_images = load_images('train-images-idx3-ubyte.gz')
    train_labels = load_labels('train-labels-idx1-ubyte.gz')
    test_images = load_images('t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('t10k-labels-idx1-ubyte.gz')
    
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_mnist()

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

class SimpleCNN:
    def __init__(self):
        self.conv1_filters = np.random.randn(8, 1, 3, 3) * 0.1  # 8 filters, 1 input channel, 3x3 kernel
        self.conv2_filters = np.random.randn(16, 8, 3, 3) * 0.1  # 16 filters, 8 input channels, 3x3 kernel
        self.fc_weights = np.random.randn(16 * 5 * 5, 10) * 0.1  # Fully connected layer

    def forward(self, x):
        self.x = x
        self.conv1 = relu(self.convolve(x, self.conv1_filters))
        self.pool1 = self.maxpool(self.conv1)
        self.conv2 = relu(self.convolve(self.pool1, self.conv2_filters))
        self.pool2 = self.maxpool(self.conv2)
        self.fc_input = self.pool2.reshape(self.pool2.shape[0], -1)
        self.fc_output = np.dot(self.fc_input, self.fc_weights)
        self.output = softmax(self.fc_output)
        return self.output
    
    def convolve(self, x, filters):
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_height, kernel_width = filters.shape
        output_height = height - kernel_height + 1
        output_width = width - kernel_width + 1
        conv_output = np.zeros((batch_size, out_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                region = x[:, :, i:i+kernel_height, j:j+kernel_width]
                conv_output[:, :, i, j] = np.tensordot(region, filters, axes=([1, 2, 3], [1, 2, 3]))
        return conv_output

    def maxpool(self, x, size=2, stride=2):
        batch_size, channels, height, width = x.shape
        pooled_height = (height - size) // stride + 1
        pooled_width = (width - size) // stride + 1
        pooled_output = np.zeros((batch_size, channels, pooled_height, pooled_width))
        for i in range(pooled_height):
            for j in range(pooled_width):
                region = x[:, :, i*stride:i*stride+size, j*stride:j*stride+size]
                pooled_output[:, :, i, j] = np.max(region, axis=(2, 3))
        return pooled_output

    def backward(self, x, y, learning_rate=0.01):
        batch_size = x.shape[0]
        grad_output = self.output - y
        grad_fc_weights = np.dot(self.fc_input.T, grad_output) / batch_size
        
        grad_fc_input = np.dot(grad_output, self.fc_weights.T)
        grad_pool2 = grad_fc_input.reshape(self.pool2.shape)
        
        grad_conv2 = grad_pool2.repeat(2, axis=2).repeat(2, axis=3)
        grad_conv2 = grad_conv2 * relu_derivative(self.conv2)
        grad_conv2_filters = self.compute_grad_filters(self.pool1, grad_conv2, self.conv2_filters.shape)
        
        grad_pool1 = self.deconvolve(grad_conv2, self.conv2_filters, self.pool1.shape)
        grad_conv1 = grad_pool1.repeat(2, axis=2).repeat(2, axis=3)
        grad_conv1 = grad_conv1 * relu_derivative(self.conv1)
        grad_conv1_filters = self.compute_grad_filters(self.x, grad_conv1, self.conv1_filters.shape)
        
        self.fc_weights -= learning_rate * grad_fc_weights
        self.conv2_filters -= learning_rate * grad_conv2_filters
        self.conv1_filters -= learning_rate * grad_conv1_filters
    
    def compute_grad_filters(self, x, grad, filter_shape):
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_height, kernel_width = filter_shape
        grad_filters = np.zeros(filter_shape)
        for i in range(kernel_height):
            for j in range(kernel_width):
                region = x[:, :, i:height-kernel_height+i+1, j:width-kernel_width+j+1]
                grad_filters[:, :, i, j] = np.tensordot(grad, region, axes=([0, 2, 3], [0, 2, 3]))
        return grad_filters / batch_size

    def deconvolve(self, grad, filters, output_shape):
        batch_size, out_channels, height, width = output_shape
        _, in_channels, kernel_height, kernel_width = filters.shape
        grad_output = np.zeros(output_shape)
        for i in range(height):
            for j in range(width):
                grad_output[:, :, i, j] = np.tensordot(grad[:, :, i:i+1, j:j+1], filters, axes=([1], [0]))
        return grad_output

# One-hot encode labels
train_labels_oh = one_hot_encode(train_labels, 10)
test_labels_oh = one_hot_encode(test_labels, 10)

# Initialize CNN
cnn = SimpleCNN()

# Training parameters
epochs = 3
batch_size = 32
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    for i in range(0, len(train_images), batch_size):
        x_batch = train_images[i:i+batch_size]
        y_batch = train_labels_oh[i:i+batch_size]
        
        # Forward pass
        outputs = cnn.forward(x_batch)
        
        # Backward pass and update weights
        cnn.backward(x_batch, y_batch, learning_rate)
    
    # Evaluate accuracy on the test set
    test_output = cnn.forward(test_images)
    test_predictions = np.argmax(test_output, axis=1)
    accuracy = np.mean(test_predictions == test_labels)
    print(f'Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}')