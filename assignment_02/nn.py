import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Constants
TRAIN_TEST_RATIO = 0.8
VAL_TEST_RATIO = 1 / 4


RANDOM_SEED = 44
WEIGHT_INIT_SIGMA = 0.5  # Standard deviation for weight initialization
LEARNING_RATE = 0.00005
LR_DECAY = 0.8
LR_UPDATE_FREQ = 1000
LR_DECAY_START = 10000
EPOCHS = 20000

HIDDEN_SIZE = 24

# Forward propagation
DROPOUT_RATIO = 2 / 24

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

# Define input (X) and output (y)
X = data.iloc[:, :-1].values  # Features (all except last column)
y = data.iloc[:, -1].values  # Target (wine quality score)

# Convert target to classification (quality 3-9 -> categories)


def qual_2_cat(quality):
    if quality <= 4:
        return 0
    elif quality <= 6:
        return 1
    else:
        return 2


vectorized_qual_2_cat = np.vectorize(qual_2_cat)
# make 3 categories, each with 2 classes (3-4), (5-6), (7-9)

# y = vectorized_qual_2_cat(y)

# One-hot encode target variable

encoder = OneHotEncoder(sparse_output=False)

y = encoder.fit_transform(y.reshape(-1, 1))


# Split into train, validation, and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=TRAIN_TEST_RATIO,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    train_size=VAL_TEST_RATIO,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=y_test,
)

# Print dataset sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Total dataset size: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {y.shape[1]}")

# Standardize features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Only fit on training data
X_test = scaler.transform(X_test)

# Define network architecture
input_size = 11  # Number of features
hidden_size = HIDDEN_SIZE  # Hidden layer neurons
output_size = 6  # Number of classes

# Initialize weights and biases
np.random.seed(RANDOM_SEED)
W1 = np.random.normal(0, WEIGHT_INIT_SIGMA, size=(input_size, hidden_size))
b1 = np.random.normal(0, WEIGHT_INIT_SIGMA, size=(1, hidden_size))
W2 = np.random.normal(0, WEIGHT_INIT_SIGMA, size=(hidden_size, output_size))
b2 = np.random.normal(0, WEIGHT_INIT_SIGMA, size=(1, output_size))


# Activation functions
def relu(Z: np.ndarray) -> np.ndarray:
    """Implement the ReLU activation function.

    Args:
        Z (numpy.ndarray): The input array (pre-activation values)

    Returns:
        numpy.ndarray: The output after applying ReLU (post-activation values)
    """

    return np.maximum(0, Z)


def softmax(Z):
    """Implement the softmax activation function.

    Softmax converts raw scores into probabilities by:
    1. Taking the exponential of each value
    2. Normalizing by dividing by the sum of all exponentials

    The formula is: softmax(z_i) = exp(z_i) / sum(exp(z_j)) for all j

    Args:
        Z (numpy.ndarray): The input array of shape (n_samples, n_classes)

    Returns:
        numpy.ndarray: Probability distribution where each row sums to 1,
                      same shape as input Z
    """

    # "Safe" Softmax: https://en.wikipedia.org/wiki/Softmax_function#Numerical_algorithms

    m = np.max(Z, axis=1, keepdims=True)
    e_x = np.exp(Z - m)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def forward_propagation(X):
    """Compute the forward pass of the neural network.

    This function:
    1. Calculates the first layer pre-activation: Z1 = X·W1 + b1
    2. Applies ReLU and dropout: A1 = dropout(relu(Z1), DROPOUT_RATIO)
    3. Calculates the second layer pre-activation: Z2 = A1·W2 + b2
    4. Applies softmax: A2 = softmax(Z2)

    Args:
        X (numpy.ndarray): Input features of shape (n_samples, n_features)

    Returns:
        tuple: A tuple containing:
            - Z1 (numpy.ndarray): First layer pre-activation values (n_samples, hidden_size)
            - A1 (numpy.ndarray): First layer activation values (n_samples, hidden_size)
            - Z2 (numpy.ndarray): Second layer pre-activation values (n_samples, n_classes)
            - A2 (numpy.ndarray): Output probabilities (n_samples, n_classes)
    """

    Z1 = np.dot(X, W1) + b1
    A1 = dropout(relu(Z1), DROPOUT_RATIO)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def dropout(X, dropout_ratio):
    """Randomly set a fraction of the input neurons to 0.

    Args:
        X (numpy.ndarray): Input array to apply dropout to.
        dropout_ratio (float): Fraction of neurons to dropout.

    Returns:
        numpy.ndarray: Input array with a fraction of neurons randomly set to 0.
    """

    if isTraining == False:
        return X

    mask = np.random.rand(*X.shape) > dropout_ratio
    return X * mask


# Compute loss (categorical cross-entropy)
def compute_loss(y_true, y_pred):
    """Compute the categorical cross-entropy loss.

    Categorical cross-entropy is calculated as:
    -sum(y_true * log(y_pred)) averaged over all samples

    A small epsilon is typically added to y_pred to avoid log(0) errors.

    Args:
        y_true (numpy.ndarray): One-hot encoded true labels of shape (n_samples, n_classes)
        y_pred (numpy.ndarray): Predicted probabilities from softmax of shape (n_samples, n_classes)

    Returns:
        float: Average categorical cross-entropy loss across all samples
    """

    return -np.sum(y_true * np.log(y_pred + 1e-15))  # add small epsilon to avoid log(0)


# Backpropagation
def backward_propagation(X, y, Z1, A1, A2):
    """Compute the backward pass of the network."""

    global W1, b1, W2, b2

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    da1 = np.dot(dZ2, W2.T)
    dZ1 = da1 * (Z1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases

    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2


def update_lr(epoch: int):
    """Update the learning rate based on the epoch number.

    Args:
        epoch (int): Current epoch number
    """
    global LEARNING_RATE

    if epoch % LR_UPDATE_FREQ == 0 and epoch > LR_DECAY_START:
        LEARNING_RATE *= LR_DECAY


# Training loop

training_losses = []

isTraining = True
for epoch in range(EPOCHS):
    # Perform forward propagation
    Z1, A1, Z2, A2 = forward_propagation(X_train)

    # Compute loss
    loss = compute_loss(y_train, A2) / X_train.shape[0]

    training_losses.append(loss)

    # Perform backward propagation
    backward_propagation(X_train, y_train, Z1, A1, A2)

    update_lr(epoch)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {LEARNING_RATE:.6f}")

    # Evaluate on validation set
    if epoch % 1000 == 0:
        _, _, _, A2_val = forward_propagation(X_test)
        val_loss = compute_loss(y_test, A2_val) / X_test.shape[0]
        print(f"Validation Loss: {val_loss:.4f}")

# isTraining = False
# Evaluate on test set
_, _, _, A2_test = forward_propagation(X_test)
y_pred = np.argmax(A2_test, axis=1)
y_true = np.argmax(y_test, axis=1)

loss = compute_loss(y_test, A2_test) / X_test.shape[0]
print(f"Test Loss: {loss:.4f}")

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Make loss plot
import matplotlib.pyplot as plt

plt.plot(training_losses)
plt.title("Training Loss")

plt.show()
