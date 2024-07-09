import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Fetch dataset
parkinsons = fetch_ucirepo(id=174)

# Data (as pandas dataframes)
X = parkinsons.data.features
y = parkinsons.data.targets

# Ensure y is 1-dimensional
y = y.to_numpy().ravel()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bat Algorithm for neural network training
class BatAlgorithm:
    def __init__(self, n_bats, n_iterations, input_size, hidden_size, output_size, f_min=0, f_max=2):
        self.n_bats = n_bats
        self.n_iterations = n_iterations
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.f_min = f_min
        self.f_max = f_max
        self.A = 0.5
        self.r = 0.5
        self.alpha = 0.9
        self.gamma = 0.9

        # Initialize bat population (weights and biases)
        self.population = [self.initialize_network() for _ in range(n_bats)]
        self.best_bat = self.population[0]
        self.best_fitness = float('inf')

    def initialize_network(self):
        network = {
            'W1': np.random.randn(self.input_size, self.hidden_size),
            'b1': np.random.randn(1, self.hidden_size),
            'W2': np.random.randn(self.hidden_size, self.output_size),
            'b2': np.random.randn(1, self.output_size)
        }
        return network

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, network, X):
        Z1 = np.dot(X, network['W1']) + network['b1']
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, network['W2']) + network['b2']
        A2 = self.sigmoid(Z2)
        return A1, A2

    def compute_loss(self, A2, y):
        m = y.shape[0]
        logprobs = -y * np.log(A2) - (1 - y) * np.log(1 - A2)
        loss = np.sum(logprobs) / m
        return loss

    def fitness(self, network, X, y):
        _, A2 = self.forward_propagation(network, X)
        return self.compute_loss(A2, y)

    def simple_bounds(self, vec):
        return np.clip(vec, -10, 10)

    def optimize(self, X, y):
        for t in range(self.n_iterations):
            for i in range(self.n_bats):
                beta = np.random.uniform(0, 1)
                freq = self.f_min + (self.f_max - self.f_min) * beta
                new_network = self.update_velocity_and_position(self.population[i], self.best_bat, freq)
                new_fitness = self.fitness(new_network, X, y)
                
                if new_fitness <= self.best_fitness and np.random.rand() < self.A:
                    self.population[i] = new_network
                    self.best_fitness = new_fitness
                    self.best_bat = new_network

            self.A *= self.alpha
            self.r *= (1 - np.exp(-self.gamma * t))

        return self.best_bat

    def update_velocity_and_position(self, bat, best_bat, freq):
        new_bat = {}
        for key in bat.keys():
            new_bat[key] = bat[key] + freq * (bat[key] - best_bat[key])
        return new_bat

# Define and train the neural network using the Bat Algorithm
ba = BatAlgorithm(n_bats=10, n_iterations=1000, input_size=X_train.shape[1], hidden_size=10, output_size=1)
best_network = ba.optimize(X_train, y_train)

# Neural Network class with backpropagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(1, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(1, output_size)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, A2, y):
        m = y.shape[0]
        logprobs = -y * np.log(A2) - (1 - y) * np.log(1 - A2)
        loss = np.sum(logprobs) / m
        return loss

    def backward_propagation(self, X, y):
        m = y.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, n_iterations):
        for _ in range(n_iterations):
            self.A2 = self.forward_propagation(X)
            self.backward_propagation(X, y)

# Define and train the neural network using backpropagation
nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=10, output_size=1)
nn.train(X_train, y_train, n_iterations=1000)

# Evaluation functions
def predict(network, X):
    _, A2 = ba.forward_propagation(network, X)
    return (A2 > 0.5).astype(int).ravel()

def evaluate_model(network, X_test, y_test):
    predictions = predict(network, X_test)
    return accuracy_score(y_test, predictions)

# Evaluate Bat Algorithm model
ba_accuracy = evaluate_model(best_network, X_test, y_test)
print(f"Bat Algorithm Model Accuracy: {ba_accuracy}")

# Evaluate Backpropagation model
nn_predictions = (nn.forward_propagation(X_test) > 0.5).astype(int).ravel()
bp_accuracy = accuracy_score(y_test, nn_predictions)
print(f"Backpropagation Model Accuracy: {bp_accuracy}")
