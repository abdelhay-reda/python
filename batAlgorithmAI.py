import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset directly from the UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
data = pd.read_csv(url)

# Extract features and target variables
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (Parkinson's or not)
    ])
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class BatAlgorithm:
    def __init__(self, n_bats, n_iterations, alpha, gamma, f_min, f_max):
        self.n_bats = n_bats
        self.n_iterations = n_iterations
        self.alpha = alpha  # loudness
        self.gamma = gamma  # pulse rate
        self.f_min = f_min
        self.f_max = f_max
    
    def optimize(self, model, X_train, y_train):
        # Flatten the weights and biases of the model into a 1D array
        initial_weights = self.flatten_weights(model.get_weights())
        n_weights = len(initial_weights)
        
        # Initialize bat population
        bat_positions = np.random.uniform(-1, 1, (self.n_bats, n_weights))
        bat_velocities = np.zeros((self.n_bats, n_weights))
        bat_frequencies = np.zeros(self.n_bats)
        bat_loudness = np.ones(self.n_bats)
        bat_pulse_rates = np.ones(self.n_bats)
        
        # Initialize best solution
        best_bat_position = bat_positions[0]
        best_fitness = self.evaluate_fitness(model, X_train, y_train, best_bat_position)
        
        for t in range(self.n_iterations):
            for i in range(self.n_bats):
                # Update frequency
                bat_frequencies[i] = self.f_min + (self.f_max - self.f_min) * np.random.rand()
                
                # Update velocity
                bat_velocities[i] += (bat_positions[i] - best_bat_position) * bat_frequencies[i]
                
                # Update position
                new_position = bat_positions[i] + bat_velocities[i]
                
                # Apply bounds
                new_position = np.clip(new_position, -1, 1)
                
                # Evaluate fitness
                fitness = self.evaluate_fitness(model, X_train, y_train, new_position)
                
                # Check if the new solution is better
                if fitness < best_fitness and np.random.rand() < bat_loudness[i]:
                    best_bat_position = new_position.copy()
                    best_fitness = fitness
                
                # Check pulse rate
                if np.random.rand() < bat_pulse_rates[i]:
                    bat_positions[i] = best_bat_position
                
                # Reduce loudness
                bat_loudness[i] *= self.alpha
                bat_pulse_rates[i] = self.gamma * (1 - np.exp(-self.gamma * t))
        
        # Set the best found weights to the model
        model.set_weights(self.reshape_weights(best_bat_position, model.get_weights()))
        return best_fitness
    
    def flatten_weights(self, weights):
        return np.concatenate([w.flatten() for w in weights])
    
    def reshape_weights(self, flat_weights, model_weights):
        reshaped_weights = []
        offset = 0
        for weight in model_weights:
            shape = weight.shape
            size = np.prod(shape)
            reshaped_weights.append(flat_weights[offset:offset + size].reshape(shape))
            offset += size
        return reshaped_weights
    
    def evaluate_fitness(self, model, X_train, y_train, flat_weights):
        model.set_weights(self.reshape_weights(flat_weights, model.get_weights()))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
        return loss

# Example usage
if __name__ == "__main__":
    # Initialize Bat Algorithm
    bat_algorithm = BatAlgorithm(n_bats=10, n_iterations=100, alpha=0.9, gamma=0.9, f_min=0, f_max=2)
    
    # Optimize MLP weights using Bat Algorithm
    best_fitness = bat_algorithm.optimize(model, X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
