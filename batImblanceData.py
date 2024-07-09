import numpy as np
import pandas as pd

class BatAlgorithm:
    def __init__(self, data, target_column, population_size=20, iterations=100):
        self.data = data
        self.target_column = target_column
        self.target_index = data.columns.get_loc(target_column)
        self.population_size = population_size
        self.iterations = iterations

    def initialize_population(self):
        self.population = np.random.choice([0, 1], size=(self.population_size, len(self.data)))
    
    def fitness(self, sample):
        class_counts = np.bincount(sample[:, self.target_index])
        if len(class_counts) == 1:  # In case all samples belong to one class
            return -np.inf
        return -np.abs(class_counts[0] - class_counts[1])

    def optimize(self):
        self.initialize_population()
        best_sample = self.data.values
        best_fitness = self.fitness(best_sample)
        
        for _ in range(self.iterations):
            for i, bat in enumerate(self.population):
                sample = self.data[bat == 1].values
                if len(sample) == 0:  # Avoid empty samples
                    continue
                
                new_fitness = self.fitness(sample)
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_sample = sample
        
        return pd.DataFrame(best_sample, columns=self.data.columns)

# Load the imbalanced data
data = pd.read_excel('generatedFiles/imbalanced_dataNew.xlsx')

# Apply the Bat Algorithm to balance the data
ba = BatAlgorithm(data, target_column='Target', population_size=20, iterations=100)
balanced_data = ba.optimize()

# Save the balanced data
balanced_data.to_excel('generatedFiles/balanced_dataNew.xlsx', index=False)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to evaluate model accuracy
def evaluate_accuracy(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Load the original imbalanced data
data = pd.read_excel('generatedFiles/imbalanced_dataNew.xlsx')

# Evaluate accuracy on imbalanced data
imbalanced_accuracy = evaluate_accuracy(data, 'Target')
print(f"Accuracy on imbalanced data: {imbalanced_accuracy}")

# Evaluate accuracy on balanced data
balanced_accuracy = evaluate_accuracy(balanced_data, 'Target')
print(f"Accuracy on balanced data: {balanced_accuracy}")
