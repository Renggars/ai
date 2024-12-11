import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class']
data = pd.read_csv(url, header=None, names=columns)

# Preprocessing: Encode categorical data
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Split dataset
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ukuran dataset
data_size = len(data)
train_size = len(X_train)
test_size = len(X_test)

# Membuat array untuk visualisasi
data_indices = np.arange(data_size)
train_indices = np.arange(train_size)
test_indices = np.arange(test_size) + train_size

# Plot
plt.figure(figsize=(10, 5))
plt.bar(data_indices, np.ones(data_size), color='gray', label='Original Data', alpha=0.6)
plt.bar(train_indices, np.ones(train_size), color='blue', label='Training Set', alpha=0.8)
plt.bar(test_indices, np.ones(test_size), color='orange', label='Testing Set', alpha=0.8)
plt.xlabel('Data Indices')
plt.ylabel('Presence')
plt.title('Dataset Splitting Visualization')
plt.legend()
plt.show()
