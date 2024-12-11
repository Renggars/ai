# Import library
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier, plot_tree # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class']
data = pd.read_csv(url, header=None, names=columns)

# Preprocessing: Encode categorical data
from sklearn.preprocessing import LabelEncoder # type: ignore
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Split dataset
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Cetak X dan y (keseluruhan dataset sebelum split)
print("X (Fitur):")
print(X)  # Menampilkan seluruh data dari X
print("\ny (Target):")
print(y)  # Menampilkan seluruh data dari y

# Cetak hasil split
print("\nX_train (Fitur Training):")
print(X_train)  # Menampilkan 5 baris pertama dari X_train
print("\nX_test (Fitur Testing):")
print(X_test)  # Menampilkan 5 baris pertama dari X_test

print("\ny_train (Target Training):")
print(y_train)  # Menampilkan 5 baris pertama dari y_train
print("\ny_test (Target Testing):")
print(y_test)  # Menampilkan 5 baris pertama dari y_test