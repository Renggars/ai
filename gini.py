import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

# Decision Tree with Gini Index
print("\n=== Decision Tree with Gini Index ===")
dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_gini.fit(X_train, y_train)

# Predict and Evaluate
y_pred_gini = dt_gini.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_gini))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gini))
print(le.classes_)

# Visualize the Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_gini, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.title("Decision Tree with Gini Index")
plt.show()

# Visualize Confusion Matrix
def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

cm_gini = confusion_matrix(y_test, y_pred_gini)
plot_confusion_matrix(cm_gini, classes=le.classes_, title="Confusion Matrix - Gini Index")
