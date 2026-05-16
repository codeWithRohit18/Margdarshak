import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("output/handelEDA.csv")

# Remove unwanted column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Convert text to numbers
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define X and y
X = df.drop(columns=['ai_risk_category'])
y = df['ai_risk_category']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred, average='weighted')

# Recall
recall = recall_score(y_test, y_pred, average='weighted')

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean())

# Metrics display
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Performance Graph
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values)
plt.title("Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

# Confusion Matrix Visualization
plt.matshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()