import pickle
import numpy as np
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Create label map
label_set = sorted(set(labels))  # ['A', 'B', ..., 'Z']
label_map = {i: label for i, label in enumerate(label_set)}
inv_label_map = {v: k for k, v in label_map.items()}  # For encoding

# Encode labels as integers
encoded_labels = np.array([inv_label_map[label] for label in labels])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, shuffle=True, stratify=encoded_labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"\n✅ Accuracy: {score * 100:.2f}%")
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_predict, target_names=label_map.values()))

# Save model + label map
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_map': label_map}, f)

print('\n💾 Model and label map saved to "model.p"')
