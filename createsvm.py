import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load datasets
train_data = pd.read_csv(r'C:\Users\HP\Desktop\Machine\Clean\train_data_clean.csv')
test_data = pd.read_csv(r'C:\Users\HP\Desktop\Machine\Clean\test_data_clean.csv')
validation_data = pd.read_csv(r'C:\Users\HP\Desktop\Machine\Clean\validation_data_clean.csv')

# Tag datasets
train_data['source'] = 'train'
test_data['source'] = 'test'
validation_data['source'] = 'validation'

# Combine all datasets
combined_data = pd.concat([train_data, test_data, validation_data], ignore_index=True)

# Features to use
selected_features = [
    'MolecularWeight', 'NumHeavyAtoms', 'LogP', 'TPSA', 'NumRotatableBonds',
    'NumRings', 'NumHydroxylGroups', 'NumAminoGroups', 'HeavyAtomMolWt',
    'ExactMolWt', 'MaxPartialCharge'
]

# Drop rows with missing features or labels
combined_data = combined_data.dropna(subset=selected_features + ['Label']).copy()

# Clean invalid values
combined_data[selected_features] = combined_data[selected_features].replace([float('inf'), -float('inf')], 0)
combined_data[selected_features] = combined_data[selected_features].fillna(0)
combined_data['Label'] = combined_data['Label'].astype(int)

# Split by source
df_train = combined_data[combined_data['source'] == 'train']
df_test = combined_data[combined_data['source'] == 'test']
df_val = combined_data[combined_data['source'] == 'validation']

X_train, y_train = df_train[selected_features], df_train['Label']
X_test, y_test = df_test[selected_features], df_test['Label']
X_val, y_val = df_val[selected_features], df_val['Label']

# Check if training data is empty
if X_train.empty or y_train.empty:
    raise ValueError("Training data is empty. Please check the data files.")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Train SVM
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train_scaled, y_train)

# Evaluate
print("\nSVM Model Evaluation:")
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, r'C:\Users\HP\Desktop\Machine\Clean\toxicity_svm_model.pkl')
print("\nModel saved to 'toxicity_svm_model.pkl'")
