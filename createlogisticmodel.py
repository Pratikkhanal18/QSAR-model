import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset from your local folder
train_data = pd.read_csv(r'C:\Users\HP\Desktop\Machine\Clean\train_data_clean.csv')
test_data = pd.read_csv(r'C:\Users\HP\Desktop\Machine\Clean\test_data_clean.csv')
validation_data = pd.read_csv(r'C:\Users\HP\Desktop\Machine\Clean\validation_data_clean.csv')

# Combine all data to ensure consistent encoding
combined_data = pd.concat([train_data, test_data, validation_data], ignore_index=True)

# Initialize the label encoder
label_encoder = LabelEncoder()

# Encode the 'Label' column to ensure it's a binary or categorical class
combined_data['Label'] = label_encoder.fit_transform(combined_data['Label'])

# Select only the specified features
selected_features = [
    'MolecularWeight', 'NumHeavyAtoms', 'LogP', 'TPSA', 'NumRotatableBonds',
    'NumRings', 'NumHydroxylGroups', 'NumAminoGroups', 'HeavyAtomMolWt', 'ExactMolWt',
    'MaxPartialCharge'
]

# Now split the data back into train, test, and validation sets
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):len(train_data) + len(test_data)]
validation_data = combined_data.iloc[len(train_data) + len(test_data):]

# Proceed with dropping 'Name' columns, separating features and target variables
X_train = train_data[selected_features]  # Only select the important features
y_train = train_data['Label']

X_test = test_data[selected_features]  # Only select the important features
y_test = test_data['Label']

X_validation = validation_data[selected_features]  # Only select the important features
y_validation = validation_data['Label']

# Check for missing or infinite values in the datasets
print("Checking for missing values:")
print(X_train.isnull().sum())

print("Checking for infinite values:")
print((X_train == float('inf')).sum())

# Handle missing or infinite values (if any)
X_train = X_train.fillna(0).replace([float('inf'), -float('inf')], 0)
X_test = X_test.fillna(0).replace([float('inf'), -float('inf')], 0)
X_validation = X_validation.fillna(0).replace([float('inf'), -float('inf')], 0)

# Standardizing the data using RobustScaler (which is more resistant to outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(X_validation)

# Initialize the Logistic Regression model with regularization
logreg_model = LogisticRegression(max_iter=1000)

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'saga']}
grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_model = grid_search.best_estimator_

# Make predictions
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)
y_pred_validation = best_model.predict(X_validation_scaled)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
validation_accuracy = accuracy_score(y_validation, y_pred_validation)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Validation Accuracy: {validation_accuracy}")

print("Classification Report (Test Data):")
print(classification_report(y_test, y_pred_test))

# Save the trained model to a file
joblib.dump(best_model, r'C:\Users\HP\Desktop\Machine\Clean\toxicity_logreg_model.pkl')
