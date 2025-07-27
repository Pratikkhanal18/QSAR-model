# Import necessary libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Function to convert SMILES to a 1024-bit fingerprint
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 2 is the radius, 1024 bits for the fingerprint
    else:
        return None

# Load the data (adjust the file paths as necessary)
train_data = pd.read_csv("C:/Users/HP/Desktop/Machine/Clean/train_data_clean.csv")
test_data = pd.read_csv("C:/Users/HP/Desktop/Machine/Clean/test_data_clean.csv")
validation_data = pd.read_csv("C:/Users/HP/Desktop/Machine/Clean/validation_data_clean.csv")

# Preprocess SMILES - convert to molecular fingerprints
train_data['Fingerprint'] = train_data['SMILES'].apply(smiles_to_fingerprint)
train_data = train_data.dropna(subset=['Fingerprint'])  # Drop rows where SMILES conversion failed

# Prepare features (fingerprints) and target labels (Label)
X_train = list(train_data['Fingerprint'])
y_train = train_data['Label']  # 'Label' column used for toxicity

# Convert fingerprints to a format suitable for machine learning
X_train = np.array([list(fp) for fp in X_train])

# Train-Test Split (for validation purposes)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model on the validation set
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")
print(f"Classification Report: \n{classification_report(y_val, y_pred)}")

# Plot the feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances in the RandomForest Classifier')
plt.show()

# Evaluate on the test data
X_test = list(test_data['SMILES'].apply(smiles_to_fingerprint))
X_test = np.array([list(fp) for fp in X_test])
y_test = test_data['Label']  # 'Label' column used for toxicity

y_test_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")

# Save the trained model for later use
joblib.dump(clf, 'toxicity_forest_model.pkl')

# Make predictions for new compounds (optional)
def predict_new_compound(smiles_code):
    # Load the saved model
    clf_loaded = joblib.load('toxicity_forest_model.pkl')

    # Convert the new SMILES code to fingerprint
    new_fingerprint = smiles_to_fingerprint(smiles_code)
    new_fingerprint = np.array([list(new_fingerprint)])

    # Make prediction
    prediction = clf_loaded.predict(new_fingerprint)
    return prediction[0]

# Example usage: Predict toxicity for a new compound
new_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example SMILES (Aspirin)
prediction = predict_new_compound(new_smiles)
print(f"Predicted Toxicity for new compound: {prediction}")
