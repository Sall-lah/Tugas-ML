import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration --- #
# Set the path to your dataset CSV file. 
# Adjust this if your dataset is in a different location when running locally.
DATASET_PATH = 'genz_social_media_usage_1M.csv'
MODEL_SAVE_PATH = 'social_media_addiction_model.joblib' # Model will be saved in the same directory as the script

print(f"Loading data from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset not found at {DATASET_PATH}. Please update DATASET_PATH to the correct location.")
    exit()

# --- Data Loading --- #
df = pd.read_csv(DATASET_PATH)

# --- Data Preprocessing --- #
print("Starting data preprocessing...")
X = df.drop('addiction_level', axis=1)
y = df['addiction_level']

# Drop categorical features (object dtype) from X
columns_to_drop = X.select_dtypes(include=['object']).columns.tolist()
if columns_to_drop:
    X = X.drop(columns=columns_to_drop)
    print(f"Dropped categorical columns from features: {columns_to_drop}")
else:
    print("No categorical columns (object dtype) found in features to drop.")

# Encode target variable 'addiction_level'
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Split data
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Data preprocessing complete.")

# --- Model Training --- #
print("Starting model training...")
model = lgb.LGBMClassifier(objective='multiclass', num_class=len(le_target.classes_), random_state=42)
model.fit(X_train, y_train_encoded)
print("Model training complete.")

# --- Model Evaluation (Optional) --- #
print("Evaluating model on test set...")
y_pred_encoded = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Accuracy: {accuracy:.4f}")

# For the classification report, we can inverse transform to get original labels
y_test_original = le_target.inverse_transform(y_test_encoded)
y_pred_original = le_target.inverse_transform(y_pred_encoded)
print("\nClassification Report:")
print(classification_report(y_test_original, y_pred_original))

# --- Model Saving --- #
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# It's good practice to also save the LabelEncoder if you need to use it for inverse transforming predictions later
joblib.dump(le_target, 'label_encoder.joblib')
print("LabelEncoder saved to label_encoder.joblib")