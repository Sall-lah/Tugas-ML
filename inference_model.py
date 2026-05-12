import pandas as pd
import time
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sharmajicoder/gen-z-social-media-usage-dataset")

print("Path to dataset files:", path)

# Define the path to your saved model
MODEL_PATH = 'social_media_addiction_model.joblib'

# Load the trained model
loaded_model = joblib.load(MODEL_PATH)

df = pd.read_csv(path)

# Recreate le_target to inverse transform predictions
# This assumes the original y_train (or a representative sample) is available or can be recreated.
# For robustness, you might save/load le_target with the model.
# As a workaround, we'll fit it again with the full 'addiction_level' column from df.
le_target_recreated = LabelEncoder()
le_target_recreated.fit(df['addiction_level'])

# Prepare a sample input (must match the features used for training)
sample_input_df = pd.DataFrame([[20, 3.0, 3, 25.0, 0, 50, 0]],
                                 columns=['age', 'daily_usage_hours', 'num_platforms_used', 'avg_session_minutes', 'night_usage', 'mental_health_score', 'screen_time_before_sleep'])

# Measure inference time
start_time = time.time()
prediction = loaded_model.predict(sample_input_df)
end_time = time.time()

inference_time_ms = (end_time - start_time) * 1000 # Convert to milliseconds

# Map the encoded prediction back to the original label
predicted_label = le_target_recreated.inverse_transform(prediction)

print(f"Inference time for a single sample: {inference_time_ms:.3f} ms")
print(f"Predicted addiction level: {predicted_label[0]}")
