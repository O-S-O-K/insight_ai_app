from tensorflow.keras.models import load_model

MODEL_PATH = "C:/Users/sschley/OneDrive/Documents/GitHub/insight_ai_app/models/cnn_model.h5"

# Load model
model = load_model(MODEL_PATH)

# Print model summary
model.summary()

# Print output shape
print("Model output shape:", model.output_shape)

