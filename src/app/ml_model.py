import joblib
from src.config import BEST_MODEL_FILE


model = None


def load_model():
    global model
    try:
        model = joblib.load(BEST_MODEL_FILE)
        print("Model loaded successfully")
    except FileNotFoundError:
        model = None
        print("Model not found. Train first.")


def predict(input_df):
    if model is None:
        raise ValueError("Model not loaded")

    # Prediction
    pred = model.predict(input_df)[0]

    # Probability of class "1"
    prob = model.predict_proba(input_df)[0][1]

    # Convert numpy values to native python
    try:
        pred = int(pred)
    except:
        pass

    try:
        prob = float(prob)
    except:
        pass

    return pred, prob