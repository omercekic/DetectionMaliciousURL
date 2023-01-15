import joblib


def predict(data):
    model = joblib.load("rf_model.sav")
    result = model.predict(data)
    return result
