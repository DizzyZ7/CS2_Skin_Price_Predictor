import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from src.data_processor import load_and_preprocess_data

MODEL_DIR = 'models'
MODEL_FILENAME = 'price_predictor_model.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def train_model():
    """
    Loads processed data, splits it, trains a RandomForestRegressor,
    evaluates it, and saves the trained model.
    """
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_preprocess_data()
    if df is None:
        print("Failed to load and preprocess data. Exiting model training.")
        return

    # Define features (X) and target (y)
    # Exclude non-numeric/non-feature columns
    features = [
        'float_value',
        'stattrak',
        'wear_numeric',
        'rarity_numeric',
        'price_lag_1',
        'rolling_mean_price_7d'
    ]
    # Add one-hot encoded weapon types
    weapon_cols = [col for col in df.columns if 'weapon_' in col]
    features.extend(weapon_cols)

    # Filter out any features that might not exist in a small mock dataset
    actual_features = [f for f in features if f in df.columns]
    X = df[actual_features]
    y = df['price']

    print(f"\nFeatures used for training: {actual_features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Initialize and train the RandomForestRegressor model
    print("\nTraining RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Also save the list of features used for consistency during prediction
    joblib.dump(actual_features, os.path.join(MODEL_DIR, 'model_features.pkl'))
    print(f"Features list saved to {os.path.join(MODEL_DIR, 'model_features.pkl')}")

if __name__ == "__main__":
    train_model()
