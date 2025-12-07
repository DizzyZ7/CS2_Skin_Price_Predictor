import joblib
import pandas as pd
import numpy as np
import os
import argparse
from src.utils import map_wear_to_int, map_rarity_to_int
from src.data_processor import load_and_preprocess_data
from datetime import datetime

MODEL_DIR = 'models'
MODEL_FILENAME = 'price_predictor_model.pkl'
FEATURES_FILENAME = 'model_features.pkl'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
FEATURES_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)

def load_model_and_features():
    """Loads the trained model and the list of features it was trained on."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run model_trainer.py first.")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}. Please run model_trainer.py first.")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

def predict_skin_price(model, features, skin_data, df_historical):
    """
    Makes a price prediction for a single skin based on input data.
    Needs historical data to compute lagged features.
    """
    # Create a DataFrame for the single skin prediction
    # Ensure all required features for the model are present
    input_df = pd.DataFrame([skin_data])

    # Convert stattrak to boolean and then to int for model consistency
    input_df['stattrak'] = input_df['stattrak'].astype(bool).astype(int)

    # Map wear and rarity to numeric
    input_df['wear_numeric'] = input_df['wear'].apply(map_wear_to_int)
    input_df['rarity_numeric'] = input_df['rarity'].apply(map_rarity_to_int)

    # Extract weapon type for one-hot encoding
    input_df['weapon_type'] = input_df['name'].apply(lambda x: x.split('|')[0].strip())
    input_df = pd.get_dummies(input_df, columns=['weapon_type'], prefix='weapon', drop_first=True)

    # Reindex input_df to match the order and presence of features during training
    # Create a DataFrame with all possible weapon_type columns (from historical data)
    # and fill with 0, then select only those used in training
    all_weapon_cols_from_training = [f for f in features if f.startswith('weapon_')]
    for col in all_weapon_cols_from_training:
        if col not in input_df.columns:
            input_df[col] = 0 # Add missing weapon columns with 0

    # Ensure float_value is numeric
    input_df['float_value'] = pd.to_numeric(input_df['float_value'])

    # --- Compute lagged features for the current skin_data ---
    # This is a simplification. In a real-world scenario, you'd need the *actual*
    # historical prices for this specific skin variant up to "yesterday"
    # from your live database. For this example, we'll try to find a proxy from
    # the loaded historical data.

    # Find matching skin variants in historical data (if any)
    skin_variant_name = skin_data['name']
    skin_variant_wear = skin_data['wear']
    skin_variant_float = skin_data['float'] # Use the float value directly
    skin_variant_stattrak = skin_data['stattrak']
    skin_variant_rarity = skin_data['rarity']

    # Filter historical data for this specific skin variant
    # This is approximate as float values can be very specific
    filtered_historical = df_historical[
        (df_historical['name'] == skin_variant_name) &
        (df_historical['wear'] == skin_variant_wear) &
        (df_historical['stattrak'] == skin_variant_stattrak) &
        (df_historical['rarity'] == skin_variant_rarity)
    ]
    # For float_value, we might need to find the closest one or use a range.
    # For simplicity, let's just find records with float values close to the input.
    # A more robust solution would track each unique float or average within float ranges.
    filtered_historical = filtered_historical[
        (filtered_historical['float_value'] >= skin_variant_float - 0.05) &
        (filtered_historical['float_value'] <= skin_variant_float + 0.05)
    ].sort_values(by='date', ascending=False)


    # Default values if no historical data is found
    price_lag_1 = input_df['price_lag_1'].iloc[0] if 'price_lag_1' in input_df.columns and not pd.isna(input_df['price_lag_1'].iloc[0]) else 0.0
    rolling_mean_price_7d = input_df['rolling_mean_price_7d'].iloc[0] if 'rolling_mean_price_7d' in input_df.columns and not pd.isna(input_df['rolling_mean_price_7d'].iloc[0]) else 0.0


    if not filtered_historical.empty:
        # Use the most recent prices from historical data for lagged features
        most_recent_price = filtered_historical['price'].iloc[0]
        price_lag_1 = most_recent_price # Use most recent as 1-day lag proxy

        # Calculate a rolling mean from historical data
        # This assumes the 'date' in historical is sorted and continuous enough
        rolling_mean_price_7d = filtered_historical['price'].head(7).mean()
        if pd.isna(rolling_mean_price_7d): rolling_mean_price_7d = most_recent_price

    input_df['price_lag_1'] = price_lag_1
    input_df['rolling_mean_price_7d'] = rolling_mean_price_7d

    # Select and order features for prediction
    X_predict = input_df[features]

    predicted_price = model.predict(X_predict)[0]
    return predicted_price

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CS2 skin price.")
    parser.add_argument("--name", type=str, required=True, help="Full name of the skin (e.g., AK-47 | Redline)")
    parser.add_argument("--wear", type=str, required=True, choices=list(map_wear_to_int.keys()), help="Wear condition (e.g., Factory New, Field-Tested)")
    parser.add_argument("--float", type=float, required=True, help="Float value of the skin (e.g., 0.15)")
    parser.add_argument("--stattrak", type=lambda x: x.lower() == 'true', required=False, default=False, help="Is it a StatTrak™ skin? (True/False)")
    parser.add_argument("--rarity", type=str, required=True, choices=list(map_rarity_to_int.keys()), help="Rarity of the skin (e.g., Covert, Mil-Spec)")

    args = parser.parse_args()

    try:
        model, features = load_model_and_features()
        print("Model and features loaded successfully.")

        # Load historical data to help compute lagged features for prediction
        # This is important because `data_processor.py` adds these features.
        # For a live system, you'd query your database for recent prices of THIS specific skin.
        historical_df = load_and_preprocess_data()
        if historical_df is None:
            raise Exception("Could not load historical data for lagged feature computation.")

        skin_data = {
            'name': args.name,
            'wear': args.wear,
            'float_value': args.float,
            'float': args.float, # Keep for easier filtering
            'stattrak': args.stattrak,
            'rarity': args.rarity
        }

        predicted_price = predict_skin_price(model, features, skin_data, historical_df)
        print(f"Предсказанная цена для {args.name} ({args.wear}) Float {args.float:.4f} (StatTrak: {args.stattrak}, Rarity: {args.rarity}): ${predicted_price:.2f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run `python src/scraper.py` and `python src/model_trainer.py` first.")
    except Exception as e:
        print(f"An error occurred: {e}")
