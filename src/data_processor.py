import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.database import get_all_skin_prices
from src.utils import map_wear_to_int, map_rarity_to_int

def load_and_preprocess_data():
    """
    Loads data from the database, performs EDA, cleaning, and feature engineering.
    Returns the processed DataFrame.
    """
    print("Loading data from database...")
    raw_data = get_all_skin_prices()
    df = pd.DataFrame(raw_data)

    if df.empty:
        print("No data found in the database. Please run scraper.py first.")
        return None

    print(f"Loaded {len(df)} records.")
    print("Initial DataFrame head:")
    print(df.head())
    print("\nInitial DataFrame info:")
    df.info()

    # --- Data Cleaning ---
    print("\nPerforming data cleaning...")
    # Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Convert 'stattrak' to boolean
    df['stattrak'] = df['stattrak'].astype(bool)

    # Handle potential missing values (for mock data, might not be an issue)
    df.dropna(inplace=True) # Drop rows with any missing values
    print(f"DataFrame size after dropping NaNs: {len(df)}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"DataFrame size after dropping duplicates: {len(df)}")


    # --- Feature Engineering ---
    print("\nPerforming feature engineering...")
    # Map categorical features to numerical
    df['wear_numeric'] = df['wear'].apply(map_wear_to_int)
    df['rarity_numeric'] = df['rarity'].apply(map_rarity_to_int)

    # Extract additional features from 'name' (e.g., weapon type)
    df['weapon_type'] = df['name'].apply(lambda x: x.split('|')[0].strip())
    # One-hot encode weapon_type
    df = pd.get_dummies(df, columns=['weapon_type'], prefix='weapon', drop_first=True)


    # --- Time-based features (Requires sorting by skin and date) ---
    df = df.sort_values(by=['name', 'wear', 'float_value', 'stattrak', 'rarity', 'date'])
    # For simplicity, let's create a unique ID for each specific skin variant
    df['skin_variant_id'] = df['name'] + '_' + df['wear'] + '_' + df['float_value'].astype(str) + '_' + df['stattrak'].astype(str) + '_' + df['rarity']

    # Example: Lagged price (price from previous day for the same skin variant)
    # This feature is more useful for time series forecasting, but can be included as an indicator of recent price
    df['price_lag_1'] = df.groupby('skin_variant_id')['price'].shift(1)
    df['price_lag_7'] = df.groupby('skin_variant_id')['price'].shift(7) # Price 7 days ago

    # Example: Rolling average price (mean price over last N days)
    df['rolling_mean_price_7d'] = df.groupby('skin_variant_id')['price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Fill NaN values created by shift/rolling operations (e.g., for first entries)
    # A simple approach is to fill with the current price or 0/mean. For prediction,
    # consider how you'd get these values for new data points.
    df['price_lag_1'].fillna(df['price'], inplace=True) # Fill first lag with current price
    df['price_lag_7'].fillna(df['price_lag_1'], inplace=True) # Fill 7-day lag with 1-day lag/current
    df['rolling_mean_price_7d'].fillna(df['price'], inplace=True)


    # --- Exploratory Data Analysis (EDA) - for understanding, not directly for model input ---
    print("\nPerforming EDA (visualizations saved to plots/ directory)...")
    os.makedirs('plots', exist_ok=True)

    # Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Distribution of Skin Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.savefig('plots/price_distribution.png')
    # plt.show() # Uncomment to display plots during execution

    # Price vs. Float_value
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='float_value', y='price', hue='wear', data=df.sample(n=min(len(df), 5000), random_state=42), alpha=0.6) # Sample for large datasets
    plt.title('Price vs. Float Value by Wear Condition')
    plt.xlabel('Float Value')
    plt.ylabel('Price ($)')
    plt.savefig('plots/price_vs_float.png')
    # plt.show()

    # Correlation matrix for numerical features
    numerical_features = ['price', 'float_value', 'wear_numeric', 'rarity_numeric', 'price_lag_1', 'rolling_mean_price_7d'] + [col for col in df.columns if 'weapon_' in col]
    corr_matrix = df[numerical_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('plots/correlation_matrix.png')
    # plt.show()

    print("\nData preprocessing and feature engineering complete.")
    print("\nProcessed DataFrame head:")
    print(df.head())
    print("\nProcessed DataFrame info:")
    df.info()

    return df

if __name__ == "__main__":
    processed_df = load_and_preprocess_data()
    if processed_df is not None:
        print("\nShape of processed DataFrame:", processed_df.shape)
        # You can save this processed_df to a CSV for inspection if needed
        # processed_df.to_csv('data/processed_skin_prices.csv', index=False)
