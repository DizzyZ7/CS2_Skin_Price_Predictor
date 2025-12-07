import random
from datetime import datetime, timedelta
import os
import sqlite3
import pandas as pd
from src.database import init_db, insert_skin_price, DATABASE_PATH

def generate_mock_data(num_records=10000):
    """
    Generates mock CS2 skin price data and inserts it into the database.
    This function replaces a real web scraper for demonstration purposes.
    """
    print("Generating mock data...")

    skin_names = [
        "AK-47 | Redline", "M4A4 | Asiimov", "AWP | Gungnir", "Desert Eagle | Blaze",
        "Knife | Karambit (Vanilla)", "Glock-18 | Fade", "AK-47 | Case Hardened",
        "M4A4 | Howl", "AWP | Dragon Lore", "USP-S | Printstream"
    ]
    wear_conditions = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
    rarities = ["Mil-Spec", "Restricted", "Classified", "Covert"]
    stattrak_options = [True, False]

    start_date = datetime.now() - timedelta(days=365 * 2) # 2 years of data

    data = []
    for _ in range(num_records):
        name = random.choice(skin_names)
        wear = random.choice(wear_conditions)
        rarity = random.choice(rarities)
        stattrak = random.choice(stattrak_options)
        float_value = round(random.uniform(0.0001, 0.9999), 4)

        # Base price logic (simplified)
        base_price = 10.0
        if "Asiimov" in name: base_price = 50.0
        if "Dragon Lore" in name: base_price = 10000.0
        if "Gungnir" in name: base_price = 5000.0
        if "Karambit" in name: base_price = 300.0
        if "Howl" in name: base_price = 2000.0
        if "Fade" in name: base_price = 150.0
        if "Blaze" in name: base_price = 200.0
        if "Case Hardened" in name: base_price = 100.0
        if "Printstream" in name: base_price = 70.0

        # Adjust price based on wear
        if wear == "Factory New": base_price *= 1.5
        elif wear == "Minimal Wear": base_price *= 1.2
        elif wear == "Field-Tested": base_price *= 1.0
        elif wear == "Well-Worn": base_price *= 0.8
        elif wear == "Battle-Scarred": base_price *= 0.6

        # Adjust price based on float value (lower float = higher price for FN/MW, lower for BS/WW)
        if wear in ["Factory New", "Minimal Wear"]:
            base_price *= (1 + (1 - float_value) * 0.2) # Better float, higher price
        else:
            base_price *= (1 + float_value * 0.1) # Higher float, slightly higher price (less bad)

        # Adjust price based on stattrak
        if stattrak: base_price *= 1.2

        # Add some random market fluctuation
        price = base_price * random.uniform(0.8, 1.2) # +/- 20% fluctuation
        price = round(price, 2)

        # Random date within the last 2 years
        random_days = random.randint(0, (datetime.now() - start_date).days)
        date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

        data.append({
            'name': name,
            'wear': wear,
            'float_value': float_value,
            'stattrak': stattrak,
            'rarity': rarity,
            'price': price,
            'date': date
        })

    # Clear existing data before inserting new mock data
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM skin_prices")
    conn.commit()
    conn.close()

    for record in data:
        insert_skin_price(record)

    print(f"Successfully generated and inserted {len(data)} mock records into {DATABASE_PATH}")

if __name__ == "__main__":
    init_db() # Ensure DB structure exists
    generate_mock_data(num_records=50000) # Generating 50,000 records for a decent dataset
