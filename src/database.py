import sqlite3
import os

DATABASE_DIR = 'data'
DATABASE_NAME = 'cs2_skin_prices.db'
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

def init_db():
    """Initializes the SQLite database: creates the directory and the table if they don't exist."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skin_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            wear TEXT NOT NULL,
            float_value REAL,
            stattrak INTEGER, -- 0 for False, 1 for True
            rarity TEXT NOT NULL,
            price REAL NOT NULL,
            date TEXT NOT NULL -- YYYY-MM-DD
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

def insert_skin_price(skin_data):
    """Inserts a single skin price record into the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO skin_prices (name, wear, float_value, stattrak, rarity, price, date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        skin_data['name'],
        skin_data['wear'],
        skin_data['float_value'],
        1 if skin_data['stattrak'] else 0,
        skin_data['rarity'],
        skin_data['price'],
        skin_data['date']
    ))
    conn.commit()
    conn.close()

def get_all_skin_prices():
    """Fetches all skin price records from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, wear, float_value, stattrak, rarity, price, date FROM skin_prices')
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

if __name__ == "__main__":
    init_db()
    # Пример использования:
    # skin_data = {
    #     'name': 'AK-47 | Redline',
    #     'wear': 'Field-Tested',
    #     'float_value': 0.25,
    #     'stattrak': False,
    #     'rarity': 'Covert',
    #     'price': 22.50,
    #     'date': '2023-10-26'
    # }
    # insert_skin_price(skin_data)
    # print(get_all_skin_prices())
