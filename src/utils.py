WEAR_MAPPING = {
    "Factory New": 0,
    "Minimal Wear": 1,
    "Field-Tested": 2,
    "Well-Worn": 3,
    "Battle-Scarred": 4
}

RARITY_MAPPING = {
    "Consumer Grade": 0,
    "Industrial Grade": 1,
    "Mil-Spec": 2,
    "Restricted": 3,
    "Classified": 4,
    "Covert": 5,
    "Contraband": 6 # Если такие существуют и нужны
}

def map_wear_to_int(wear_string):
    """Maps wear string to an integer."""
    return WEAR_MAPPING.get(wear_string, -1) # -1 for unknown wear

def map_rarity_to_int(rarity_string):
    """Maps rarity string to an integer."""
    return RARITY_MAPPING.get(rarity_string, -1) # -1 for unknown rarity

# Можно добавить другие утилиты, например, для нормализации строк
