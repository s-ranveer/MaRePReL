# We would describe the different resource tiles and buildings in the environment as follows:
# Resource tiles:
# 1. Wood
wood_small = {
    "type": "wood",
    "count": 5
}
wood_medium = {
    "type": "wood",
    "count": 10
}
wood_large = {
    "type": "wood",
    "count": 20
}
# 2. Stone
stone_small = {
    "type": "stone",
    "count": 5
}
stone_medium = {
    "type": "stone",
    "count": 10
}
stone_large = {
    "type": "stone",
    "count": 20
}
# 3. Food
food_small = {
    "type": "food",
    "count": 5
}
food_medium = {
    "type": "food",
    "count": 10
}
food_large = {
    "type": "food",
    "count": 20
}
# 4. Water
water_small = {
    "type": "water",
    "count": 5
}
water_medium = {
    "type": "water",
    "count": 10
}
water_large = {
    "type": "water",
    "count": 20
}

# Gold
gold_small = {
    "type": "gold",
    "count": 5
}
gold_medium = {
    "type": "gold",
    "count": 10
}
gold_large = {
    "type": "gold",
    "count": 20
}

# Buildings:
# 1. House
house = {
    "type": "house",
    "cost": {
        "wood": 5,
        "stone": 5
    },
}
# 2. Mansion
mansion= {
    "type": "mansion",
    "cost": {
        "wood": 10,
        "stone": 5
    },
}

# 3. Castle
castle = {
    "type": "castle",
    "cost": {
        "wood": 10,
        "stone": 10
    },
}

# Map the types to their layout representation
type_to_layout = {
    "wood": "P",
    "stone": "S",
    "house": "H",
    "mansion": "M",
    "castle": "C"
}

MAX_INVENTORY_PER_RESOURCE = 5
BUILDINGS_COUNT = 3
RESOURCE_TILES_COUNT = 4