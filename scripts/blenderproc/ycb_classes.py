"""
YCB Object Classes Definition
Maps class IDs to object names for the YCB dataset.
"""

from typing import Dict, List

# YCB Object class mapping (103 classes)
YCB_CLASSES: Dict[int, str] = {
    0: "001_chips_can",
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "012_strawberry",
    12: "013_apple",
    13: "014_lemon",
    14: "015_peach",
    15: "016_pear",
    16: "017_orange",
    17: "018_plum",
    18: "019_pitcher_base",
    19: "021_bleach_cleanser",
    20: "022_windex_bottle",
    21: "023_wine_glass",
    22: "024_bowl",
    23: "025_mug",
    24: "026_sponge",
    25: "027-skillet",
    26: "028_skillet_lid",
    27: "029_plate",
    28: "030_fork",
    29: "031_spoon",
    30: "032_knife",
    31: "033_spatula",
    32: "035_power_drill",
    33: "036_wood_block",
    34: "037_scissors",
    35: "038_padlock",
    36: "039_key",
    37: "040_large_marker",
    38: "041_small_marker",
    39: "042_adjustable_wrench",
    40: "043_phillips_screwdriver",
    41: "044_flat_screwdriver",
    42: "046_plastic_bolt",
    43: "047_plastic_nut",
    44: "048_hammer",
    45: "049_small_clamp",
    46: "050_medium_clamp",
    47: "051_large_clamp",
    48: "052_extra_large_clamp",
    49: "053_mini_soccer_ball",
    50: "054_softball",
    51: "055_baseball",
    52: "056_tennis_ball",
    53: "057_racquetball",
    54: "058_golf_ball",
    55: "059_chain",
    56: "061_foam_brick",
    57: "062_dice",
    58: "063-a_marbles",
    59: "063-b_marbles",
    60: "063-c_marbles",
    61: "063-d_marbles",
    62: "063-e_marbles",
    63: "063-f_marbles",
    64: "065-a_cups",
    65: "065-b_cups",
    66: "065-c_cups",
    67: "065-d_cups",
    68: "065-e_cups",
    69: "065-f_cups",
    70: "065-g_cups",
    71: "065-h_cups",
    72: "065-i_cups",
    73: "065-j_cups",
    74: "070-a_colored_wood_blocks",
    75: "070-b_colored_wood_blocks",
    76: "071_nine_hole_peg_test",
    77: "072-a_toy_airplane",
    78: "072-b_toy_airplane",
    79: "072-c_toy_airplane",
    80: "072-d_toy_airplane",
    81: "072-e_toy_airplane",
    82: "072-f_toy_airplane",
    83: "072-g_toy_airplane",
    84: "072-h_toy_airplane",
    85: "072-i_toy_airplane",
    86: "072-j_toy_airplane",
    87: "072-k_toy_airplane",
    88: "073-a_lego_duplo",
    89: "073-b_lego_duplo",
    90: "073-c_lego_duplo",
    91: "073-d_lego_duplo",
    92: "073-e_lego_duplo",
    93: "073-f_lego_duplo",
    94: "073-g_lego_duplo",
    95: "073-h_lego_duplo",
    96: "073-i_lego_duplo",
    97: "073-j_lego_duplo",
    98: "073-k_lego_duplo",
    99: "073-l_lego_duplo",
    100: "073-m_lego_duplo",
    101: "076_timer",
    102: "077_rubiks_cube",
}

# Reverse mapping: name to ID
YCB_NAME_TO_ID: Dict[str, int] = {v: k for k, v in YCB_CLASSES.items()}

# Number of classes
NUM_CLASSES: int = len(YCB_CLASSES)


def get_class_name(class_id: int) -> str:
    """Get class name from ID."""
    return YCB_CLASSES.get(class_id, "unknown")


def get_class_id(class_name: str) -> int:
    """Get class ID from name."""
    return YCB_NAME_TO_ID.get(class_name, -1)


def get_all_class_names() -> List[str]:
    """Get list of all class names in order."""
    return [YCB_CLASSES[i] for i in range(NUM_CLASSES)]


# Material type hints for domain randomization
METALLIC_OBJECTS = [
    "001_chips_can",
    "002_master_chef_can",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "010_potted_meat_can",
    "027-skillet",
    "028_skillet_lid",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "049_small_clamp",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "059_chain",
]

GLOSSY_OBJECTS = [
    "003_cracker_box",
    "004_sugar_box",
    "006_mustard_bottle",
    "008_pudding_box",
    "009_gelatin_box",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle",
]

PLASTIC_OBJECTS = [
    "046_plastic_bolt",
    "047_plastic_nut",
    "061_foam_brick",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "073-g_lego_duplo",
    "073-h_lego_duplo",
    "073-i_lego_duplo",
    "073-j_lego_duplo",
    "073-k_lego_duplo",
    "073-l_lego_duplo",
    "073-m_lego_duplo",
]


def get_material_type(class_name: str) -> str:
    """Get material type for an object class."""
    if class_name in METALLIC_OBJECTS:
        return "metallic"
    elif class_name in GLOSSY_OBJECTS:
        return "glossy"
    elif class_name in PLASTIC_OBJECTS:
        return "plastic"
    else:
        return "default"
