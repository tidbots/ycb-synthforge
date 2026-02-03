# YCB Object Class List

This document describes the 85 YCB object classes available in YCB SynthForge.

## Class List

### Food & Beverages (10)

| ID | Object Name | Format |
|----|-------------|--------|
| 001 | chips_can | tsdf* |
| 002 | master_chef_can | google_16k |
| 003 | cracker_box | google_16k |
| 004 | sugar_box | google_16k |
| 005 | tomato_soup_can | google_16k |
| 006 | mustard_bottle | google_16k |
| 007 | tuna_fish_can | google_16k |
| 008 | pudding_box | google_16k |
| 009 | gelatin_box | google_16k |
| 010 | potted_meat_can | google_16k |

### Fruits (8)

| ID | Object Name | Format |
|----|-------------|--------|
| 011 | banana | google_16k |
| 012 | strawberry | google_16k |
| 013 | apple | google_16k |
| 014 | lemon | google_16k |
| 015 | peach | google_16k |
| 016 | pear | google_16k |
| 017 | orange | google_16k |
| 018 | plum | google_16k |

### Kitchen Items (11)

| ID | Object Name | Format |
|----|-------------|--------|
| 019 | pitcher_base | google_16k |
| 021 | bleach_cleanser | google_16k |
| 022 | windex_bottle | google_16k |
| 023 | wine_glass | tsdf* |
| 024 | bowl | google_16k |
| 025 | mug | google_16k |
| 026 | sponge | google_16k |
| 028 | skillet_lid | google_16k |
| 029 | plate | google_16k |
| 030 | fork | google_16k |
| 031 | spoon | google_16k |
| 032 | knife | google_16k |
| 033 | spatula | google_16k |

### Tools (14)

| ID | Object Name | Format |
|----|-------------|--------|
| 035 | power_drill | google_16k |
| 036 | wood_block | google_16k |
| 037 | scissors | google_16k |
| 038 | padlock | google_16k |
| 040 | large_marker | google_16k |
| 041 | small_marker | tsdf* |
| 042 | adjustable_wrench | google_16k |
| 043 | phillips_screwdriver | google_16k |
| 044 | flat_screwdriver | google_16k |
| 048 | hammer | google_16k |
| 049 | small_clamp | tsdf* |
| 050 | medium_clamp | google_16k |
| 051 | large_clamp | google_16k |
| 052 | extra_large_clamp | google_16k |

### Sports (6)

| ID | Object Name | Format |
|----|-------------|--------|
| 053 | mini_soccer_ball | google_16k |
| 054 | softball | google_16k |
| 055 | baseball | google_16k |
| 056 | tennis_ball | google_16k |
| 057 | racquetball | google_16k |
| 058 | golf_ball | tsdf* |

### Others (36)

| ID | Object Name | Format |
|----|-------------|--------|
| 059 | chain | google_16k |
| 061 | foam_brick | google_16k |
| 062 | dice | tsdf* |
| 063-a | marbles | google_16k |
| 063-b | marbles | google_16k |
| 065-a~j | cups (10 types) | google_16k |
| 070-a | colored_wood_blocks | google_16k |
| 070-b | colored_wood_blocks | google_16k |
| 071 | nine_hole_peg_test | google_16k |
| 072-a | toy_airplane | google_16k |
| 073-a~f | lego_duplo (6 types) | google_16k |
| 073-g~m | lego_duplo (7 types) | tsdf* |
| 076 | timer | tsdf* |
| 077 | rubiks_cube | google_16k |

**\*** Uses tsdf format

## Mesh Format Selection

| Format | Number of Objects | Description |
|--------|-------------------|-------------|
| google_16k | 72 | High-quality textures, primary format |
| tsdf | 13 | Objects with quality issues in google_16k |

### Objects Using tsdf Format (13)

```
001_chips_can
041_small_marker
049_small_clamp
058_golf_ball
062_dice
073-g_lego_duplo
073-h_lego_duplo
073-i_lego_duplo
073-j_lego_duplo
073-k_lego_duplo
073-l_lego_duplo
073-m_lego_duplo
076_timer
```

## Excluded Objects (6)

The following objects are excluded due to mesh quality issues in both google_16k/tsdf formats:

```
072-b_toy_airplane
072-c_toy_airplane
072-d_toy_airplane
072-e_toy_airplane
072-h_toy_airplane
072-k_toy_airplane
```

## Changing Object Format

To change the format for specific objects, configure in `scripts/blenderproc/generate_dataset.py`:

```python
# Objects to completely exclude
EXCLUDED_OBJECTS = {
    "072-b_toy_airplane",
    "problematic_object_name",  # Add
}

# Objects to use tsdf format (when google_16k has issues)
USE_TSDF_FORMAT = {
    "001_chips_can",
    "problematic_object_name",  # Add
}
```

## Related Documentation

- [Adding Custom Models](custom-models-e.md)
- [Troubleshooting](troubleshooting-e.md)
