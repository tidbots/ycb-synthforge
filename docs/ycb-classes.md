# YCBオブジェクトクラス一覧

このドキュメントでは、YCB SynthForgeで利用可能な85種類のYCBオブジェクトクラスを説明します。

## クラス一覧

### 食品・飲料 (10個)

| ID | オブジェクト名 | 形式 |
|----|---------------|------|
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

### 果物 (8個)

| ID | オブジェクト名 | 形式 |
|----|---------------|------|
| 011 | banana | google_16k |
| 012 | strawberry | google_16k |
| 013 | apple | google_16k |
| 014 | lemon | google_16k |
| 015 | peach | google_16k |
| 016 | pear | google_16k |
| 017 | orange | google_16k |
| 018 | plum | google_16k |

### キッチン用品 (11個)

| ID | オブジェクト名 | 形式 |
|----|---------------|------|
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

### 工具 (14個)

| ID | オブジェクト名 | 形式 |
|----|---------------|------|
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

### スポーツ (6個)

| ID | オブジェクト名 | 形式 |
|----|---------------|------|
| 053 | mini_soccer_ball | google_16k |
| 054 | softball | google_16k |
| 055 | baseball | google_16k |
| 056 | tennis_ball | google_16k |
| 057 | racquetball | google_16k |
| 058 | golf_ball | tsdf* |

### その他 (36個)

| ID | オブジェクト名 | 形式 |
|----|---------------|------|
| 059 | chain | google_16k |
| 061 | foam_brick | google_16k |
| 062 | dice | tsdf* |
| 063-a | marbles | google_16k |
| 063-b | marbles | google_16k |
| 065-a~j | cups (10種) | google_16k |
| 070-a | colored_wood_blocks | google_16k |
| 070-b | colored_wood_blocks | google_16k |
| 071 | nine_hole_peg_test | google_16k |
| 072-a | toy_airplane | google_16k |
| 073-a~f | lego_duplo (6種) | google_16k |
| 073-g~m | lego_duplo (7種) | tsdf* |
| 076 | timer | tsdf* |
| 077 | rubiks_cube | google_16k |

**\*** tsdf形式を使用

## メッシュ形式の選択

| 形式 | 使用オブジェクト数 | 説明 |
|------|------------------|------|
| google_16k | 72個 | 高品質テクスチャ、基本形式 |
| tsdf | 13個 | google_16kで品質問題があるオブジェクト |

### tsdf形式を使用するオブジェクト (13個)

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

## 除外オブジェクト (6個)

以下のオブジェクトはgoogle_16k/tsdf両形式でメッシュ品質に問題があるため除外:

```
072-b_toy_airplane
072-c_toy_airplane
072-d_toy_airplane
072-e_toy_airplane
072-h_toy_airplane
072-k_toy_airplane
```

## オブジェクトの形式変更

特定のオブジェクトの形式を変更する場合、`scripts/blenderproc/generate_dataset.py`で設定します:

```python
# 完全に除外するオブジェクト
EXCLUDED_OBJECTS = {
    "072-b_toy_airplane",
    "問題のあるオブジェクト名",  # 追加
}

# tsdf形式を使用するオブジェクト（google_16kに問題がある場合）
USE_TSDF_FORMAT = {
    "001_chips_can",
    "問題のあるオブジェクト名",  # 追加
}
```

## 関連ドキュメント

- [カスタムモデルの追加](custom-models.md)
- [トラブルシューティング](troubleshooting.md)
