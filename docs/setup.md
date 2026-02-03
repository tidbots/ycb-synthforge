# セットアップガイド

このドキュメントでは、YCB SynthForgeの環境構築手順を説明します。

## 必要環境

- Docker & Docker Compose v2+ (`docker compose` コマンドを使用)
  - 注意: 旧版の `docker-compose` (v1.x) は非対応
- NVIDIA GPU (CUDA対応)
- NVIDIA Container Toolkit
- 推奨: RTX 3090/4090 (VRAM 24GB)

## Dockerイメージのビルド

```bash
# 全イメージをビルド
docker compose build

# 個別ビルド
docker compose build blenderproc
docker compose build yolo26_train
```

## YOLO26重みのダウンロード

```bash
# 利用可能なモデル一覧を表示
python scripts/download_weights.py --list

# デフォルト (nano + small) をダウンロード
python scripts/download_weights.py

# 特定のモデルをダウンロード
python scripts/download_weights.py --models yolo26n yolo26s yolo26m

# 全モデルをダウンロード
python scripts/download_weights.py --all

# 強制的に再ダウンロード
python scripts/download_weights.py --models yolo26m --force
```

### モデル一覧

| モデル | パラメータ数 | サイズ | 用途 |
|--------|------------|--------|------|
| yolo26n | 2.6M | ~5 MB | 最速・エッジデバイス向け |
| yolo26s | 9.4M | ~19 MB | バランス型 |
| yolo26m | 20.1M | ~40 MB | 推奨・汎用 |
| yolo26l | 25.3M | ~49 MB | 高精度 |
| yolo26x | 56.9M | ~109 MB | 最高精度 |

## YCB 3Dモデルのダウンロード

**重要**: 本プロジェクトでは`google_16k`形式を基本とし、一部オブジェクトで`tsdf`形式を使用します。

```bash
# オブジェクト一覧を表示
python scripts/download_ycb_models.py --list

# カテゴリ一覧を表示
python scripts/download_ycb_models.py --list-categories

# 全オブジェクトをgoogle_16k形式でダウンロード
python scripts/download_ycb_models.py --all --format google_16k

# tsdf形式もダウンロード（一部オブジェクトで必要）
python scripts/download_ycb_models.py --all --format berkeley

# カテゴリ指定でダウンロード
python scripts/download_ycb_models.py --category food fruit kitchen --format google_16k

# 特定オブジェクトのみダウンロード
python scripts/download_ycb_models.py --objects 003_cracker_box 005_tomato_soup_can --format google_16k

# 強制的に再ダウンロード
python scripts/download_ycb_models.py --all --format google_16k --force
```

### 形式一覧

| 形式 | 説明 | 用途 |
|------|------|------|
| google_16k | 16kポリゴン、高品質テクスチャ | ✅ 基本形式（72オブジェクト） |
| tsdf | TSDF再構成メッシュ | ✅ 一部オブジェクトで使用（13オブジェクト） |
| google_64k | 64kポリゴン、より高解像度 | |
| google_512k | 512kポリゴン、最高解像度 | |
| poisson | poisson再構成（非推奨） | ❌ テクスチャ破損あり |

### tsdf形式のマテリアル修正

tsdf形式のOBJファイルはマテリアル参照が欠落しているため、初回ダウンロード後に修正が必要です:

```bash
# tsdf形式のOBJファイルを修正（usemtl行を追加）
docker compose run --rm fix_tsdf_materials
```

**注意**: 修正前のファイルは `.obj.backup` として自動保存されます。

## CC0テクスチャのダウンロード

[ambientCG](https://ambientcg.com/)からPBRテクスチャをダウンロード（CC0ライセンス）。

```bash
# カテゴリ一覧を表示
python scripts/download_cctextures.py --list-categories

# デフォルト100テクスチャをダウンロード
python scripts/download_cctextures.py

# カテゴリ指定でダウンロード
python scripts/download_cctextures.py --category floor wall table

# プレフィックス指定 (Wood*, Metal* 各20枚)
python scripts/download_cctextures.py --prefix Wood Metal --limit 20

# 特定テクスチャをダウンロード
python scripts/download_cctextures.py --textures Wood001 Metal002 Tiles005

# 高解像度でダウンロード (1K/2K/4K/8K)
python scripts/download_cctextures.py --resolution 4K

# オンラインで検索
python scripts/download_cctextures.py --search Marble --limit 30
```

### カテゴリ一覧

| カテゴリ | 用途 | プレフィックス |
|---------|------|---------------|
| floor | 床 | Wood, WoodFloor, Tiles, Marble, Concrete |
| wall | 壁 | Bricks, PaintedPlaster, Wallpaper, Facade |
| table | テーブル | Wood, Metal, Plastic, Marble, Granite |
| metal | 金属 | Metal, MetalPlates, DiamondPlate, Rust |
| fabric | 布 | Fabric, Leather, Carpet, Wicker |
| natural | 自然 | Ground, Grass, Rock, Gravel, Sand |
| industrial | 工業 | Asphalt, Concrete, CorrugatedSteel |

## 次のステップ

セットアップが完了したら、[パイプライン実行](pipeline.md)に進んでください。
