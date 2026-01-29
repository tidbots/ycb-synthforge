# YCB Object Detection with YOLO26 and Domain Randomization

## プロジェクト概要
YCB Objectデータセットの3DモデルをBlenderProcでフォトリアリスティックに合成し、ドメインランダマイゼーションを適用してロバストな物体検出モデルを構築する。最新のYOLO26を使用し、全てDockerコンテナ内で動作し、独自データの追加学習にも対応する。

**学習戦略**: ゼロから学習するのではなく、YOLO26のCOCO事前学習済みモデルを初期値として、YCBオブジェクトの検出タスクにファインチューニング（追加学習）する。

## 目標
1. BlenderProcで高品質な合成データを生成（COCO形式）
2. ドメインランダマイゼーションでSim-to-Real転移を改善
3. **YOLO26**（最新版、2026年1月14日リリース）の**事前学習済みモデルをファインチューニング**
4. 合成データ + 実画像（少量）でロバストなYCB検出器を構築
5. 独自データの追加学習パイプラインを構築
6. 全プロセスをDocker化して再現性を確保

## YOLO26について
**YOLO26は2026年1月にリリースされた最新のYOLOモデル**です。主な特徴：
- **エンドツーエンドのNMSフリー推論**: 後処理が不要で高速
- **DFL削除**: エクスポートを簡素化し、エッジ互換性を拡張
- **CPU推論が最大43%高速化**: エッジデバイスに最適
- **MuSGDオプティマイザ**: より安定したトレーニングと高速な収束
- **デュアルヘッドアーキテクチャ**: 1対1ヘッド（NMSフリー）と1対多ヘッド（NMS必要）

公式ドキュメント: https://docs.ultralytics.com/ja/models/yolo26/

## ファインチューニング戦略（Transfer Learning）

### 基本方針
**ゼロから学習するのではなく、事前学習済みモデルを活用**

- **ベースモデル**: YOLO26のCOCO事前学習済みモデル（80クラス）
  - `yolo26m.pt` を推奨（精度と速度のバランスが良い）
  - COCOデータセットで既に一般的な物体検出能力を獲得済み
  
- **追加学習するクラス**: YCB Object 77クラス
  - 001_chips_can
  - 002_master_chef_can
  - 003_cracker_box
  - 004_sugar_box
  - ... など77オブジェクト
  
- **学習データ**:
  1. **合成データ（メイン）**: BlenderProcで生成した10,000〜20,000枚
  2. **実画像（少量）**: 可能であれば100〜500枚を追加
     - Sim-to-Real gapを埋める
     - モデルのロバスト性を大幅に向上

### ファインチューニングの利点
1. **学習時間の大幅短縮**
   - ゼロからの学習: 数日〜1週間
   - ファインチューニング: 数時間〜1日
   
2. **少ないデータで高精度**
   - 事前学習で得た「物体とは何か」の知識を活用
   - YCB特有の特徴だけを追加学習
   
3. **汎化性能の向上**
   - COCOで学習した多様な背景・照明への対応力
   - 過学習のリスク低減

### データ構成の推奨バランス

#### パターンA: 合成データのみ（最小構成）
```
訓練データ: 10,000枚（合成）
検証データ: 2,000枚（合成）
テストデータ: 実画像を別途用意して評価
```

#### パターンB: 合成 + 実画像（推奨）⭐
```
訓練データ: 
  - 合成: 9,500枚
  - 実画像: 500枚（データ拡張で2,000枚相当）
検証データ:
  - 合成: 1,800枚
  - 実画像: 200枚
テストデータ: 
  - 実画像: 500枚（別途収集）
```

#### パターンC: 大規模（最高精度）
```
訓練データ:
  - 合成: 19,000枚
  - 実画像: 1,000枚（データ拡張で4,000枚相当）
検証データ:
  - 合成: 3,800枚
  - 実画像: 200枚
テストデータ:
  - 実画像: 1,000枚
```

### 実画像の収集方法
実画像を少量でも混ぜることで、Sim-to-Real転移が劇的に改善されます。

**収集のポイント**:
- YCBオブジェクトを様々な環境で撮影
- 多様な照明条件（自然光、蛍光灯、LED）
- 異なる背景（机、棚、床、雑多な環境）
- 様々な角度・距離から撮影
- スマートフォンのカメラでOK

**アノテーション方法**:
1. [LabelImg](https://github.com/HumanSignal/labelImg) - 手動アノテーション
2. [CVAT](https://github.com/opencv/cvat) - チーム作業向け
3. [Roboflow](https://roboflow.com/) - クラウドベース（無料枠あり）

### 学習の段階的アプローチ

#### Stage 1: 合成データでベース構築
```bash
# YOLO26m（COCO事前学習済み）からスタート
docker-compose run yolo26_train python3 scripts/training/train_yolo26.py \
  --data data/synthetic/yolo/dataset.yaml \
  --weights weights/yolo26m.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --name ycb_synthetic_v1
```

#### Stage 2: 実画像を追加してファインチューニング
```bash
# Stage 1で学習したモデルをベースに、実画像を追加
docker-compose run yolo26_train python3 scripts/training/train_yolo26.py \
  --data data/combined/dataset.yaml \
  --weights outputs/trained_models/ycb_synthetic_v1/weights/best.pt \
  --epochs 30 \
  --batch 16 \
  --imgsz 640 \
  --freeze 10 \
  --name ycb_combined_v1
```

#### Stage 3: 全体を微調整
```bash
# 全層を解凍して微調整
docker-compose run yolo26_train python3 scripts/training/train_yolo26.py \
  --data data/combined/dataset.yaml \
  --weights outputs/trained_models/ycb_combined_v1/weights/best.pt \
  --epochs 20 \
  --batch 16 \
  --imgsz 640 \
  --lr0 0.0001 \
  --name ycb_final_v1
```

## プロジェクト構成

```
project_root/
├── claude.md                          # このファイル
├── docker/
│   ├── Dockerfile.blenderproc        # データ生成用
│   ├── Dockerfile.yolo26             # YOLO26学習・推論用
│   └── docker-compose.yml
├── models/
│   └── ycb/                          # YCB 3Dモデル（ダウンロード済み）
│       ├── 001_chips_can/
│       ├── 002_master_chef_can/
│       ├── 003_cracker_box/
│       └── ...（77オブジェクト）
├── resources/
│   └── cctextures/                   # 背景テクスチャ（ダウンロード済み）
│       ├── Wood001/
│       ├── Wood002/
│       ├── Concrete/
│       ├── Metal/
│       └── ...
├── weights/
│   ├── yolo26n.pt                    # ベースモデル（ダウンロード済み）
│   ├── yolo26s.pt
│   ├── yolo26m.pt
│   ├── yolo26l.pt
│   └── yolo26x.pt
├── scripts/
│   ├── blenderproc/
│   │   ├── generate_dataset.py       # メイン生成スクリプト
│   │   ├── config.yaml               # 生成パラメータ設定
│   │   ├── scene_setup.py            # シーン構築
│   │   ├── lighting.py               # 照明設定
│   │   ├── camera.py                 # カメラエフェクト
│   │   └── materials.py              # マテリアル設定
│   ├── data_processing/
│   │   ├── coco_to_yolo.py           # COCO→YOLO変換
│   │   ├── train_val_split.py        # データ分割
│   │   └── merge_datasets.py         # 合成+実画像の統合
│   ├── training/
│   │   ├── train_yolo26.py           # YOLO26学習スクリプト
│   │   ├── finetune_yolo26.py        # ファインチューニング専用
│   │   └── train_config.yaml         # 学習設定
│   ├── evaluation/
│   │   ├── evaluate.py               # モデル評価
│   │   └── inference.py              # 推論実行
│   └── custom_data/
│       ├── prepare_custom_data.py    # 独自データ準備
│       └── finetune.py               # ファインチューニング
├── data/
│   ├── synthetic/                    # 合成データ
│   │   ├── coco/
│   │   │   ├── images/
│   │   │   └── annotations.json
│   │   └── yolo/
│   │       ├── images/
│   │       │   ├── train/
│   │       │   └── val/
│   │       ├── labels/
│   │       │   ├── train/
│   │       │   └── val/
│   │       └── dataset.yaml
│   ├── real/                         # 実画像データ
│   │   ├── raw/                      # 生画像（収集したまま）
│   │   ├── annotated/                # アノテーション済み
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── dataset.yaml
│   ├── combined/                     # 合成+実画像の統合データセット
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── dataset.yaml
│   └── custom/                       # 独自データ用
│       ├── images/
│       └── labels/
├── outputs/
│   ├── trained_models/               # 学習済みモデル
│   │   ├── stage1_synthetic/        # Stage 1: 合成データのみ
│   │   ├── stage2_combined/         # Stage 2: 合成+実画像
│   │   └── stage3_final/            # Stage 3: 最終調整
│   ├── predictions/                  # 推論結果
│   └── metrics/                      # 評価メトリクス
└── README.md
```

## データ生成要件（ドメインランダマイゼーション）

### 1. 背景のランダム化
**床テクスチャ**
- 木材フローリング（明/中/暗）
- タイル（セラミック、大理石）
- コンクリート（新品/劣化）
- カーペット（短毛/長毛）
- リノリウム

**棚テクスチャ**
- 金属棚（スチール、アルミ）
- 木製棚（パイン、オーク、合板）
- プラスチック棚

**壁テクスチャ**
- 白壁（マット/半光沢）
- コンクリート壁
- レンガ（赤/白）
- 壁紙パターン
- 塗装壁（様々な色）

**ランダム化パラメータ**
- テクスチャスケール: 0.5x 〜 2.0x
- 色相シフト: ±30°
- 明度調整: 0.7 〜 1.3
- 汚れ/摩耗オーバーレイ: 0% 〜 40%

### 2. 照明のランダム化
**色温度**
- 範囲: 2700K（電球色） 〜 6500K（昼光色）
- ステップ: 500K刻み

**光源配置**
- 点光源: 1〜4個をランダム配置
- エリアライト: 天井照明シミュレート
- 方向: elevation 30°〜90°、azimuth 0°〜360°

**強さ**
- 範囲: 100W 〜 1000W相当
- 環境光: 0.1 〜 0.5

**影**
- ハードシャドウ（点光源）
- ソフトシャドウ（エリアライト）
- 影の濃さ: 0.3 〜 0.9

**HDRIマップ**
- 屋内環境: オフィス、倉庫、工場
- ランダム回転: 0°〜360°

### 3. カメラエフェクトのランダム化
**露出**
- EV値: -1.5 〜 +1.5
- 自動露出シミュレーション

**ISO/ゲイン**
- ISO相当: 100、200、400、800、1600、3200
- ノイズレベル: ISO値に比例

**ホワイトバランス**
- 色温度オフセット: ±1000K
- Tint調整: ±0.1

**ブレ/ボケ**
- モーションブラー: 0 〜 0.05（シャッタースピード相当）
- 被写界深度: F値 1.8 〜 11.0
- フォーカス距離: 物体中心 ± 0.2m

**レンズ歪み**
- 樽型歪み: -0.1 〜 0
- 糸巻き型歪み: 0 〜 0.1
- ビネット: 0 〜 0.3

**その他**
- クロマティックアバレーション（色収差）
- レンズフレア（まれに）

### 4. Clutter Objects（雑多物体）
**配置数**: 3 〜 15個/シーン

**物体候補**
- 文房具（ペン、ノート、クリップ）
- 食器（マグカップ、皿、カトラリー）
- 工具（ドライバー、レンチ）
- 電子機器（マウス、ケーブル、充電器）
- 日用品（ティッシュ箱、リモコン）

**配置ルール**
- YCBオブジェクトの周囲にランダム配置
- 一部はYCBオブジェクトと重なり/接触
- 物理シミュレーションで自然な配置
- **ラベル付けなし**（背景として扱う）

**目的**
- オクルージョン耐性向上
- 誤検出の削減
- 実環境の雑然とした状況を再現

### 5. 反射・材質のランダム化
**金属缶（Master Chef Can等）**
- Metallic: 0.8 〜 1.0
- Roughness: 0.05 〜 0.3（鏡面〜半光沢）
- Specular: 0.5 〜 1.0
- 環境マッピング強度: 0.5 〜 1.0

**光沢箱（Cracker Box等）**
- Metallic: 0.0
- Roughness: 0.1 〜 0.5
- Specular: 0.3 〜 0.7
- クリアコートレイヤー（印刷物再現）

**プラスチック製品**
- Roughness: 0.2 〜 0.6
- Subsurface Scattering（半透明性）

**PBRパラメータ**
- Base Color: テクスチャ + 色相シフト ±10°
- Normal Map強度: 0.5 〜 1.5
- Fresnel効果を適用

**"ハイライト地獄"再現**
- 複数光源 + 高反射 = 複雑なハイライトパターン
- ブルーム効果（グレア）
- レンズフレア

### 6. 物体配置のランダム化
**視点**
- カメラ距離: 0.4m 〜 2.0m
- Elevation角: 10° 〜 70°
- Azimuth角: 0° 〜 360°（全方位）

**物体の向き**
- X軸回転: 0° 〜 360°
- Y軸回転: 0° 〜 360°
- Z軸回転: 0° 〜 360°

**スケール**
- 基本は実サイズ
- 誤差: ±5%（製造ばらつき再現）

**配置方法**
- 物理シミュレーション（重力落下）
- 平面/棚/積み重ね
- 一部は宙に浮いた状態も許容

## ドメインランダマイゼーション戦略

### Level 1: 基礎ランダム化
- 照明の色温度・強さ
- カメラの露出・ホワイトバランス
- 物体の位置・回転

### Level 2: 中程度ランダム化
- 背景テクスチャの種類
- 複数光源の配置
- ノイズ・ブラーの追加
- Clutterオブジェクトの配置

### Level 3: 高度なランダム化
- 材質パラメータ（Roughness, Metallic）
- レンズ歪み・色収差
- 複雑な反射・ハイライト
- 環境HDRIのバリエーション

**実装方針**: 段階的に適用し、精度への影響を評価

## データ生成パラメータ

### 数量
- **訓練用**: 10,000 〜 20,000枚
- **検証用**: 2,000 〜 4,000枚
- **テスト用**: 1,000枚（別途実画像推奨）

### 画像設定
- **解像度**: 640×640（YOLO26デフォルト）または 1280×1280
- **フォーマット**: PNG（可逆圧縮）
- **ビット深度**: 8-bit RGB

### 物体設定
- **YCBオブジェクト/画像**: 2 〜 8個
- **Clutterオブジェクト/画像**: 3 〜 15個
- **オクルージョン率**: 0% 〜 60%

### レンダリング
- **サンプル数**: 128 〜 256（ノイズ低減）
- **デノイジング**: 有効
- **レンダリングエンジン**: Cycles（物理ベース）

## Docker環境

### Dockerfile.blenderproc
```dockerfile
FROM ubuntu:22.04

# Blender + BlenderProc環境
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    wget unzip

# BlenderProcインストール
RUN pip3 install blenderproc

# 必要なPythonライブラリ
RUN pip3 install numpy opencv-python pycocotools pyyaml

WORKDIR /workspace

CMD ["python3", "scripts/blenderproc/generate_dataset.py"]
```

### Dockerfile.yolo26
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# YOLO26環境
# ultralyticsの最新版をインストール（YOLO26対応版）
RUN pip install --upgrade ultralytics opencv-python pycocotools pyyaml tqdm

# 評価用ライブラリ
RUN pip install scikit-learn matplotlib seaborn pandas

WORKDIR /workspace

CMD ["bash"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  blenderproc:
    build:
      context: .
      dockerfile: docker/Dockerfile.blenderproc
    volumes:
      - ./models:/workspace/models:ro
      - ./resources:/workspace/resources:ro
      - ./scripts:/workspace/scripts:ro
      - ./data:/workspace/data
    environment:
      - DISPLAY=${DISPLAY}
      - BLENDERPROC_THREADS=8
    shm_size: '8gb'

  yolo26_train:
    build:
      context: .
      dockerfile: docker/Dockerfile.yolo26
    volumes:
      - ./data:/workspace/data:ro
      - ./weights:/workspace/weights
      - ./scripts:/workspace/scripts:ro
      - ./outputs:/workspace/outputs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
    shm_size: '16gb'

  yolo26_inference:
    build:
      context: .
      dockerfile: docker/Dockerfile.yolo26
    volumes:
      - ./data:/workspace/data:ro
      - ./outputs:/workspace/outputs
      - ./scripts:/workspace/scripts:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
```

## ワークフロー

### Phase 1: データ生成（合成データ）
```bash
# BlenderProcコンテナでデータ生成
docker-compose run blenderproc python3 scripts/blenderproc/generate_dataset.py \
  --config scripts/blenderproc/config.yaml \
  --output data/synthetic/coco \
  --num_scenes 10000
```

### Phase 2: データ変換（COCO → YOLO）
```bash
# COCO → YOLO変換
docker-compose run yolo26_train python3 scripts/data_processing/coco_to_yolo.py \
  --coco_json data/synthetic/coco/annotations.json \
  --output_dir data/synthetic/yolo \
  --train_ratio 0.85
```

### Phase 3: 実画像の準備（オプションだが強く推奨）⭐
```bash
# 1. 実画像を data/real/raw/ に配置
# 2. LabelImgやCVATでアノテーション
# 3. YOLO形式に変換
docker-compose run yolo26_train python3 scripts/data_processing/prepare_real_data.py \
  --raw_dir data/real/raw \
  --output_dir data/real/annotated
```

### Phase 4: データセット統合（合成 + 実画像）
```bash
# 合成データと実画像を統合
docker-compose run yolo26_train python3 scripts/data_processing/merge_datasets.py \
  --synthetic data/synthetic/yolo \
  --real data/real/annotated \
  --output data/combined \
  --real_ratio 0.05  # 実画像を5%混ぜる
```

### Phase 5: Stage 1 - 合成データでファインチューニング
```bash
# YOLO26のCOCO事前学習済みモデルからスタート
docker-compose run yolo26_train python3 scripts/training/train_yolo26.py \
  --data data/synthetic/yolo/dataset.yaml \
  --weights weights/yolo26m.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --project outputs/trained_models \
  --name stage1_synthetic
```

### Phase 6: Stage 2 - 実画像を追加してファインチューニング
```bash
# Stage 1のモデルをベースに、実画像を追加
docker-compose run yolo26_train python3 scripts/training/train_yolo26.py \
  --data data/combined/dataset.yaml \
  --weights outputs/trained_models/stage1_synthetic/weights/best.pt \
  --epochs 30 \
  --batch 16 \
  --imgsz 640 \
  --freeze 10 \
  --project outputs/trained_models \
  --name stage2_combined
```

### Phase 7: Stage 3 - 全体を微調整（オプション）
```bash
# 全層を解凍して微調整
docker-compose run yolo26_train python3 scripts/training/train_yolo26.py \
  --data data/combined/dataset.yaml \
  --weights outputs/trained_models/stage2_combined/weights/best.pt \
  --epochs 20 \
  --batch 16 \
  --imgsz 640 \
  --lr0 0.0001 \
  --project outputs/trained_models \
  --name stage3_final
```

### Phase 8: 評価
```bash
# 最終モデルを評価
docker-compose run yolo26_inference python3 scripts/evaluation/evaluate.py \
  --model outputs/trained_models/stage3_final/weights/best.pt \
  --data data/combined/dataset.yaml \
  --output outputs/metrics/final
```

### Phase 9: 推論実行
```bash
# テスト画像で推論
docker-compose run yolo26_inference python3 scripts/evaluation/inference.py \
  --model outputs/trained_models/stage3_final/weights/best.pt \
  --source data/test_images/ \
  --output outputs/predictions/ \
  --conf 0.5
```

## 独自データ追加学習パイプライン

### YCBモデルをベースに独自クラスを追加

YCBで学習したモデルをベースに、さらに独自のオブジェクトクラスを追加することができます。

### 準備
1. 独自データを `data/custom/images/` に配置
2. アノテーション（YOLO形式）を `data/custom/labels/` に配置

### クラスID調整
```python
# YCB: 0-76 (77クラス)
# 独自データ: 77以降に追加
# 例: 77=custom_object_1, 78=custom_object_2
```

### データセット統合
```yaml
# data/custom/dataset.yaml
path: /workspace/data/custom
train: images/train
val: images/val

names:
  0: 002_master_chef_can
  1: 003_cracker_box
  # ... YCB 0-76
  77: custom_object_1
  78: custom_object_2
  79: custom_object_3
```

### ファインチューニング（YCBモデル → 独自クラス追加）
```bash
# YCBで学習済みのモデルをベースに独自クラスを追加
docker-compose run yolo26_train python3 scripts/custom_data/finetune.py \
  --base_model outputs/trained_models/stage3_final/weights/best.pt \
  --custom_data data/custom/dataset.yaml \
  --epochs 50 \
  --freeze 10 \
  --batch 16 \
  --imgsz 640 \
  --name ycb_plus_custom
```

### 学習戦略

**オプションA: YCBの知識を保持しながら追加**
```python
# 最初の10層を凍結して、YCBの特徴抽出能力を維持
freeze = 10
```

**オプションB: YCBと独自クラスを同時に最適化**
```python
# 全層を解凍して、両方のクラスに対して最適化
freeze = 0  # すべてのレイヤーを学習
lr0 = 0.0001  # 低い学習率でゆっくり調整
```

### データ混合の推奨比率
独自クラスを追加する際は、YCBデータも一部混ぜることを推奨：

```
訓練データ:
  - YCB（合成+実画像）: 2,000枚（20%）
  - 独自クラス: 8,000枚（80%）
検証データ:
  - YCB: 200枚（20%）
  - 独自クラス: 800枚（80%）
```

これにより、YCBの検出性能を維持しながら、独自クラスも高精度で検出できます。

## YOLO26の特徴と使用方法

### デュアルヘッドアーキテクチャ
YOLO26は2つの予測ヘッドを持ちます：

**1対1ヘッド（デフォルト、NMSフリー）**
- エンドツーエンドで予測を生成
- 後処理（NMS）が不要
- より高速な推論
- 出力形式: `(N, 300, 6)` - 1画像あたり最大300個の検出

**1対多ヘッド（NMS必要）**
- 従来のYOLOと同様
- わずかに高い精度の可能性
- 出力形式: `(N, nc + 4, 8400)`

### ヘッドの切り替え方法
```python
from ultralytics import YOLO

model = YOLO("yolo26m.pt")

# 1対1ヘッド使用（デフォルト、NMSフリー）
results = model.predict("image.jpg")  # end2end=True（デフォルト）
metrics = model.val(data="dataset.yaml")
model.export(format="onnx")

# 1対多ヘッド使用（NMS必要）
results = model.predict("image.jpg", end2end=False)
metrics = model.val(data="dataset.yaml", end2end=False)
model.export(format="onnx", end2end=False)
```

### モデルサイズの選択
- **yolo26n.pt**: 最小・最速（2.4M params）
- **yolo26s.pt**: 小型（9.5M params）
- **yolo26m.pt**: 中型・推奨（20.4M params）⭐
- **yolo26l.pt**: 大型（24.8M params）
- **yolo26x.pt**: 最大・最高精度（55.7M params）

### MuSGDオプティマイザ
YOLO26は新しいMuSGDオプティマイザを導入：
- SGDとMuonのハイブリッド
- より安定したトレーニング
- 高速な収束
- LLMトレーニングの技術を応用

## コーディング規約

### Python
- PEP 8準拠
- 型ヒント必須（Python 3.10+）
- Docstring: Google形式
- ロガー使用（print禁止）

### 設定管理
- YAMLファイルで一元管理
- 環境変数で機密情報管理
- デフォルト値を明示

### エラーハンドリング
- try-except-finally構造
- ログに詳細なエラー情報
- 異常終了時はクリーンアップ

### 再現性
- 乱数シード固定（`random.seed(42)`, `np.random.seed(42)`）
- BlenderProc内部のシードも設定
- 全パラメータをログ保存

## BlenderProcスクリプト設計

### generate_dataset.py
```python
"""
メインデータ生成スクリプト
- YCBモデル読み込み
- シーン構築（背景、照明、カメラ）
- ドメインランダマイゼーション適用
- レンダリング + COCOアノテーション出力
"""
```

### config.yaml
```yaml
# データ生成設定
scene:
  num_images: 10000
  objects_per_scene: [2, 8]
  clutter_per_scene: [3, 15]

camera:
  resolution: [640, 640]
  distance: [0.4, 2.0]
  elevation: [10, 70]
  
lighting:
  num_lights: [1, 4]
  color_temp: [2700, 6500]
  
materials:
  metallic_range: [0.8, 1.0]
  roughness_range: [0.05, 0.3]
```

### scene_setup.py
```python
"""
シーン構築モジュール
- 背景（床/壁/棚）生成
- テクスチャ適用
- Clutterオブジェクト配置
- 物理シミュレーション
"""
```

### lighting.py
```python
"""
照明設定モジュール
- 複数光源配置
- 色温度調整
- HDRI環境マップ
- 影の設定
"""
```

### camera.py
```python
"""
カメラエフェクトモジュール
- 露出・ISO調整
- ホワイトバランス
- モーションブラー・被写界深度
- レンズ歪み
"""
```

### materials.py
```python
"""
マテリアル設定モジュール
- PBRパラメータ設定
- 反射・ツヤの調整
- テクスチャマッピング
"""
```

## YOLO26学習設定

### dataset.yaml（YCBのみ）
```yaml
path: /workspace/data/synthetic/yolo
train: images/train
val: images/val

# YCB Object 77クラス
names:
  0: 002_master_chef_can
  1: 003_cracker_box
  2: 004_sugar_box
  3: 005_tomato_soup_can
  4: 006_mustard_bottle
  # ... 76まで
```

### dataset.yaml（合成 + 実画像）
```yaml
path: /workspace/data/combined
train: images/train
val: images/val

# YCB Object 77クラス
names:
  0: 002_master_chef_can
  1: 003_cracker_box
  2: 004_sugar_box
  # ... 76まで
```

### train_config.yaml（基本設定）
```yaml
# ベースモデル（COCO事前学習済み）
model: yolo26m.pt  # nano, s, m, l, x から選択

# Stage 1: 合成データでファインチューニング
epochs: 50
batch: 16
imgsz: 640
optimizer: MuSGD  # YOLO26の新オプティマイザ
lr0: 0.001  # 初期学習率
weight_decay: 0.0005

# データ拡張（合成データには控えめに）
augmentation:
  hsv_h: 0.01  # 色相調整（小さめ）
  hsv_s: 0.5   # 彩度調整
  hsv_v: 0.3   # 明度調整
  degrees: 5.0  # 回転（小さめ）
  translate: 0.05  # 平行移動（小さめ）
  scale: 0.3   # スケール変換
  flipud: 0.0  # 上下反転なし
  fliplr: 0.5  # 左右反転
  mosaic: 0.5  # モザイク拡張（控えめ）
  mixup: 0.0   # ミックスアップなし
```

### train_config_stage2.yaml（実画像追加時）
```yaml
# Stage 2: 実画像を追加してファインチューニング
# ベース: Stage 1で学習したモデル
epochs: 30
batch: 16
imgsz: 640
optimizer: MuSGD
lr0: 0.0005  # Stage 1より低い学習率
weight_decay: 0.0005
freeze: 10  # 最初の10層を凍結

# データ拡張（実画像には強めに）
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  translate: 0.1
  scale: 0.5
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0  # モザイク拡張を強化
  mixup: 0.1   # ミックスアップを追加
```

### train_config_stage3.yaml（最終調整）
```yaml
# Stage 3: 全体を微調整
# ベース: Stage 2で学習したモデル
epochs: 20
batch: 16
imgsz: 640
optimizer: MuSGD
lr0: 0.0001  # さらに低い学習率
weight_decay: 0.0005
freeze: 0  # 全層を解凍

# データ拡張（バランス型）
augmentation:
  hsv_h: 0.01
  hsv_s: 0.6
  hsv_v: 0.35
  degrees: 7.0
  translate: 0.075
  scale: 0.4
  flipud: 0.0
  fliplr: 0.5
  mosaic: 0.7
  mixup: 0.05
```

## 評価メトリクス

### 検出性能
- mAP@0.5 (COCO metric)
- mAP@0.5:0.95 (COCO metric)
- mAP@0.5:0.95(e2e) - エンドツーエンド評価（YOLO26特有）
- Precision / Recall
- F1スコア

### クラス別性能
- AP per class
- Confusion Matrix

### 速度
- FPS（推論速度）
- レイテンシ
- CPU推論速度（YOLO26の強み）

### 出力形式
- JSON（詳細メトリクス）
- CSV（サマリー）
- 可視化グラフ（Matplotlib）

## 期待する出力

### データ生成
- COCO形式JSONファイル
- 10,000+ 枚の合成画像（PNG）
- 生成ログ（パラメータ記録）

### 学習
- 学習済みモデル（`.pt`ファイル）
- 学習曲線（Loss, mAP）
- 検証結果サマリー

### 推論
- バウンディングボックス付き画像
- 検出結果JSON
- 評価レポート（PDF/HTML）

## トラブルシューティング

### BlenderProc関連
- **Out of Memory**: `shm_size`を増やす、バッチサイズ削減
- **レンダリング遅延**: サンプル数削減、解像度低下
- **GPUエラー**: CPUレンダリングに切り替え

### YOLO26学習関連
- **過学習**: Augmentation強化、Early Stopping
- **精度低下**: Syn-to-Real gap → ドメインランダム強化
- **クラス不均衡**: クラスウェイト調整

### Docker関連
- **NVIDIA Runtime未検出**: `nvidia-docker`インストール確認
- **ボリュームマウント失敗**: パス・権限確認

## 拡張ロードマップ

### 短期（1-2週間）
- [ ] BlenderProcスクリプト実装
- [ ] 少量の合成データで動作確認（100枚）
- [ ] YOLO26ファインチューニングパイプライン確立
- [ ] Stage 1: 合成データでの学習完了

### 中期（1ヶ月）
- [ ] 大量合成データ生成（10,000枚）
- [ ] 実画像の収集とアノテーション（100〜500枚）
- [ ] Stage 2: 合成+実画像でのファインチューニング
- [ ] ドメインランダマイゼーション最適化
- [ ] 実画像でのSim-to-Real評価

### 長期（2-3ヶ月）
- [ ] Stage 3: 最終調整と評価
- [ ] 独自データの追加学習
- [ ] マルチモーダル（RGB-D）対応検討
- [ ] リアルタイム推論最適化
- [ ] YOLO26のセグメンテーション版（yolo26-seg）への拡張
- [ ] エッジデバイス（Jetson、Raspberry Pi）への展開

## リソース

### 公式ドキュメント
- **YOLO26**: https://docs.ultralytics.com/ja/models/yolo26/
- BlenderProc: https://github.com/DLR-RM/BlenderProc
- Ultralytics: https://docs.ultralytics.com/
- COCO Format: https://cocodataset.org/#format-data

### データセット
- YCB Object Models: http://ycbbenchmarks.com/
- CC0 Textures: https://ambientcg.com/
- Poly Haven HDRI: https://polyhaven.com/

### 論文
- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World (Tobin et al., 2017)
- Muon Optimizer: https://arxiv.org/abs/2502.16982

## Claudeへの指示

### コード生成時
- 詳細なコメント・Docstringを記述
- 型ヒントを必ず付与
- エラーハンドリングを適切に実装
- ロガー出力で進捗確認可能にする

### BlenderProcスクリプト
- APIドキュメントを参照しながら実装
- ドメインランダマイゼーションを段階的に適用
- レンダリング結果を可視化して確認
- YCBモデルのパス（models/ycb/）を正しく参照
- 背景テクスチャ（resources/cctextures/）を活用

### YOLO26スクリプト
- **ファインチューニングを基本戦略として実装**:
  - 常にCOCO事前学習済みモデル（yolo26m.pt等）から開始
  - ゼロからの学習は行わない
  - 段階的な学習（Stage 1 → 2 → 3）を推奨
- **YOLO26の最新機能を活用**:
  - デフォルトでNMSフリーの1対1ヘッド使用
  - MuSGDオプティマイザを活用
  - エンドツーエンド評価メトリクスを記録
- **実画像の活用を促す**:
  - 合成データのみで学習後、実画像を追加
  - データ拡張を適切に設定
  - Sim-to-Real gapを意識した設計
- Ultralyticsライブラリの最新API使用
- 学習過程をWandB等でトラッキング（オプション）
- 推論結果を可視化
- weights/配下の事前学習済みモデルを活用

### データ処理スクリプト
- **合成データと実画像の統合**:
  - 適切な比率で混合（例: 95% 合成 + 5% 実画像）
  - クラスバランスを考慮
  - データ拡張の違いを考慮
- **アノテーションの品質管理**:
  - バウンディングボックスの妥当性チェック
  - クラスIDの一貫性確認
  - 画像とラベルの対応確認

### Docker設定
- マルチステージビルドで最適化
- キャッシュを活用してビルド時間短縮
- セキュリティベストプラクティス遵守
- NVIDIA GPUサポートを適切に設定

### デバッグ
- 小規模データで動作確認してからスケールアップ
- ログを詳細に記録
- 中間結果を可視化して検証
- 各Stageの学習曲線を比較

### パスの参照
- YCB 3Dモデル: `models/ycb/` 配下
- 背景テクスチャ: `resources/cctextures/` 配下
- YOLO26ベースモデル: `weights/` 配下
- 実画像: `data/real/` 配下
- これらのパスはすべて絶対パスまたはプロジェクトルートからの相対パスで参照

### 最適化の優先順位
1. **転移学習**: 事前学習済みモデルの活用が最優先
2. **実データの活用**: 少量でも実画像を混ぜることでSim-to-Real gapを大幅改善
3. **精度**: ドメインランダマイゼーションで合成データの質を向上
4. **速度**: YOLO26のNMSフリー推論を活用
5. **スケーラビリティ**: Docker化で再現性と展開容易性を確保

### ファインチューニングのベストプラクティス
- **層の凍結**: 初期段階では下位層を凍結し、後から解凍
- **学習率の調整**: ファインチューニング時は低めの学習率を使用
- **Early Stopping**: 過学習を防ぐため早期停止を活用
- **段階的な学習**: 一度に全て学習せず、段階的にアプローチ
- **データ拡張**: 合成データと実画像で異なる拡張戦略を適用

