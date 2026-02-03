# 追加学習（Incremental Learning）

このドキュメントでは、学習済みモデルに新しいオブジェクトを追加する方法を説明します。

## 概要

追加学習は、既存の学習済みモデルに新しいクラスを追加する手法です。全データを再学習せずに効率的にモデルを拡張できます。

## 手法の比較

| 手法 | 学習時間 | 精度維持 | 実装難易度 |
|------|---------|---------|-----------|
| 全データ再学習 | 長い | 高い | 簡単 |
| リプレイ (サブセット) | 短い | やや低下 | 簡単 |
| Backbone凍結 | 短い | やや低下 | 簡単 |
| 知識蒸留 | 中程度 | 高い | やや複雑 |

## ステップ1: サブセット作成

元データから代表的なサンプルを抽出（クラス均等サンプリング）:

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/create_subset.py \
  --source /workspace/yolo_dataset \
  --output /workspace/data/ycb_subset \
  --num_samples 5000 \
  --val_samples 500
```

### パラメータ

| パラメータ | 説明 |
|-----------|------|
| `--source` | 元のデータセットパス |
| `--output` | 出力先パス |
| `--num_samples` | 訓練用サンプル数 |
| `--val_samples` | 検証用サンプル数 |

## ステップ2: データ統合

サブセットと新しいオブジェクトのデータを統合:

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/merge_for_incremental.py \
  --base /workspace/data/ycb_subset \
  --new /workspace/data/new_objects \
  --output /workspace/data/merged_dataset
```

### パラメータ

| パラメータ | 説明 |
|-----------|------|
| `--base` | ベースデータセット（サブセット） |
| `--new` | 新しいオブジェクトのデータセット |
| `--output` | 統合後の出力先 |

## ステップ3: Backbone凍結で追加学習

```bash
docker compose run --rm yolo26_train python \
  scripts/training/train_incremental.py \
  --weights /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --data /workspace/data/merged_dataset/dataset.yaml \
  --freeze 10 \
  --epochs 30 \
  --lr0 0.001
```

### パラメータ

| パラメータ | 説明 |
|-----------|------|
| `--weights` | 元の学習済みモデル |
| `--data` | 統合データセットの設定ファイル |
| `--freeze` | 凍結するレイヤー数（0-22） |
| `--epochs` | 学習エポック数 |
| `--lr0` | 初期学習率（追加学習では低めに設定） |

## パラメータ比較

| パラメータ | 通常学習 | 追加学習 |
|-----------|---------|---------|
| データ量 | 30,000枚 | 5,500枚 |
| freeze | 0 | 10 |
| lr0 | 0.01 | 0.001 |
| epochs | 50-100 | 30-50 |
| **推定時間** | ~1時間 | ~15分 |

## 推奨ケース

| ケース | 推奨手法 |
|--------|---------|
| 新クラスが1-5個 | Backbone凍結 |
| 新クラスが多数 | 全データ再学習 |
| 精度維持が最優先 | 知識蒸留または全データ再学習 |
| 時間が限られている | Backbone凍結 |

## 注意点

1. **忘却問題**: 追加学習では、元のクラスの精度が低下する可能性があります（カタストロフィック忘却）
2. **サブセットの重要性**: リプレイ用のサブセットは、元データの多様性を維持するよう選択してください
3. **学習率**: 追加学習では低い学習率を使用し、既存の知識を保護します

## 関連ドキュメント

- [カスタムモデルの追加](custom-models.md)
- [アンサンブル推論](ensemble-inference.md)
- [パイプライン実行](pipeline.md)
