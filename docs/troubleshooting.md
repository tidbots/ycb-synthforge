# トラブルシューティング

このドキュメントでは、YCB SynthForgeで発生する可能性のある問題と解決方法を説明します。

## GPU関連

### GPUが認識されない

```bash
# ホストでNVIDIA確認
nvidia-smi

# コンテナ内で確認
docker compose run --rm yolo26_train nvidia-smi
```

**解決策:**
- NVIDIA Container Toolkitがインストールされているか確認
- Dockerデーモンを再起動: `sudo systemctl restart docker`

### メモリ不足エラー

`docker-compose.yml`で`shm_size`を増加:

```yaml
services:
  blenderproc:
    shm_size: '16gb'
  yolo26_train:
    shm_size: '16gb'
```

### CUDA Out of Memory

学習時のバッチサイズを減らす:

```bash
--batch 8   # 16から8に減らす
```

## YCBモデル関連

### YCBモデルが見つからない

google_16kまたはtsdf形式のモデルが必要です:

```
models/ycb/{object_name}/google_16k/textured.obj
models/ycb/{object_name}/tsdf/textured.obj
```

**解決策:**
```bash
# google_16k形式
python scripts/download_ycb_models.py --all --format google_16k

# tsdf形式（一部オブジェクトで必要）
python scripts/download_ycb_models.py --all --format berkeley
```

### tsdf形式でテクスチャが表示されない

tsdf形式のOBJファイルはマテリアル参照（usemtl）が欠落しています。

**解決策:**
```bash
docker compose run --rm fix_tsdf_materials
```

### テクスチャが壊れて表示される

`poisson`形式を使用している可能性があります。

**解決策:**
`google_16k`または`tsdf`形式を使用してください。

### 特定のオブジェクトのメッシュが歪む

**確認方法:**
```bash
# サムネイル生成（全オブジェクトのgoogle_16k/tsdf比較）
docker compose run --rm thumbnail_generator

# 結果を確認
xdg-open data/thumbnails/comparison_grid.png
```

**解決策:**
問題のあるオブジェクトを発見した場合、`generate_dataset.py`で設定:

```python
# scripts/blenderproc/generate_dataset.py

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

### メッシュ品質の自動検証

```bash
# メッシュの自動検証（Non-manifold edges等をチェック）
docker compose run --rm mesh_validator

# 結果を確認
cat data/mesh_validation_results.json
```

## Docker関連

### docker-composeエラー

旧版 `docker-compose` (v1.x) ではCompose file形式が非対応:

```bash
# エラー例
The Compose file is invalid because: Unsupported config option for services

# 解決: docker compose (v2+) を使用
docker compose run -d blenderproc  # ○ 正しい
docker-compose run -d blenderproc  # × 旧版は非対応
```

### コンテナがクラッシュする

ログを確認:
```bash
docker logs <container_id>
```

共有メモリ不足の場合は`shm_size`を増加してください。

## Python/依存関係

### NumPy互換性警告

`Dockerfile.yolo26`で`numpy<2`を指定済み。警告が出る場合は再ビルド:

```bash
docker compose build yolo26_train --no-cache
```

### モジュールが見つからない

```bash
# コンテナを再ビルド
docker compose build --no-cache
```

## データ生成関連

### 生成画像にオブジェクトが表示されない

**考えられる原因:**
1. オブジェクトのスケールが大きすぎる/小さすぎる
2. カメラ距離が不適切
3. オブジェクトがシーン外に配置されている

**確認方法:**
```bash
# オブジェクトサイズを確認
python3 << 'EOF'
from pathlib import Path

def get_obj_size(obj_path):
    min_c, max_c = [float('inf')]*3, [float('-inf')]*3
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                for i in range(3):
                    v = float(parts[i+1])
                    min_c[i], max_c[i] = min(min_c[i], v), max(max_c[i], v)
    return [max_c[i] - min_c[i] for i in range(3)]

for obj in Path('models').glob('**/textured.obj'):
    size = get_obj_size(obj)
    print(f"{obj}: {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm")
EOF
```

### アノテーションがない

COCO形式の`annotations.json`が生成されているか確認:
```bash
ls -la data/synthetic/coco/
cat data/synthetic/coco/annotations.json | head -100
```

## 学習関連

### 学習が収束しない

**考えられる原因:**
1. 学習率が高すぎる
2. データが少なすぎる
3. クラス不均衡

**解決策:**
- 学習率を下げる: `--lr0 0.001`
- データを増やす
- クラスバランスを確認

### mAPが低い

**考えられる原因:**
1. データの多様性が不足
2. ドメインランダム化が不十分
3. モデルサイズが小さすぎる

**解決策:**
- より大きなモデルを使用: `yolo26m` → `yolo26l`
- データ生成設定を見直す
- エポック数を増やす

## 関連ドキュメント

- [セットアップガイド](setup.md)
- [設定ファイルガイド](configuration.md)
- [パイプライン実行](pipeline.md)
