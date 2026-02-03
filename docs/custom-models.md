# カスタムモデルの追加

このドキュメントでは、YCB以外の独自3Dモデルを追加して学習データに含める方法を説明します。

## ディレクトリ構造

```
models/
├── ycb/                          # YCBオブジェクト (クラスID: 0-102)
│   └── 002_master_chef_can/
│       └── google_16k/
│           └── textured.obj
└── tidbots/                      # カスタムオブジェクト (クラスID: 103-)
    ├── my_bottle/
    │   └── google_16k/
    │       ├── textured.obj
    │       └── textured.png
    └── my_gripper/
        └── google_16k/
            ├── textured.obj
            └── textured.png
```

## 設定ファイル

`scripts/blenderproc/config.yaml` でモデルソースを設定:

```yaml
model_sources:
  # YCBモデル
  ycb:
    path: "/workspace/models/ycb"
    include:                      # 特定オブジェクトのみ使用
      - "002_master_chef_can"
      - "005_tomato_soup_can"
      - "006_mustard_bottle"

  # カスタムモデル
  tidbots:
    path: "/workspace/models/tidbots"
    include: []                   # 空=全オブジェクト使用
```

## クラスIDの割り当て

| ソース | クラスID範囲 | 説明 |
|--------|-------------|------|
| ycb | 0-102 | 既存のYCB IDを維持 |
| tidbots | 103- | 自動で連番割り当て |
| (追加ソース) | 続きから連番 | ソース順に割り当て |

## 対応モデル形式

以下の形式を自動検出（優先順）:

1. `object_name/google_16k/textured.obj` (YCB形式)
2. `object_name/tsdf/textured.obj`
3. `object_name/textured.obj` (シンプル形式)
4. `object_name/*.obj` (任意のOBJ)

## 3Dモデルの変換

ダウンロードした3DモデルをOBJ形式に変換するスクリプトを用意しています。

### Blender形式 (.blend) → OBJ

```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_blend_to_obj.py \
  /tmp/mymodel/model.blend \
  /tmp/mymodel/output \
  /tmp/mymodel/textures  # テクスチャディレクトリ（オプション）
```

### COLLADA形式 (.dae) → OBJ

```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_dae_to_obj.py \
  /tmp/mymodel/model.dae \
  /tmp/mymodel/output
```

### FBX形式 (.fbx) → OBJ

```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_fbx_to_obj.py \
  /tmp/mymodel/model.fbx \
  /tmp/mymodel/output
```

### 変換後のコピー

```bash
mkdir -p models/tidbots/my_object/google_16k
cp /tmp/mymodel/output/* models/tidbots/my_object/google_16k/
```

## モデルのスケール確認・修正

ダウンロードした3Dモデルはスケールがバラバラなことが多いです。生成画像にオブジェクトが表示されない場合、スケールを確認してください。

### スケール確認

```bash
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

for obj in Path('models/tidbots').glob('*/google_16k/textured.obj'):
    size = get_obj_size(obj)
    print(f"{obj.parent.parent.name}: {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm")
EOF
```

### スケール修正（例: 0.03倍に縮小）

```bash
python3 << 'EOF'
from pathlib import Path
import shutil

def scale_obj(obj_path, scale):
    shutil.copy(obj_path, str(obj_path) + '.backup')
    lines = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                lines.append(f"v {float(p[1])*scale:.6f} {float(p[2])*scale:.6f} {float(p[3])*scale:.6f}\n")
            else:
                lines.append(line)
    with open(obj_path, 'w') as f:
        f.writelines(lines)

# 例: coke_zeroを0.03倍に縮小
scale_obj(Path('models/tidbots/coke_zero/google_16k/textured.obj'), 0.03)
EOF
```

### 適切なサイズの目安

| オブジェクト | 実際のサイズ |
|-------------|-------------|
| 缶（350ml） | 6-7 × 12-13 cm |
| ペットボトル | 6-8 × 20-25 cm |
| りんご | 7-8 × 7-8 cm |

## カスタムモデルのみで学習

YCBを使わず、独自モデルのみで学習する場合:

```yaml
# scripts/blenderproc/config.yaml
model_sources:
  # YCB disabled
  # ycb:
  #   path: "/workspace/models/ycb"
  #   include: []

  # カスタムモデルのみ使用
  tidbots:
    path: "/workspace/models/tidbots"
    include: []  # 空=全オブジェクト使用

scene:
  num_images: 2000          # 少数クラスなら2000枚程度で十分
  objects_per_scene: [1, 5]  # クラス数に合わせて調整
```

### 推奨データ量の目安

| クラス数 | 推奨枚数 | 1クラスあたり |
|---------|---------|--------------|
| 5 | 2,000 | 400枚 |
| 10 | 3,000 | 300枚 |
| 20 | 5,000 | 250枚 |
| 50+ | 10,000+ | 200枚+ |

## 特定オブジェクトのみ使用

全オブジェクトではなく、特定のオブジェクトのみを使用する場合:

```yaml
model_sources:
  ycb:
    path: "/workspace/models/ycb"
    include:
      - "002_master_chef_can"     # 缶
      - "003_cracker_box"         # 箱
      - "006_mustard_bottle"      # ボトル
      - "024_bowl"                # 食器
      - "025_mug"                 # マグカップ
```

これにより、学習対象を絞り込んで効率的にモデルを作成できます。

## 関連ドキュメント

- [設定ファイルガイド](configuration.md)
- [YCBクラス一覧](ycb-classes.md)
- [追加学習](incremental-learning.md)
