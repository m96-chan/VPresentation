# char.png を蒸留してリアルタイム化する（NVIDIA GPU / RTX 5090）

teacher（general poser）は任意画像を動かせるが重い（M3 CoreML で ~6fps）。
**student（蒸留モデル）**にすると小さく速くなり、CoreML化で **~25fps** で動く。
student の作成には NVIDIA GPU での **distillation（蒸留学習）** が必要。

> 学習時間の目安：RTX A6000 で ~30時間（THA4 作者）。RTX 5090 ならより速いが数時間〜十数時間規模。

---

## 0. 前提（5090 Linux 機）
- NVIDIA GPU + CUDA、Python 3.11（pyenv）
- ディスク数GB（teacher models 610MB + pose_dataset 90MB + 学習中間物）

> ⚠️ **THA4 の `poetry/pyproject.toml` は `torch 1.13.1+cu117` に固定されているが、
> これは RTX 5090（Blackwell, sm_120）を認識できない**（cu117 は sm_120 カーネルを持たない）。
> poetry は使わず、**pip で新しい torch（2.x, cu128 系）を入れる**こと。
> `torch.cuda.get_device_capability(0)` が `(12, 0)` になれば OK。

## 1. THA4 本体を用意（リポジトリ内に配置）
`third_party/tha4_src` は `.gitignore` 済みで、既存の `tools/export_onnx.py` 等が
参照している場所と同じ。ここに素の THA4 リポジトリを clone する：
```bash
git clone https://github.com/pkhungurn/talking-head-anime-4-demo.git third_party/tha4_src

# venv は poetry ではなく pip + 新しい torch(cu128) で作る
~/.pyenv/versions/3.11.14/bin/python3.11 -m venv .venv-distill
.venv-distill/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-distill/bin/pip install numpy scipy pillow omegaconf opencv-python-headless matplotlib tensorboard
```
（wxpython / mediapipe / tkinter は distiller_ui などの GUI 専用で、CLI 蒸留 `tha4.app.distill` には不要）

## 2. 学習に必要なデータを取得（VPresentation 側の `data/` に置く）
```bash
# teacher models (610MB) -> VPresentation/data/tha4/
bash tools/fetch_weights.sh

# pose_dataset (学習用ポーズ, 90MB) -> VPresentation/data/pose_dataset.pt
curl -L -o data/pose_dataset.pt \
  "https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1"
```

`tha4.app.distill` は `data/tha4/*.pt`・`data/pose_dataset.pt`・`data/distill_examples/<char>/`
を **cwd（＝`third_party/tha4_src`）からの相対パス**で決め打ちで参照するため、
VPresentation 側に置いた実体をシンボリックリンクで見せる：
```bash
REPO="$(pwd)"   # VPresentation リポジトリ root
ln -sfn "$REPO/data/pose_dataset.pt"        third_party/tha4_src/data/pose_dataset.pt
ln -sfn "$REPO/data/images/char_512.png"    third_party/tha4_src/data/images/char_512.png
ln -sfn "$REPO/data/images/char_face_mask.png" third_party/tha4_src/data/images/char_face_mask.png
ln -sfn "$REPO/data/distill_examples/char"  third_party/tha4_src/data/distill_examples/char
for f in eyebrow_decomposer face_morpher eyebrow_morphing_combiner body_morpher upscaler; do
  ln -sfn "$REPO/data/tha4/$f.pt" "third_party/tha4_src/data/tha4/$f.pt"
done
```

## 3. char.png（512² RGBA）と face mask を配置
`data/images/char_512.png` / `data/images/char_face_mask.png` は VPresentation
リポジトリに既に用意されている。
- `char_512.png` … 背景透過・512²・頭部が上半分中央（前処理済み）
- `char_face_mask.png` … **目と口の領域を白で塗ったマスク**（顔器官の位置）。
  THA4 側の検証は **モード `RGB`・各チャンネル 0 か 255 のみ** を要求する
  （`L`/グレースケールで保存すると `DistillerConfig.check()` で弾かれる）。

> ⚠️ **マスクは要確認/調整**。自動生成の概算だと顔からズレることがある
>（実例: 口のマスクが顎ではなく首・チョーカーにかかっていた）。
> `char_512.png` に赤枠で重ねて目視確認するのが手早い。
> より正確にやるなら `poetry run python -m tha4.app.distiller_ui` の GUI で
> char_512 を開き、目・口を直接マークする方法もある（THA4 純正）。

### 新しいキャラを足すとき

`tools/preprocess_character.py` の head-frac ヒューリスティクスは、
チビ寄りの全身立ち絵しか想定していない。バストアップの絵・アホ毛・広がった髪型では
頭の位置がずれるので、**顔を実測して明示配置する**（`djsaxia` / `ronome` はこの方法）。

THA4 は顔を固定座標で切り出す（眉 `192..320 × 64..192`、顔 `192..320 × 80..208`）ので、
合わせるべきなのは体ではなく **目線と口の位置**：

```bash
# 元絵で目の高さ・口の高さ・顔の中心 x を実測して渡す。
# --keep-alpha は既に切り抜き済みの絵（rembg 不要）
.venv-distill/bin/python tools/preprocess_character.py ronome.png data/images/ronome_512.png \
  --keep-alpha --src-cx 1030 --src-eye-y 575 --src-mouth-y 695 \
  --dst-eye-y 153 --dst-mouth-y 186.5
```

倍率は目〜口の距離から決まる。`--dst-eye-y` を下げすぎるとアホ毛が上端で切れるので、
アルファの bbox が上端に触っていないことを確認する。

マスクは 3 つの矩形（左目+眉 / 右目+眉 / 口+顎）で、**眉が上に動く分だけ目の上に余裕を取る**
（THA4 の distiller-ui doc の推奨）。

### 蒸留前に teacher で確認する（20時間を無駄にしないため）

フレーミングとマスクの間違いは、蒸留を回す前に teacher でポーズを付ければ見える。

```bash
.venv-distill/bin/python tools/preview_teacher.py \
  data/images/ronome_512.png out/ronome_teacher.png --mask data/images/ronome_face_mask.png
```

rest / mask 重ね / 瞬き / あ / い+眉上げ / 首振り / 上向き の 7 コマが出るので、
**目が閉じきるか・口が開くか・眉が上がるか・首を振っても破綻しないか**を見る。
teacher は任意画像を動かせるので、これは student の品質の予測ではなく、
「そもそも正しい問いを投げているか」の確認。数秒で終わる。

## 4. 蒸留 config を作成（GUIを使わない場合）
`data/distill_examples/char/config.yaml`（VPresentation 側に作成 → 上記シンボリックリンク経由で
`third_party/tha4_src/data/distill_examples/char/config.yaml` としても見える）:
```yaml
prefix: data/distill_examples/char
character_image_file_name: data/images/char_512.png
face_mask_image_file_name: data/images/char_face_mask.png
face_morpher_random_seed_0: 12771885812175595441
face_morpher_random_seed_1: 14367217090963479175
body_morpher_random_seed_0: 2892221210020292507
body_morpher_random_seed_1: 9998918537095922080
num_cpu_workers: 4
num_gpus: 1
```
（seed 値は THA4 同梱の `data/distill_examples/lambda_00/config.yaml` と同じデフォルト値）

## 5. 蒸留を実行（数時間〜）
`src/tha4/distiller/distill_face_morpher.py` などのパスも cwd 相対のため、
**`third_party/tha4_src` を cwd にして**実行する：
```bash
cd third_party/tha4_src
PYTHONPATH=src ../../.venv-distill/bin/python -m tha4.app.distill \
  --config_file data/distill_examples/char/config.yaml
```
完了すると `data/distill_examples/char/`（= VPresentation 側の実体）配下に student の
`face_morpher.pt` / `body_morpher.pt` が生成される。

> 起動確認済み：上記コマンドで teacher（eyebrow_decomposer / combiner / face_morpher）が
> ロードされ、実際に student (SIREN face_morpher) の学習が開始し、チェックポイントと
> sample_output の画像が生成されることを確認済み（RTX 5090, torch 2.11+cu128）。
> 本番は `face_morpher` だけで数十万〜100万サンプル規模、`body_morpher` も含め数時間かかるため、
> `nohup` や `tmux`/`screen` 上でバックグラウンド実行すること。

## 6. VPresentation に戻して 25fps 化
生成物を VPresentation の character_models に配置：
```bash
# VPresentation 側
mkdir -p data/character_models/char
cp <THA4>/data/distill_examples/char/face_morpher.pt data/character_models/char/
cp <THA4>/data/distill_examples/char/body_morpher.pt data/character_models/char/
cp data/images/char_512.png data/character_models/char/character.png
cat > data/character_models/char/character_model.yaml <<'YAML'
character_image_file_name: character.png
face_morpher_file_name: face_morpher.pt
body_morpher_file_name: body_morpher.pt
YAML

# student を CoreML 化（Mac, .venv-coreml / torch2.7）
.venv-coreml/bin/python tools/convert_coreml.py data/character_models/char

# カメラで ~25fps（student は自動選択）
.venv/bin/python gui/vpresentation_camera.py data/character_models/char
```

これで char.png が lambda_00 と同じ **~25fps** で動く。

**蒸留機（Linux + NVIDIA）でそのまま確認する場合**、CoreML 化は不要。
同じ student を PyTorch/CUDA で回すバックエンドが自動で選ばれる（~52fps）：

```bash
.venv-distill/bin/python gui/vpresentation_camera.py data/character_models/char

# 後から見返せるようにフレームを保存する（デモの記録用）
.venv-distill/bin/python gui/vpresentation_camera.py data/character_models/char \
  --record out/cam --frames 120
```

蒸留が終わった直後に、その場でカメラバインドして確認できる。

---

## メモ
- 蒸留は teacher で教師画像を大量生成しながら student を学習する。GPUメモリ・時間を要する。
- マスク/フレーミングが顔とズレていると品質が落ちる。GUI での確認を推奨。
- student は per-character。別キャラは再度蒸留が必要。
