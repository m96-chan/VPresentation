# char.png を蒸留してリアルタイム化する（NVIDIA GPU / RTX 5090）

teacher（general poser）は任意画像を動かせるが重い（M3 CoreML で ~6fps）。
**student（蒸留モデル）**にすると小さく速くなり、CoreML化で **~25fps** で動く。
student の作成には NVIDIA GPU での **distillation（蒸留学習）** が必要。

> 学習時間の目安：RTX A6000 で ~30時間（THA4 作者）。RTX 5090 ならより速いが数時間〜十数時間規模。

---

## 0. 前提（5090 Linux 機）
- NVIDIA GPU + CUDA、Python 3.10、Poetry
- ディスク数GB（teacher models 610MB + pose_dataset + 学習中間物）

## 1. THA4 本体を用意
```bash
git clone https://github.com/pkhungurn/talking-head-anime-4-demo.git
cd talking-head-anime-4-demo
poetry install            # torch(CUDA)含む依存
```

## 2. 学習に必要なデータを取得
```bash
# teacher models (610MB) -> data/tha4/
curl -L -o data/tha4/tha4-models.zip \
  "https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1"
unzip data/tha4/tha4-models.zip -d data/tha4/

# pose_dataset (学習用ポーズ) -> data/pose_dataset.pt
curl -L -o data/pose_dataset.pt \
  "https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1"
```

## 3. char.png（512² RGBA）と face mask を配置
VPresentation リポジトリから持ってくる：
```bash
cp <VPresentation>/data/images/char_512.png       data/images/char_512.png
cp <VPresentation>/data/images/char_face_mask.png data/images/char_face_mask.png
```
- `char_512.png` … 背景透過・512²・頭部が上半分中央（前処理済み）
- `char_face_mask.png` … **目と口の領域を白で塗ったマスク**（顔器官の位置）

> ⚠️ **マスクは要確認/調整**。同梱の char_face_mask は自動生成の概算です。
> **推奨**: `poetry run python -m tha4.app.distiller_ui` の GUI で char_512 を開き、
> 目・口を正確にマークして config とマスクを作るのが確実（THA4 純正の方法）。

## 4. 蒸留 config を作成（GUIを使わない場合）
`data/distill_examples/char/config.yaml`:
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

## 5. 蒸留を実行（数時間〜）
```bash
poetry run python -m tha4.app.distill --config_file data/distill_examples/char/config.yaml
```
完了すると `data/distill_examples/char/` 配下に student の
`face_morpher.pt` / `body_morpher.pt` が生成される。

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

---

## メモ
- 蒸留は teacher で教師画像を大量生成しながら student を学習する。GPUメモリ・時間を要する。
- マスク/フレーミングが顔とズレていると品質が落ちる。GUI での確認を推奨。
- student は per-character。別キャラは再度蒸留が必要。
