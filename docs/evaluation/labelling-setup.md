# Labelling setup (Label Studio)

Step by step for building `data/eval/annotations.json` (#19). Read
[annotation-guideline.md](annotation-guideline.md) first.

Label Studio runs locally in your browser. It gets its **own virtual environment** so its
dependencies never touch the services' `venv`.

## 1. Pre-fill boxes with the current detector (service venv)

```powershell
venv\Scripts\activate
python scripts/eval/prelabel.py          # all 20 images in sample_images/
```

Writes `data/eval/prelabels.json` (not committed): one task per image, pre-filled with the
current detector's boxes so you correct instead of drawing everything.

## 2. Install and start Label Studio (separate venv)

```powershell
python -m venv venv-label
venv-label\Scripts\activate
pip install label-studio

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = (Get-Location).Path   # run from the repo root
label-studio start
```

It opens http://localhost:8080. Create a local account (stays on your machine).

## 3. Create the project

1. **Create Project**, name it `retail-eval`.
2. **Labeling Setup → Custom template → Code**: paste the content of
   `evaluation/labelstudio_config.xml`, then **Save**.
3. **Settings → Cloud Storage → Add Source Storage → Local files**:
   - Absolute local path: the full path of your `sample_images` folder
     (e.g. `D:\infilect_test_project\retail-ai-pipeline-final draft\sample_images`)
   - Leave **"Treat every bucket object as a source file" unticked**
   - **Add Storage**. Do **not** click *Sync*; the tasks come from the import below.
   This only gives Label Studio permission to show images from that folder.
4. **Import** → upload `data/eval/prelabels.json`.
5. **Settings → Annotation**: make sure **"Use predictions to prefill annotations"** is on and
   the model version `yolov8n-coco-baseline` is selected.

## 4. Label

For each image:

1. Open the task. The detector's boxes are already there.
2. Fix boxes (drag edges), delete wrong ones (select + `Backspace`), add missed ones
   (press the `product` label or its hotkey, then draw).
3. Select each box and type its **group code** in the box's text field (sidebar).
4. **Submit**. Use **Update** if you come back to it later.

Save regularly: **Submit** stores the image's labels in Label Studio's own database
(in your user profile), so closing the browser loses nothing.

## 5. Export and convert

1. Project page → **Export** → **JSON** (not "JSON-MIN") → save as
   `data/eval/labelstudio_export.json`.
2. Convert (service venv):

   ```powershell
   venv\Scripts\activate
   python scripts/eval/convert_labelstudio.py data/eval/labelstudio_export.json
   ```

   It refuses to write anything while an image is unlabelled or a box has no group code,
   and lists what to fix. To save progress before all 20 images are done, add
   `--allow-partial`.
3. Commit `data/eval/annotations.json` and `data/eval/labelstudio_export.json`
   (the export lets you reopen and fix labels later).

## Troubleshooting

| Problem | Fix |
|---|---|
| Images show as broken | Check both environment variables were set **in the same terminal** before `label-studio start`, and that the Local files storage path is the `sample_images` folder |
| No pre-filled boxes | Settings → Annotation → enable predictions; check `prelabels.json` has boxes |
| Converter: "labelled at WxH but file is ..." | The image was replaced after labelling; relabel it |
