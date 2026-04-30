# MaskGuard 🛡️
### Real-Time Face Mask Detection in Crowded Environments

> **University of Roehampton — Deep Learning Applications (CMP-L016-0)**  
> Gayatri Dintakurthi | A00074428

A real-time **three-class** face mask detection system built on **YOLOv11n** with custom attention modules. Detects **Mask**, **No Mask**, and **Incorrect Mask** simultaneously in crowded scenes — deployed live via Flask and ngrok directly from Google Colab.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Overall mAP@0.5 (val) | **85.6%** |
| Overall mAP@0.5 (test) | **84.1%** |
| Mask AP@0.5 | 94.7% |
| No Mask AP@0.5 | 91.7% |
| Incorrect Mask AP@0.5 | 70.5% (+23.2 pp gain) |
| 5-Fold CV mAP (mean ± σ) | 0.812 ± 0.031 |
| Inference speed (T4 GPU) | ~154 ms/frame (~6.5 FPS) |

---

## Architecture

MaskGuard extends **YOLOv11n** with three custom attention modules applied at every stage of the detection pipeline:

| Module | Location | What it does |
|---|---|---|
| **DS-C2f** | Backbone | Adds Squeeze-and-Excitation channel attention after each C2f block |
| **DIConv** | Neck | Four-branch parallel convolution (pointwise, 1×k, k×1, k×k) for multi-scale context |
| **DAM** | Detection Head | Dual channel + spatial attention before bounding box prediction |

---

## Dataset

**Kaggle Face Mask Detection** — 853 images, 3,794 annotations, Pascal VOC XML format.

| Class | Instances | Share |
|---|---|---|
| Mask | 3,011 | 79.4% |
| No Mask | 668 | 17.6% |
| Incorrect Mask | 115 | 3.0% |

**26.2:1 class imbalance** handled via:
- Inverse-frequency class weights in AdamW (up to 5.7× for Incorrect Mask)
- Copy-paste augmentation (p=0.3) + Mosaic (p=1.0)
- Targeted minority-class oversampling to 80% of majority count
- Stratified 70/15/15 train/val/test split (seed=42)

---

## How to Run

**Everything runs in Google Colab — no local setup required.**

### Option A — Skip training, use pre-trained weights

`best.pt` is included in this repo. To run inference immediately without training:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('your_image.jpg', conf=0.25)
results[0].show()
```

### Option B — Run the full pipeline from scratch

### Step 1 — Open the notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gayatri0409/face-mask-detection/blob/main/MaskGuard.ipynb)


### Step 2 — Run cells in order

| Cells | What happens |
|---|---|
| 1–4 | Install dependencies (Ultralytics, Flask, pyngrok) |
| 5–8 | Download Kaggle dataset, convert XML → YOLO format |
| 9–11 | Consolidate labels, stratified split, compute class weights |
| 12–14 | Define DS-C2f, DIConv, DAM modules; patch YOLOv11n |
| 15–16 | Train for 150 epochs on T4 GPU |
| 17 | 5-fold cross-validation (diagnostic check) |
| 18 | External dataset evaluation |
| 19 | Start Flask server on port 5000 |
| 20 | Start ngrok tunnel → get public URL |

### Step 3 — Open the live demo
After Cell 20, a public HTTPS URL appears in the output. Open it in any browser to upload images or use your webcam for real-time detection.

> ⚠️ The ngrok URL changes every Colab session. Re-run Cells 19 and 20 to get a fresh one.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | YOLOv11n (2.6M parameters) |
| Optimizer | AdamW |
| Learning rate | 0.001 → 0.00001 (cosine annealing) |
| Warmup | 3 epochs |
| Batch size | 16 |
| Input resolution | 640×640 |
| Epochs | 150 |
| Random seed | 42 (fixed throughout) |
| Loss weights | Box=7.5, DFL=1.5, Class=inverse-frequency |

---

## Deployment

The live system uses:
- **Flask** — web server serving `/predict` endpoint (HTTP POST, returns base64 JPEG + JSON detections)
- **ngrok** — HTTPS tunnel making the Colab server publicly accessible
- **Confidence threshold** — 0.15 for live deployment (vs 0.25 for evaluation) to maximise recall on Incorrect Mask

---

## Reproducibility

All random operations use `SEED = 42` fixed identically across:
`Python random` · `NumPy` · `PyTorch` · `CUDA` · `sklearn KFold`

Run the notebook top-to-bottom to reproduce all results from scratch.

---

## Project Structure

```
face-mask-detection/
├── MaskGuard.ipynb        ← the notebook
├── MaskGuard_Report.pdf   ← the report (PDF, safe to share)
├── best.pt                ← trained weights
└── README.md              ← project overview
```

> **Note:** The dataset is downloaded automatically inside the notebook from Kaggle — no manual download needed.

---

## References

Key papers behind the design:

- YOLOv11 — Jocher et al., Ultralytics, 2024
- Squeeze-and-Excitation — Hu et al., CVPR 2018
- CBAM — Woo et al., ECCV 2018
- FMD-YOLOv8 — Wang et al., Sensors 2023
- Focal Loss / RetinaNet — Lin et al., ICCV 2017
- FPN — Lin et al., CVPR 2017
- SMOTE — Chawla et al., JAIR 2002
- AdamW — Loshchilov & Hutter, ICLR 2019

Full reference list is in the project report.

---

## License

This project was submitted as academic coursework. Please do not reuse or copy for other academic submissions.