<!-- WildFire: AI-Powered Wildfire Detection -->
# WildFire

AI-Powered Wildfire Detection (Streamlit + NASA GIBS)

Brief: This is a small Streamlit app that fetches NASA GIBS satellite imagery around a point of interest and applies a MobileNet-based classifier to estimate wildfire likelihood.

---

## Features
- Upload an image and analyze it for fire probability.
- Fetch and analyze historical satellite imagery from NASA GIBS (per-day WMS requests).
- Preprocessing pipeline based on MobileNetV2 expected preprocessing.
- Metrics and inference details (inference time, distribution, entropy, etc.).

---

## Prerequisites
- Python 3.10+ recommended (3.11 also works).
- Optional: GPU + tf-gpu build for faster inference.

---

## Installation
1. Clone the repo
```bash
git clone <repo-url>
cd WildFire
```
2. (Optional) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
Notes:
- If you want CPU-only TensorFlow, use `pip install tensorflow-cpu` or change `tensorflow` to `tensorflow-cpu` in `requirements.txt`.

---

## Usage
Run the Streamlit app:
```bash
streamlit run main.py
```
Controls:
- Sidebar: change model threshold, lat/lon, days to look back, and GIBS layer.
- Tabs: upload a single image or fetch NASA GIBS imagery for a batch analysis.

Model: The default path for the model is `models/mobilenet.h5`. If you prefer your own model, place it there or update `Config.DEFAULT_MODEL_PATH` in `main.py`.

---

## File Overview
- `main.py` — main app logic and Streamlit UI.
- `models/mobilenet.h5` — pretrained model (HDF5) used for classification (if present).
- `requirements.txt` — dependencies.

---

## Troubleshooting & Notes
- If the app says the model isn't found, make sure `models/mobilenet.h5` exists or update `Config.DEFAULT_MODEL_PATH`.
- Pillow version < 9: the app uses a fallback for `Image.Resampling.LANCZOS`.
- The app uses `asyncio.run()` to fetch images; Streamlit is not a fully async-hosted environment — the implementation uses `.run()` to minimize loop conflicts.
- If you get TensorFlow import errors in the editor/linter, ensure TensorFlow is installed in your environment. Linter may not see packages installed in the container/environment.

Known issues:
- Model output assumptions: the classifier may return logits, softmax or sigmoid; the code attempts to handle multiple possible outputs but you should confirm the model behavior.
- Consider adding model caching (`@st.cache_resource`) for faster UX and tests for the I/O paths.

---

## Extending & Contributing
- Add unit tests for predictable behaviors (preprocessing, URL building, prediction outputs).
- Add model training or fine-tuning scripts if you want to improve detection accuracy.
- Add CI (GitHub Actions) to run a lint and tests automatically.

---

## License
This repository doesn't include an explicit license file; add one if you plan to publish or collaborate widely (e.g., MIT or Apache).

---

## Contact
If you'd like help with enhancements, model training, or debugging, open an issue or contact me at @bisongaallan@gmail.com


