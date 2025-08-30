import asyncio
import aiohttp
import io
import logging
import numpy as np
import streamlit as st
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import time
from datetime import datetime, timedelta
import json

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
class Config:
    DEFAULT_TARGET_SIZE = (224, 224)
    DEFAULT_MODEL_PATH = "mobilenet.h5"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    TIMEOUT_SECONDS = 30
    MAX_CONCURRENT_DOWNLOADS = 5
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]

    # GIBS WMS settings
    GIBS_WMS_BASE = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    DEFAULT_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"  # fast, pretty true color
    DEFAULT_BBOX_SPAN_DEG = 0.25  # ~area around lat/lon; tweak if needed
    WMS_WIDTH = 512  # fetch larger then downscale -> nicer results
    WMS_HEIGHT = 512


# =========================
# Enhanced Image Processor
# =========================
class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = Config.DEFAULT_TARGET_SIZE):
        self.target_size = target_size

    def validate_image(self, img: Image.Image) -> bool:
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False
        if img.size[0] * img.size[1] > 80_000_000:  # sanity cap
            return False
        return True

    def preprocess_image(self, img: Image.Image) -> Optional[np.ndarray]:
        try:
            if not self.validate_image(img):
                raise ValueError("Invalid image format or size")

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            arr = keras_image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            # IMPORTANT: match MobileNetV2 training
            arr = mobilenet_v2_preprocess(arr)  # scales to [-1,1]
            return arr

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return None

    def get_image_stats(self, img: Image.Image) -> Dict[str, Any]:
        return {
            "size": img.size,
            "mode": img.mode,
            "format": img.format,
            "has_transparency": img.mode in ['RGBA', 'LA'] or 'transparency' in img.info
        }


# =========================
# Enhanced Wildfire Model
# =========================
class WildfireModel:
    def __init__(self, model_path: str = Config.DEFAULT_MODEL_PATH):
        self.model = None
        self.model_loaded = False
        self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            st.error(f"⚠️ Model file '{model_path}' not found. Place it next to the app or update the path.")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            st.error(f"⚠️ Failed to load model: {str(e)}")
            return False

    def predict(self, img_array: np.ndarray) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        if not self.model_loaded or self.model is None:
            return None, None
        try:
            t0 = time.time()
            preds = self.model.predict(img_array, verbose=0)[0]
            dt = (time.time() - t0) * 1000

            # Handle shapes: [1], [2], or logits
            if preds.ndim == 1 and preds.shape[0] == 1:
                fire_prob = float(preds[0])
            elif preds.ndim == 1 and preds.shape[0] == 2:
                fire_prob = float(preds[1])  # assume [no_fire, fire]
            else:
                # softmax-like or multi-class: take max
                fire_prob = float(np.max(preds))

            metrics = {
                "std_dev": float(np.std(preds)),
                "confidence_range": [float(np.min(preds)), float(np.max(preds))],
                "entropy": float(-np.sum(preds * np.log(np.clip(preds, 1e-12, 1.0)))),
                "inference_time_ms": round(dt, 2),
                "prediction_distribution": preds.tolist()
            }
            return fire_prob, metrics

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            st.error(f"Prediction error: {str(e)}")
            return None, None

    def get_model_info(self) -> Dict[str, Any]:
        if not self.model_loaded:
            return {"status": "not_loaded"}
        try:
            return {
                "status": "loaded",
                "input_shape": str(self.model.input_shape),
                "output_shape": str(self.model.output_shape),
                "total_params": int(self.model.count_params()),
                "layers": len(self.model.layers)
            }
        except Exception:
            return {"status": "error"}


# =========================
# NASA GIBS (WMS) Scraper
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def make_gibs_wms_url(
    layer: str,
    date_str: str,
    lat: float,
    lon: float,
    span_deg: float,
    width: int,
    height: int,
    fmt: str = "image/jpeg",
) -> str:
    # Construct a small bbox around the point
    half = span_deg / 2.0
    south = clamp(lat - half, -89.9, 89.9)
    north = clamp(lat + half, -89.9, 89.9)
    west = clamp(lon - half, -179.9, 179.9)
    east = clamp(lon + half, -179.9, 179.9)

    # WMS 1.3.0 expects CRS=EPSG:4326 and bbox order: south,west,north,east
    params = (
        f"service=WMS&request=GetMap&version=1.3.0"
        f"&layers={layer}"
        f"&styles="
        f"&format={fmt}"
        f"&transparent=false"
        f"&height={height}&width={width}"
        f"&crs=EPSG:4326"
        f"&bbox={south},{west},{north},{east}"
        f"&time={date_str}"
    )
    return f"{Config.GIBS_WMS_BASE}?{params}"

class APIImageScraper:
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT_SECONDS)

    async def create_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=Config.MAX_CONCURRENT_DOWNLOADS)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={"User-Agent": "WildfireDetectionApp/1.0"}
            )

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def build_gibs_urls(
        self,
        lat: float,
        lon: float,
        days_back: int,
        layer: str,
        span_deg: float,
        width: int,
        height: int,
        fmt: str = "image/jpeg"
    ) -> List[str]:
        """
        Build per-day NASA GIBS WMS URLs for the last `days_back` days including today.
        """
        urls = []
        today = datetime.utcnow().date()
        for i in range(days_back):
            day = today - timedelta(days=i)
            date_str = day.isoformat()  # YYYY-MM-DD
            url = make_gibs_wms_url(layer, date_str, lat, lon, span_deg, width, height, fmt)
            urls.append(url)
        logger.info(f"Built {len(urls)} GIBS URLs (layer={layer})")
        return urls

    async def download_single_image(self, url: str) -> Tuple[str, Optional[bytes], Dict[str, Any]]:
        meta = {
            "url": url,
            "downloaded_at": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "size_bytes": 0,
            "content_type": None,
            "download_time_ms": 0
        }
        t0 = time.time()
        try:
            async with self.session.get(url) as resp:
                meta["download_time_ms"] = round((time.time() - t0) * 1000, 2)
                if resp.status == 200:
                    data = await resp.read()
                    if len(data) > Config.MAX_IMAGE_SIZE:
                        raise ValueError(f"Image too large: {len(data)} bytes")
                    meta["success"] = True
                    meta["size_bytes"] = len(data)
                    meta["content_type"] = resp.headers.get("content-type", "unknown")
                    return url, data, meta
                else:
                    meta["error"] = f"HTTP {resp.status}"
                    return url, None, meta
        except Exception as e:
            meta["error"] = str(e)
            logger.error(f"Download failed for {url}: {e}")
            return url, None, meta

    async def download_images(self, urls: List[str]) -> Tuple[List[Tuple[str, bytes]], List[Dict[str, Any]]]:
        await self.create_session()
        sem = asyncio.Semaphore(Config.MAX_CONCURRENT_DOWNLOADS)

        async def run(url):
            async with sem:
                return await self.download_single_image(url)

        tasks = [run(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        ok, metas = [], []
        for (url, data, meta) in results:
            metas.append(meta)
            if data is not None:
                ok.append((url, data))
        return ok, metas


# =========================
# Helper Functions
# =========================
def display_prediction_results(
    fire_prob: float,
    confidence_metrics: Dict[str, Any],
    confidence_threshold: float,
    image_stats: Dict[str, Any]
) -> None:

    if fire_prob >= confidence_threshold:
        st.error("🔥 **WILDFIRE LIKELY DETECTED**")
    else:
        st.success("✅ **NO WILDFIRE DETECTED**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Fire Probability", f"{fire_prob:.1%}")
    with c2:
        st.metric("Confidence (1−σ)", f"{(1 - confidence_metrics.get('std_dev', 0)):.2f}")
    with c3:
        st.metric("Inference Time", f"{confidence_metrics.get('inference_time_ms', 0):.0f} ms")

    with st.expander("📊 Details"):
        st.json({
            "prediction_metrics": confidence_metrics,
            "image_properties": image_stats,
            "threshold_used": confidence_threshold
        })


def process_downloaded_images(
    downloaded_images: List[Tuple[str, bytes]],
    model: WildfireModel,
    processor: ImageProcessor,
    confidence_threshold: float = Config.DEFAULT_CONFIDENCE_THRESHOLD
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = []
    stats = {
        "total_images": len(downloaded_images),
        "successful_predictions": 0,
        "failed_predictions": 0,
        "wildfire_detections": 0,
        "average_confidence": 0.0,
        "processing_start": datetime.now().isoformat()
    }

    prog = st.progress(0)
    status = st.empty()

    for i, (url, img_bytes) in enumerate(downloaded_images):
        try:
            status.text(f"Processing image {i+1}/{len(downloaded_images)}")
            prog.progress((i + 1) / max(1, len(downloaded_images)))

            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_stats = processor.get_image_stats(image)
            arr = processor.preprocess_image(image)
            if arr is None:
                raise ValueError("Preprocessing failed")

            fire_prob, conf = model.predict(arr)
            if fire_prob is None:
                raise ValueError("Model prediction failed")

            is_fire = fire_prob >= confidence_threshold
            results.append({
                "url": url,
                "status": "🔥 Wildfire Likely" if is_fire else "✅ No Wildfire",
                "fire_probability": fire_prob,
                "confidence_metrics": conf,
                "image_stats": image_stats,
                "is_wildfire": is_fire,
                "processed_at": datetime.now().isoformat()
            })

            stats["successful_predictions"] += 1
            if is_fire:
                stats["wildfire_detections"] += 1

        except Exception as e:
            logger.error(f"Processing error for {url}: {e}")
            results.append({
                "url": url,
                "status": "❌ Processing Failed",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            })
            stats["failed_predictions"] += 1

    if stats["successful_predictions"] > 0:
        vals = [r["fire_probability"] for r in results if "fire_probability" in r]
        if vals:
            stats["average_confidence"] = float(np.mean(vals))

    stats["processing_end"] = datetime.now().isoformat()
    prog.empty()
    status.empty()
    return results, stats


# =========================
# Streamlit UI
# =========================
def create_sidebar():
    st.sidebar.header("🔧 Configuration")
    confidence_threshold = st.sidebar.slider(
        "Fire Probability Threshold",
        min_value=0.1, max_value=0.9,
        value=Config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05
    )

    st.sidebar.header("📍 Location")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        lat = st.number_input("Latitude", value=37.7749, format="%.4f")
    with c2:
        lon = st.number_input("Longitude", value=-122.4194, format="%.4f")

    st.sidebar.header("🛰️ GIBS Settings")
    layer = st.sidebar.selectbox(
        "Imagery Layer",
        options=[
            "MODIS_Terra_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
        ],
        index=0
    )
    date_range = st.sidebar.selectbox("Days Back (include today)", options=[1, 3, 7, 14, 30], index=0)
    span = st.sidebar.slider("BBox Span (degrees)", 0.05, 1.0, Config.DEFAULT_BBOX_SPAN_DEG, 0.05)

    st.sidebar.caption("Larger spans cover wider areas but may include clouds or non-fire regions.")

    return confidence_threshold, lat, lon, int(date_range), layer, float(span)


def display_batch_results(results: List[Dict[str, Any]], stats: Dict[str, Any]):
    st.subheader("📊 Batch Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Images", stats["total_images"])
    with c2:
        st.metric("Successful", stats["successful_predictions"])
    with c3:
        st.metric("Wildfire Detected", stats["wildfire_detections"])
    with c4:
        rate = (stats["successful_predictions"] / max(1, stats["total_images"])) * 100
        st.metric("Success Rate", f"{rate:.1f}%")

    if stats["successful_predictions"] > 0:
        st.metric("Avg Fire Probability", f"{stats['average_confidence']:.1%}")

    st.subheader("🔍 Details")
    for i, r in enumerate(results):
        with st.expander(f"Image {i+1}: {r['status']}"):
            st.write(f"**Source:** {r['url']}")
            if "fire_probability" in r:
                st.write(f"**Fire Probability:** {r['fire_probability']:.1%}")
                st.progress(min(max(r['fire_probability'], 0.0), 1.0))
                if "confidence_metrics" in r:
                    m = r["confidence_metrics"]
                    st.write(f"- Inference Time: {m.get('inference_time_ms', 0):.1f} ms")
                    st.write(f"- Model Confidence (1−σ): {(1 - m.get('std_dev', 0)):.2f}")
                if "image_stats" in r:
                    s = r["image_stats"]
                    st.write(f"- Dimensions: {s.get('size')}, Mode: {s.get('mode')}, Format: {s.get('format')}")
            elif "error" in r:
                st.error(f"Processing failed: {r['error']}")


async def run_batch_analysis(
    lat: float, lon: float, days_back: int, layer: str, span_deg: float,
    confidence_threshold: float, model: WildfireModel, processor: ImageProcessor, scraper: APIImageScraper
):
    with st.spinner("🛰️ Building NASA GIBS requests..."):
        urls = await scraper.build_gibs_urls(
            lat=lat, lon=lon, days_back=days_back, layer=layer,
            span_deg=span_deg, width=Config.WMS_WIDTH, height=Config.WMS_HEIGHT, fmt="image/jpeg"
        )

    if not urls:
        st.warning("No imagery URLs generated. Try a different date range or layer.")
        return

    with st.spinner("⬇️ Downloading imagery..."):
        imgs, metas = await scraper.download_images(urls)

    if not imgs:
        st.error("No images downloaded. Check network or try smaller date range.")
        return

    ok = len(imgs)
    if ok < len(urls):
        st.warning(f"Downloaded {ok}/{len(urls)} images.")
    else:
        st.success(f"Downloaded {ok} images.")

    with st.spinner("🔍 Running wildfire detection..."):
        results, stats = process_downloaded_images(imgs, model, processor, confidence_threshold)

    display_batch_results(results, stats)
    return results, stats


def main():
    st.set_page_config(page_title="Wildfire Detection System", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

    st.markdown('<h1 style="text-align:center;color:#ff4b4b;">🌍 AI-Powered Wildfire Detection (NASA GIBS)</h1>', unsafe_allow_html=True)
    st.caption("Fetch NASA GIBS imagery around a point, preprocess to MobileNetV2, and classify fire likelihood.")

    confidence_threshold, lat, lon, days_back, layer, span = create_sidebar()

    try:
        processor = ImageProcessor()
        model = WildfireModel(Config.DEFAULT_MODEL_PATH)
        scraper = APIImageScraper()

        info = model.get_model_info()
        if info.get("status") == "loaded":
            with st.sidebar:
                st.success("✅ Model loaded")
                with st.expander("Model Details"):
                    st.json(info)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["📤 Upload Image", "🛰️ NASA GIBS Satellite Analysis"])

    with tab1:
        st.header("📤 Upload Image Analysis")
        up = st.file_uploader("Choose an image…", type=Config.SUPPORTED_FORMATS)
        if up:
            if up.size > Config.MAX_IMAGE_SIZE:
                st.error(f"File too large (> {Config.MAX_IMAGE_SIZE / (1024*1024):.1f} MB)")
            else:
                try:
                    im = Image.open(up).convert("RGB")
                    stats = processor.get_image_stats(im)
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.image(im, caption="Uploaded Image", use_column_width=True)
                    with c2:
                        st.write("**Image Properties**")
                        st.write(f"- Size: {stats['size']}")
                        st.write(f"- Mode: {stats['mode']}")
                        st.write(f"- Format: {stats.get('format', 'Unknown')}")

                    with st.spinner("🔍 Analyzing…"):
                        arr = processor.preprocess_image(im)
                        if arr is not None:
                            p, m = model.predict(arr)
                            if p is not None:
                                display_prediction_results(p, m, confidence_threshold, stats)
                            else:
                                st.error("Prediction failed.")
                        else:
                            st.error("Preprocessing failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.header("🛰️ Real-time NASA GIBS Analysis")
        st.write(f"Layer: `{layer}`  ·  Center: ({lat:.4f}, {lon:.4f})  ·  Span: ±{span/2:.2f}°  ·  Days back: {days_back}")
        if st.button("🚀 Fetch & Analyze NASA GIBS", type="primary"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_batch_analysis(
                    lat, lon, days_back, layer, span,
                    confidence_threshold, model, processor, scraper
                ))
            finally:
                loop.run_until_complete(scraper.close_session())
                loop.close()


if __name__ == "__main__":
    main()
