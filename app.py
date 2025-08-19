# app.py
import os
import time
import base64
import signal
import atexit
import logging
import threading
import numpy as np
from queue import Queue, Empty
from typing import List, Any, Dict
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, Future

from flask import Flask, request, jsonify, abort

from sentence_transformers import SentenceTransformer

# ---------- Configuration (env) ----------
EMBED_API_KEYS = os.environ.get("EMBED_API_KEYS", "")  # comma-separated
ALLOWED_API_KEYS = set([k.strip() for k in EMBED_API_KEYS.split(",") if k.strip()])

MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
BATCH_WAIT_SECONDS = float(os.environ.get("BATCH_WAIT_SECONDS", "0.06"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))
RATE_LIMIT_TOKENS_PER_MIN = int(os.environ.get("RATE_LIMIT_TOKENS_PER_MIN", "1200"))
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "200"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "20.0"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ---------- Logging ----------
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("embed-flask")

# ---------- App ----------
app = Flask(__name__)

# ---------- Simple in-process token bucket per api-key ----------
class TokenBucket:
    def __init__(self, rate_per_min: int, burst: int):
        self.capacity = burst
        self.tokens = burst
        self.rate_per_min = rate_per_min
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, amount: int = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            refill = elapsed * (self.rate_per_min / 60.0)
            self.tokens = min(self.capacity, self.tokens + refill)
            self.last = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

_rate_buckets: Dict[str, TokenBucket] = {}

def get_bucket(key: str) -> TokenBucket:
    if key not in _rate_buckets:
        _rate_buckets[key] = TokenBucket(RATE_LIMIT_TOKENS_PER_MIN, RATE_LIMIT_BURST)
    return _rate_buckets[key]

def check_api_key(key: str) -> bool:
    if not ALLOWED_API_KEYS:
        return True
    return (key or "") in ALLOWED_API_KEYS

# ---------- Batch processing infrastructure ----------
class BatchItem:
    def __init__(self, text: str, normalize: bool, future: Future):
        self.text = text
        self.normalize = normalize
        self.future = future

class BatchWorker(threading.Thread):
    def __init__(self, model: SentenceTransformer, queue: Queue, max_batch: int, wait_seconds: float):
        super().__init__(daemon=True)
        self.model = model
        self.queue = queue
        self.max_batch = max_batch
        self.wait_seconds = wait_seconds
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def run(self):
        logger.info("Batch worker started")
        while not self._stop_event.is_set():
            try:
                first_item = self.queue.get(timeout=0.2)
            except Empty:
                continue

            items = [first_item]
            start = time.monotonic()
            # gather more items up to max_batch or until wait_seconds elapse
            while len(items) < self.max_batch:
                remaining = max(0.0, self.wait_seconds - (time.monotonic() - start))
                try:
                    item = self.queue.get(timeout=remaining)
                    items.append(item)
                except Empty:
                    break

            texts = [it.text for it in items]

            try:
                # run encode in threadpool to avoid blocking
                future = self._executor.submit(self.model.encode, texts, convert_to_numpy=True)
                embeddings = future.result(timeout=REQUEST_TIMEOUT)
                # normalize if requested
                # embeddings is numpy array shape (n, d)
                for it, emb in zip(items, embeddings):
                    vec = emb.astype(np.float32)
                    it.future.set_result(vec.tolist())
            except Exception as e:
                logger.exception("Embedding batch failed")
                for it in items:
                    if not it.future.done():
                        it.future.set_exception(e)

    def stop(self):
        self._stop_event.set()
        self._executor.shutdown(wait=False)
        logger.info("Batch worker stopping")

# ---------- Initialize model & queue (blocking on startup) ----------
logger.info("Loading model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
queue = Queue()
worker = BatchWorker(model=model, queue=queue, max_batch=MAX_BATCH_SIZE, wait_seconds=BATCH_WAIT_SECONDS)
worker.start()

# ---------- Helpers ----------
def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not check_api_key(key):
            logger.warning("Unauthorized request: missing/invalid API key")
            return jsonify({"detail": "Invalid API key"}), 401
        # rate limit per key
        bucket = get_bucket(key or "anon")
        if not bucket.consume(1):
            logger.warning("Rate limit exceeded for key %s", key)
            return jsonify({"detail": "Rate limit exceeded"}), 429
        return func(*args, **kwargs)
    return wrapper

def float32_to_base64(arr: List[float]) -> str:
    a = np.array(arr, dtype=np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")

# ---------- Endpoints ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "queue_size": queue.qsize(),
        "max_batch_size": MAX_BATCH_SIZE
    })

@app.route("/embed", methods=["POST"])
@require_api_key
def embed_single():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"detail": "Missing 'text' field"}), 400
    text = data["text"]
    if not isinstance(text, str) or len(text) == 0:
        return jsonify({"detail": "Invalid 'text'"}), 400
    if len(text) > 2000:
        return jsonify({"detail": "Text too long (max 2000 chars)"}), 400

    normalize = data.get("normalize", False)
    output = data.get("output", "list")  # 'list', 'base64', 'fp16'

    # enqueue and wait for result
    fut = Future()
    item = BatchItem(text=text, normalize=normalize, future=fut)
    queue.put(item)

    try:
        embedding = fut.result(timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logger.exception("Embedding request failed")
        return jsonify({"detail": "Embedding error"}), 500

    if output == "list":
        return jsonify({"embedding": embedding})
    elif output == "base64":
        return jsonify({"embedding": float32_to_base64(embedding)})
    elif output == "fp16":
        arr = np.array(embedding, dtype=np.float32).astype(np.float16)
        return jsonify({"embedding": arr.tolist()})
    else:
        return jsonify({"embedding": embedding})

@app.route("/embed/batch", methods=["POST"])
@require_api_key
def embed_batch():
    data = request.get_json(force=True)
    if not data or "texts" not in data:
        return jsonify({"detail": "Missing 'texts' field"}), 400
    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"detail": "Invalid 'texts' list"}), 400
    if len(texts) > 256:
        return jsonify({"detail": "Too many items (max 256)"}), 400

    normalize = data.get("normalize", False)
    output = data.get("output", "list")

    futures: List[Future] = []
    for t in texts:
        fut = Future()
        queue.put(BatchItem(text=t, normalize=normalize, future=fut))
        futures.append(fut)

    embeddings = []
    try:
        # wait for each future
        for fut in futures:
            val = fut.result(timeout=REQUEST_TIMEOUT)
            embeddings.append(val)
    except Exception as e:
        logger.exception("Batch embedding failure")
        return jsonify({"detail": "Batch embedding error"}), 500

    if output == "list":
        return jsonify({"embeddings": embeddings})
    elif output == "base64":
        encoded = [float32_to_base64(e) for e in embeddings]
        return jsonify({"embeddings": encoded})
    elif output == "fp16":
        arrs = [np.array(e, dtype=np.float32).astype(np.float16).tolist() for e in embeddings]
        return jsonify({"embeddings": arrs})
    else:
        return jsonify({"embeddings": embeddings})

# ---------- Graceful shutdown ----------
def shutdown_handler(signum, frame):
    logger.info("Shutdown signal received: %s", signum)
    try:
        worker.stop()
    except Exception:
        pass
    logger.info("Exiting")
    # allow process to exit naturally

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
atexit.register(lambda: worker.stop())

# ---------- Run local (for dev) ----------
if __name__ == "__main__":
    # dev server (not for production). Render will run via gunicorn specified in Procfile.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
