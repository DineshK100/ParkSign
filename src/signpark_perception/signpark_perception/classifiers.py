"""Sign classifiers: local VLM via Ollama (default), Claude VLM, OCR + keyword baseline,
and ground truth (for debugging).

All classifiers take an RGB image (numpy HxWx3 uint8) and return a SignResult.
They are plain Python (no ROS) so the offline evaluation (E1) uses the same code.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

SIGN_CLASSES = ['head_in', 'back_in', 'parallel', 'angled', 'no_parking']


@dataclass
class SignResult:
    sign_class: str          # one of SIGN_CLASSES or "unknown"
    text_read: str = ''
    confidence: float = 0.0
    latency_s: float = 0.0
    raw: str = ''


def _to_png_b64(img: np.ndarray, min_width: int = 256) -> str:
    pil = Image.fromarray(img)
    if pil.width < min_width:  # small crops: upscale a bit so the model gets a sensible size
        s = min_width / pil.width
        pil = pil.resize((min_width, int(pil.height * s)), Image.BICUBIC)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.standard_b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# Claude (VLM)
# ---------------------------------------------------------------------------

PROMPT = """You are the sign reader of a self-parking car. The image shows one parking sign \
(possibly photographed at an angle, far away, or in poor light).

Classify which parking maneuver the sign requires, using exactly one class:
- head_in: drive in forwards (e.g. "HEAD-IN PARKING ONLY", "PULL IN ONLY", "FRONT-IN").
- back_in: reverse into the spot (e.g. "BACK-IN PARKING ONLY", "REVERSE PARKING", a car-reversing arrow symbol).
- parallel: parallel parking along the curb.
- angled: angle / diagonal parking.
- no_parking: parking is not allowed (e.g. "NO PARKING", a crossed-out P, "TOW-AWAY ZONE", fire lane).
- unknown: no parking sign is readable in the image.

Also transcribe the text you can read on the sign (empty string if none) and give your \
confidence from 0 to 1."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'sign_class': {'type': 'string', 'enum': SIGN_CLASSES + ['unknown']},
        'text_read': {'type': 'string'},
        'confidence': {'type': 'number'},
    },
    'required': ['sign_class', 'text_read', 'confidence'],
    'additionalProperties': False,
}


class ClaudeClassifier:
    """Claude vision model with a JSON-schema constrained answer.

    Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
    """

    name = 'claude'

    def __init__(self, model: str = 'claude-opus-5-5', effort: str = 'low'):
        import anthropic  # imported lazily so OCR / ground-truth work without the SDK
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def classify(self, img: np.ndarray) -> SignResult:
        t0 = time.time()
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=1024,
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',  # if a safety classifier declines, the API retries on a fallback model
                output_config={'effort': self.effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
                messages=[{'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                 'data': _to_png_b64(img)}},
                    {'type': 'text', 'text': PROMPT},
                ]}],
            )
        except self._anthropic.RateLimitError as e:
            return SignResult('unknown', raw=f'rate limited: {e}', latency_s=time.time() - t0)
        except self._anthropic.APIStatusError as e:
            return SignResult('unknown', raw=f'API error {e.status_code}: {e.message}', latency_s=time.time() - t0)
        except self._anthropic.APIConnectionError as e:
            return SignResult('unknown', raw=f'connection error: {e}', latency_s=time.time() - t0)
        dt = time.time() - t0
        if resp.stop_reason == 'refusal':
            return SignResult('unknown', raw='refusal', latency_s=dt)
        text = next((b.text for b in resp.content if b.type == 'text'), '')
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return SignResult('unknown', raw=text, latency_s=dt)
        cls = d.get('sign_class', 'unknown')
        if cls not in SIGN_CLASSES:
            cls = 'unknown'
        return SignResult(cls, d.get('text_read', ''), float(d.get('confidence', 0.0)), dt, text)


# ---------------------------------------------------------------------------
# Local VLM via Ollama (default; no API key, runs on the GPU)
# ---------------------------------------------------------------------------

class OllamaClassifier:
    """Any Ollama vision model (default qwen2.5vl:3b). Uses Ollama's JSON-schema `format`
    so the answer is always one of the allowed classes."""

    name = 'ollama'

    def __init__(self, model: str = 'qwen2.5vl:3b', host: str = 'http://localhost:11434', timeout: float = 60.0):
        self.model, self.host, self.timeout = model, host.rstrip('/'), timeout

    def classify(self, img: np.ndarray) -> SignResult:
        import urllib.error
        import urllib.request
        t0 = time.time()
        body = json.dumps({
            'model': self.model,
            'messages': [{'role': 'user', 'content': PROMPT + '\nAnswer in JSON.', 'images': [_to_png_b64(img)]}],
            'format': SCHEMA,
            'stream': False,
            'keep_alive': '30m',
            'options': {'temperature': 0},
        }).encode()
        req = urllib.request.Request(self.host + '/api/chat', data=body, headers={'Content-Type': 'application/json'})
        text, err = None, ''
        for _ in range(2):  # one retry (model loading, GPU busy)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = json.loads(r.read())['message']['content']
                break
            except (urllib.error.URLError, TimeoutError, KeyError, ConnectionError) as e:
                err = str(e)
        if text is None:
            return SignResult('unknown', raw=f'ollama error: {err}', latency_s=time.time() - t0)
        dt = time.time() - t0
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return SignResult('unknown', raw=text, latency_s=dt)
        cls = d.get('sign_class', 'unknown')
        if cls not in SIGN_CLASSES:
            # small local models sometimes read the text right but pick "unknown": fall back to keywords
            cls = keyword_class(d.get('text_read', ''))
        return SignResult(cls, d.get('text_read', ''), float(d.get('confidence', 0.0) or 0.0), dt, text)


# ---------------------------------------------------------------------------
# OCR baseline (Tesseract + keyword rules)
# ---------------------------------------------------------------------------

# Order matters: "NO PARKING" must win over plain "PARKING" rules.
KEYWORD_RULES = [
    ('no_parking', [r'\bNO\s*PARK', r'TOW', r'FIRE\s*LANE', r'NO\s*STOPPING']),
    ('back_in', [r'BACK', r'REVERSE']),
    ('head_in', [r'HEAD', r'PULL\s*IN', r'FRONT', r'NOSE']),
    ('parallel', [r'PARALLEL', r'CURB']),
    ('angled', [r'ANGLE', r'DIAGONAL']),
]


def keyword_class(text: str) -> str:
    t = text.upper()
    for cls, pats in KEYWORD_RULES:
        if any(re.search(p, t) for p in pats):
            return cls
    return 'unknown'


class OcrClassifier:
    name = 'ocr'

    def __init__(self):
        import pytesseract  # needs: sudo apt install tesseract-ocr
        self.tess = pytesseract

    def classify(self, img: np.ndarray) -> SignResult:
        t0 = time.time()
        pil = Image.fromarray(img).convert('L')
        if pil.width < 600:  # Tesseract likes ~30 px tall characters
            s = 600 / pil.width
            pil = pil.resize((600, int(pil.height * s)), Image.BICUBIC)
        text = self.tess.image_to_string(pil, config='--psm 6')
        cls = keyword_class(text)
        return SignResult(cls, ' '.join(text.split()), 1.0 if cls != 'unknown' else 0.0, time.time() - t0, text)


class GroundTruthClassifier:
    """Returns the true class (set per call). Lets you test everything downstream without API calls."""
    name = 'ground_truth'

    def classify(self, img: np.ndarray, truth: str = 'unknown') -> SignResult:
        return SignResult(truth, '', 1.0, 0.0)


def make_classifier(backend: str, **kw):
    if backend == 'ollama':
        return OllamaClassifier(**kw)
    if backend == 'claude':
        return ClaudeClassifier(**kw)
    if backend == 'ocr':
        return OcrClassifier()
    if backend == 'ground_truth':
        return GroundTruthClassifier()
    raise ValueError(f'unknown backend {backend!r} (ollama | claude | ocr | ground_truth)')
