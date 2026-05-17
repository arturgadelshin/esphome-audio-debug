import asyncio
import base64
import io
import json
import logging
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from aioesphomeapi import (
    APIClient,
    VoiceAssistantFeature,
)

log = logging.getLogger("audio_debug")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "device_host": "192.168.1.48",
    "device_port": 6053,
    "asr_host": "localhost",
    "asr_port": 10306,
    "filter_defaults": {
        "normalize": {"target_db": -10},
        "spectral_subtraction": {"noise_frames": 40},
    },
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


CONFIG = load_config()

app = FastAPI()

STATE = {
    "connected": False,
    "device_info": None,
    "auto_capture": True,
    "capturing": False,
    "audio_chunks": [],
    "capture_start": None,
    "wake_word": None,
    "last_recording": None,
    "cli": None,
    "unsubscribe_va": None,
    "max_duration": 10,
    "stop_task": None,
}


def save_wav(data: bytes, sample_rate: int = 16000, bits: int = 16, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return buf.getvalue()


def compute_metrics(data: bytes, sample_rate: int = 16000) -> dict:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    if len(samples) == 0:
        return {"rms": 0, "peak": 0, "duration": 0, "snr_estimate": 0}

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    duration = len(samples) / sample_rate

    if rms > 0:
        sorted_s = np.sort(np.abs(samples))
        noise_level = float(np.mean(sorted_s[: len(sorted_s) // 4]))
        signal_level = float(np.mean(sorted_s[len(sorted_s) * 3 // 4 :]))
        snr = (signal_level / noise_level) if noise_level > 0 else 0
    else:
        snr = 0

    rms_db = 20 * np.log10(rms / 32768) if rms > 0 else -96
    peak_db = 20 * np.log10(peak / 32768) if peak > 0 else -96

    n_fft = min(2048, len(samples))
    if n_fft > 1:
        from scipy.signal import spectrogram as scipy_spectrogram

        f, t, Sxx = scipy_spectrogram(samples, fs=sample_rate, nperseg=n_fft, noverlap=n_fft // 2)
        spectrogram_data = {
            "frequencies": f.tolist(),
            "times": t.tolist(),
            "power_db": (10 * np.log10(Sxx + 1e-10)).tolist(),
        }
    else:
        spectrogram_data = None

    return {
        "rms": round(rms, 1),
        "rms_db": round(rms_db, 1),
        "peak": int(peak),
        "peak_db": round(peak_db, 1),
        "duration": round(duration, 3),
        "snr_estimate": round(snr, 2),
        "sample_count": len(samples),
    }


def compute_spectrogram(data: bytes, sample_rate: int = 16000) -> dict | None:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    n_fft = min(1024, len(samples))
    if n_fft < 2:
        return None
    from scipy.signal import spectrogram as scipy_spectrogram
    f, t, Sxx = scipy_spectrogram(samples, fs=sample_rate, nperseg=n_fft, noverlap=n_fft // 2)
    Sdb = 10 * np.log10(Sxx + 1e-10)
    rows, cols = Sdb.shape
    step_r = max(1, rows // 64)
    step_c = max(1, cols // 256)
    down = Sdb[::step_r, ::step_c]
    return {
        "frequencies": f[::step_r].tolist(),
        "times": t[::step_c].tolist(),
        "power_db": down.tolist(),
    }


def save_recording(raw: bytes, wake_word: str = None) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ""
    if wake_word:
        tag = f"_{wake_word.replace(' ', '_')}"
    fname = f"rec_{ts}{tag}.wav"
    wav_data = save_wav(raw)
    (RECORDINGS_DIR / fname).write_bytes(wav_data)
    metrics = compute_metrics(raw)
    metrics["filename"] = fname
    (RECORDINGS_DIR / f"{fname}.json").write_text(json.dumps(metrics, ensure_ascii=False))
    STATE["last_recording"] = metrics
    log.info("Saved %s (%.2fs, RMS %.1fdB, SNR %.2f)", fname, metrics["duration"], metrics["rms_db"], metrics["snr_estimate"])
    return metrics


def finish_capture():
    STATE["capturing"] = False
    raw = b"".join(STATE["audio_chunks"]) if STATE["audio_chunks"] else b""
    STATE["audio_chunks"] = []
    if raw and STATE["auto_capture"]:
        save_recording(raw, STATE["wake_word"])


async def connect_device(host: str, port: int, password: str):
    if STATE["cli"] and STATE["connected"]:
        return STATE["device_info"]

    if STATE["unsubscribe_va"]:
        STATE["unsubscribe_va"]()
        STATE["unsubscribe_va"] = None

    cli = APIClient(host, port, password or None)
    await cli.connect()
    await cli.list_entities_services()

    device_info = await cli.device_info()
    STATE["cli"] = cli
    STATE["connected"] = True
    STATE["device_info"] = {
        "name": device_info.name,
        "mac": device_info.mac_address,
        "esphome_version": device_info.esphome_version,
    }

    flags = device_info.voice_assistant_feature_flags_compat(cli.api_version)
    STATE["device_info"]["api_audio"] = bool(flags & VoiceAssistantFeature.API_AUDIO)

    log.info("Connected to %s (ESPHome %s, API_AUDIO=%s)",
             device_info.name, device_info.esphome_version, STATE["device_info"]["api_audio"])

    async def on_start(conversation_id, run_flags, audio_settings, wake_word):
        log.info("VA start: wake_word=%s flags=%s", wake_word, run_flags)
        if STATE["stop_task"]:
            STATE["stop_task"].cancel()
            STATE["stop_task"] = None
        STATE["audio_chunks"] = []
        STATE["capture_start"] = time.time()
        STATE["wake_word"] = wake_word
        STATE["capturing"] = True
        loop = asyncio.get_event_loop()
        STATE["stop_task"] = loop.call_later(STATE["max_duration"], _timed_stop)
        return 0

    def _timed_stop():
        STATE["stop_task"] = None
        if STATE["capturing"]:
            log.info("Auto-stop after %ds", STATE["max_duration"])
            finish_capture()

    async def on_stop(abort):
        if STATE["stop_task"]:
            STATE["stop_task"].cancel()
            STATE["stop_task"] = None
        if not STATE["capturing"]:
            return
        log.info("VA stop: abort=%s chunks=%d", abort, len(STATE["audio_chunks"]))
        finish_capture()

    async def on_audio(data: bytes):
        if STATE["capturing"]:
            STATE["audio_chunks"].append(data)

    STATE["unsubscribe_va"] = cli.subscribe_voice_assistant(
        handle_start=on_start,
        handle_stop=on_stop,
        handle_audio=on_audio,
    )

    return STATE["device_info"]


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/connect")
async def api_connect():
    try:
        info = await connect_device(CONFIG["device_host"], CONFIG["device_port"], None)
        return JSONResponse(info)
    except Exception as e:
        STATE["connected"] = False
        log.error("Connect failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/disconnect")
async def api_disconnect():
    if STATE["stop_task"]:
        STATE["stop_task"].cancel()
        STATE["stop_task"] = None
    if STATE["unsubscribe_va"]:
        STATE["unsubscribe_va"]()
        STATE["unsubscribe_va"] = None
    if STATE["cli"]:
        await STATE["cli"].disconnect()
    STATE["cli"] = None
    STATE["connected"] = False
    STATE["capturing"] = False
    return JSONResponse({"status": "disconnected"})


@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "connected": STATE["connected"],
        "capturing": STATE["capturing"],
        "auto_capture": STATE["auto_capture"],
        "device": STATE["device_info"],
        "last_recording": STATE["last_recording"],
    })


@app.post("/api/auto_capture/{enabled}")
async def api_auto_capture(enabled: bool):
    STATE["auto_capture"] = enabled
    return JSONResponse({"auto_capture": STATE["auto_capture"]})


@app.get("/api/config")
async def api_get_config():
    return JSONResponse(CONFIG)


@app.post("/api/config")
async def api_set_config(body: dict):
    global CONFIG
    for key in ("device_host", "device_port", "asr_host", "asr_port"):
        if key in body:
            CONFIG[key] = body[key]
    if "filter_defaults" in body:
        CONFIG["filter_defaults"].update(body["filter_defaults"])
        FILTER_DEFAULTS.update(body["filter_defaults"])
    save_config(CONFIG)
    return JSONResponse(CONFIG)


@app.post("/api/asr")
async def api_asr(body: dict):
    audio_b64 = body.get("audio_b64")
    rec_id = body.get("rec_id")
    filters = body.get("filters")

    if audio_b64:
        wav_bytes = base64.b64decode(audio_b64)
    elif rec_id:
        samples = load_pcm(rec_id)
        if filters:
            samples = apply_filter_chain(samples, filters)
        wav_bytes = save_wav(pcm_to_bytes(samples))
    else:
        raise HTTPException(400, "Provide audio_b64 or rec_id")

    try:
        text = await transcribe_wyoming(wav_bytes)
        return JSONResponse({"text": text})
    except Exception as e:
        log.error("ASR error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


async def transcribe_wyoming(wav_bytes: bytes) -> str:
    from wyoming.client import AsyncTcpClient
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioStart, AudioStop, AudioChunk, wav_to_chunks

    host = CONFIG["asr_host"]
    port = CONFIG["asr_port"]
    log.info("ASR connecting to %s:%s", host, port)

    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Transcribe().event())
        with io.BytesIO(wav_bytes) as wav_buf:
            chunks = list(wav_to_chunks(wav_buf, chunk_seconds=2.0))

        if chunks:
            first = chunks[0]
            await client.write_event(AudioStart(
                rate=first.rate, width=first.width, channels=first.channels
            ).event())
            for chunk in chunks:
                await client.write_event(chunk.event())
            await client.write_event(AudioStop().event())

        while True:
            event = await client.read_event()
            if event is None:
                break
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                return transcript.text
    return ""


@app.get("/api/recordings")
async def api_recordings():
    recordings = []
    seen = set()
    for jf in sorted(RECORDINGS_DIR.glob("*.json"), reverse=True):
        wav_name = jf.name.replace(".json", "")
        wav_path = RECORDINGS_DIR / wav_name
        if not wav_path.exists():
            continue
        seen.add(wav_name)
        data = json.loads(jf.read_text(encoding="utf-8"))
        data["id"] = wav_name
        data["size_kb"] = round(wav_path.stat().st_size / 1024, 1)
        recordings.append(data)
    for wf in sorted(RECORDINGS_DIR.glob("*.wav"), reverse=True):
        if wf.name in seen:
            continue
        data = {"id": wf.name, "filename": wf.name, "size_kb": round(wf.stat().st_size / 1024, 1)}
        recordings.append(data)
    return JSONResponse(recordings)


@app.get("/api/recordings/{rec_id}/audio")
async def api_recording_audio(rec_id: str):
    path = RECORDINGS_DIR / rec_id
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path), media_type="audio/wav", filename=rec_id)


def load_pcm(rec_id: str) -> np.ndarray:
    path = RECORDINGS_DIR / rec_id
    if not path.exists():
        raise HTTPException(404)
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64)


def pcm_to_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -32768, 32767).astype(np.int16)
    return clipped.tobytes()


def samples_to_waveform(samples: np.ndarray) -> dict:
    downsample = max(1, len(samples) // 2000)
    down = samples[::downsample]
    return {
        "samples": down.tolist(),
        "total_samples": len(samples),
        "downsample": downsample,
    }


def apply_highpass(samples: np.ndarray, cutoff: float, fs: int = 16000) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff, btype="high", fs=fs, output="sos")
    return sosfilt(sos, samples)


def apply_lowpass(samples: np.ndarray, cutoff: float, fs: int = 16000) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff, btype="low", fs=fs, output="sos")
    return sosfilt(sos, samples)


def apply_noise_gate(samples: np.ndarray, threshold_db: float = -40) -> np.ndarray:
    rms = np.sqrt(np.mean(samples ** 2))
    if rms == 0:
        return samples
    threshold = 32768 * (10 ** (threshold_db / 20))
    env = np.abs(samples)
    win = int(16000 * 0.01)
    if win < 2:
        return samples
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")
    mask = env_smooth > threshold
    return samples * mask


def apply_spectral_subtraction(samples: np.ndarray, noise_frames: int = 5, fs: int = 16000) -> np.ndarray:
    from scipy.signal import stft, istft
    nperseg = 256
    f, t, Zxx = stft(samples, fs=fs, nperseg=nperseg)
    noise_profile = np.mean(np.abs(Zxx[:, :min(noise_frames, Zxx.shape[1])]), axis=1, keepdims=True)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    mag_clean = np.maximum(mag - noise_profile * 1.5, 0)
    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, reconstructed = istft(Zxx_clean, fs=fs, nperseg=nperseg)
    return reconstructed[:len(samples)]


def apply_normalize(samples: np.ndarray, target_db: float = -6) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    target = 32768 * (10 ** (target_db / 20))
    gain = target / peak
    return samples * gain


def apply_limiter(samples: np.ndarray, ceiling_db: float = -1) -> np.ndarray:
    ceiling = 32768 * (10 ** (ceiling_db / 20))
    out = np.copy(samples)
    over = np.abs(out) > ceiling
    out[over] = np.sign(out[over]) * ceiling
    return out


def apply_compressor(samples: np.ndarray, threshold_db: float = -20, ratio: float = 4.0, fs: int = 16000) -> np.ndarray:
    out = np.copy(samples)
    frame_len = int(fs * 0.01)
    for i in range(0, len(samples), frame_len):
        e = min(i + frame_len, len(samples))
        frame = out[i:e]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 1:
            continue
        rms_db = 20 * np.log10(rms / 32768)
        if rms_db > threshold_db:
            over = rms_db - threshold_db
            new_db = threshold_db + over / ratio
            gain = 10 ** ((new_db - rms_db) / 20)
            out[i:e] = frame * gain
    return out


def apply_agc(samples: np.ndarray, frame_ms: float = 20, target_rms_db: float = -20, fs: int = 16000) -> np.ndarray:
    frame_len = int(fs * frame_ms / 1000)
    target_rms = 32768 * (10 ** (target_rms_db / 20))
    out = np.copy(samples)
    n_frames = max(1, len(samples) // frame_len)
    for i in range(n_frames):
        s = i * frame_len
        e = min(s + frame_len, len(samples))
        frame = samples[s:e]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > 100:
            gain = min(target_rms / rms, 10.0)
            out[s:e] = frame * gain
    return out


FILTERS = {
    "highpass": apply_highpass,
    "lowpass": apply_lowpass,
    "noise_gate": apply_noise_gate,
    "spectral_subtraction": apply_spectral_subtraction,
    "normalize": apply_normalize,
    "limiter": apply_limiter,
    "compressor": apply_compressor,
    "agc": apply_agc,
}

FILTER_DEFAULTS = {
    "highpass": {"cutoff": 150},
    "lowpass": {"cutoff": 4000},
    "noise_gate": {"threshold_db": -40},
    "spectral_subtraction": {"noise_frames": 40},
    "normalize": {"target_db": -10},
    "limiter": {"ceiling_db": -1},
    "compressor": {"threshold_db": -20, "ratio": 4.0},
    "agc": {"frame_ms": 20, "target_rms_db": -20},
}

FILTER_DEFAULTS.update(CONFIG.get("filter_defaults", {}))


def apply_filter_chain(samples: np.ndarray, steps: list) -> np.ndarray:
    for step in steps:
        fname = step.get("filter") if isinstance(step, dict) else step
        params = step.get("params", {}) if isinstance(step, dict) else {}
        if fname not in FILTERS:
            raise HTTPException(400, f"Unknown filter: {fname}")
        defaults = FILTER_DEFAULTS.get(fname, {})
        kwargs = {**defaults, **params}
        samples = FILTERS[fname](samples, **kwargs)
    return samples


@app.post("/api/recordings/{rec_id}/filter")
async def api_filter(rec_id: str, body: dict = None):
    body = body or {}
    filter_names = body.get("filters", [])
    chain = body.get("chain", [])

    steps = []
    for fname in filter_names:
        steps.append({"filter": fname})
    steps.extend(chain)

    if not steps:
        raise HTTPException(400, "No filters specified")

    samples = load_pcm(rec_id)
    samples = apply_filter_chain(samples, steps)

    raw = pcm_to_bytes(samples)
    metrics = compute_metrics(raw)
    waveform = samples_to_waveform(samples)
    spectrogram = compute_spectrogram(raw)
    wav_b64 = base64.b64encode(save_wav(raw)).decode("ascii")
    return JSONResponse({
        "metrics": metrics,
        "waveform": waveform,
        "spectrogram": spectrogram,
        "audio_b64": wav_b64,
    })


@app.get("/api/filters")
async def api_list_filters():
    return JSONResponse([
        {"id": "highpass", "name": "High-pass", "desc": "Убирает низкочастотный гул",
         "params": {"cutoff": {"type": "int", "default": 150, "label": "Cutoff Hz", "min": 20, "max": 8000}}},
        {"id": "lowpass", "name": "Low-pass", "desc": "Оставляет только частоты ниже cutoff",
         "params": {"cutoff": {"type": "int", "default": 4000, "label": "Cutoff Hz", "min": 500, "max": 15999}}},
        {"id": "noise_gate", "name": "Noise Gate", "desc": "Обрезает тихие участки",
         "params": {"threshold_db": {"type": "float", "default": -40, "label": "Threshold dB", "min": -80, "max": 0}}},
        {"id": "spectral_subtraction", "name": "Spectral Subtraction", "desc": "Вычитает шум по спектру",
         "params": {"noise_frames": {"type": "int", "default": 5, "label": "Noise frames", "min": 1, "max": 50}}},
        {"id": "normalize", "name": "Normalize", "desc": "Нормализация громкости (peak)",
         "params": {"target_db": {"type": "float", "default": -6, "label": "Target dB", "min": -40, "max": 0}}},
        {"id": "limiter", "name": "Limiter", "desc": "Жёсткое ограничение пиков",
         "params": {"ceiling_db": {"type": "float", "default": -1, "label": "Ceiling dB", "min": -20, "max": 0}}},
        {"id": "compressor", "name": "Compressor", "desc": "Сжатие динамического диапазона",
         "params": {"threshold_db": {"type": "float", "default": -20, "label": "Threshold dB", "min": -60, "max": 0},
                    "ratio": {"type": "float", "default": 4.0, "label": "Ratio", "min": 1, "max": 20}}},
        {"id": "agc", "name": "AGC", "desc": "Автоматическая регулировка усиления",
         "params": {"target_rms_db": {"type": "float", "default": -20, "label": "Target RMS dB", "min": -40, "max": 0},
                    "frame_ms": {"type": "float", "default": 20, "label": "Frame ms", "min": 5, "max": 100}}},
    ])


@app.get("/api/recordings/{rec_id}/waveform")
async def api_recording_waveform(rec_id: str):
    samples = load_pcm(rec_id)
    return JSONResponse(samples_to_waveform(samples))


@app.get("/api/recordings/{rec_id}/metrics")
async def api_recording_metrics(rec_id: str):
    samples = load_pcm(rec_id)
    raw = pcm_to_bytes(samples)
    metrics = compute_metrics(raw)
    metrics["filename"] = rec_id
    return JSONResponse(metrics)


@app.get("/api/recordings/{rec_id}/spectrogram")
async def api_recording_spectrogram(rec_id: str):
    samples = load_pcm(rec_id)
    raw = pcm_to_bytes(samples)
    spec = compute_spectrogram(raw)
    return JSONResponse(spec or {"error": "too short"})


@app.delete("/api/recordings/{rec_id}")
async def api_recording_delete(rec_id: str):
    path = RECORDINGS_DIR / rec_id
    json_path = RECORDINGS_DIR / f"{rec_id}.json"
    deleted = 0
    if path.exists():
        path.unlink()
        deleted += 1
    if json_path.exists():
        json_path.unlink()
        deleted += 1
    return JSONResponse({"deleted": deleted})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8899)
