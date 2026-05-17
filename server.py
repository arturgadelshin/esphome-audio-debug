import asyncio
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
        info = await connect_device("192.168.1.48", 6053, None)
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


def apply_normalize(samples: np.ndarray, target_db: float = -3) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    target = 32768 * (10 ** (target_db / 20))
    gain = target / peak
    return samples * gain


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
    "highpass_150": lambda s: apply_highpass(s, 150),
    "highpass_300": lambda s: apply_highpass(s, 300),
    "lowpass_4000": lambda s: apply_lowpass(s, 4000),
    "noise_gate": lambda s: apply_noise_gate(s, -40),
    "spectral_subtraction": lambda s: apply_spectral_subtraction(s),
    "normalize": lambda s: apply_normalize(s, -3),
    "agc": lambda s: apply_agc(s),
}


@app.post("/api/recordings/{rec_id}/filter")
async def api_filter(rec_id: str, body: dict = None):
    body = body or {}
    filter_names = body.get("filters", [])
    chain = body.get("chain", None)

    samples = load_pcm(rec_id)

    if chain:
        for step in chain:
            fname = step.get("filter")
            if fname not in FILTERS:
                raise HTTPException(400, f"Unknown filter: {fname}")
            samples = FILTERS[fname](samples)
    else:
        for fname in filter_names:
            if fname not in FILTERS:
                raise HTTPException(400, f"Unknown filter: {fname}")
            samples = FILTERS[fname](samples)

    raw = pcm_to_bytes(samples)
    metrics = compute_metrics(raw)
    waveform = samples_to_waveform(samples)
    spectrogram = compute_spectrogram(raw)
    return JSONResponse({
        "metrics": metrics,
        "waveform": waveform,
        "spectrogram": spectrogram,
    })


@app.get("/api/filters")
async def api_list_filters():
    return JSONResponse([
        {"id": "highpass_150", "name": "High-pass 150Hz", "desc": "Убирает низкочастотный гул"},
        {"id": "highpass_300", "name": "High-pass 300Hz", "desc": "Агрессивнее, убирает басы"},
        {"id": "lowpass_4000", "name": "Low-pass 4kHz", "desc": "Оставляет только голос (0-4kHz)"},
        {"id": "noise_gate", "name": "Noise Gate -40dB", "desc": "Обрезает тихие участки"},
        {"id": "spectral_subtraction", "name": "Spectral Subtraction", "desc": "Вычитает шум по спектру"},
        {"id": "normalize", "name": "Normalize -3dB", "desc": "Нормализация громкости"},
        {"id": "agc", "name": "AGC", "desc": "Автоматическая регулировка усиления"},
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
