# ESPHome Audio Debug Tool

Браузерный инструмент для захвата, анализа и обработки аудио с ESP32-S3 Voice Assistant через aioesphomeapi. Позволяет тестировать DSP фильтры до внедрения в прошивку ESPHome.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Возможности

- **Пассивный захват аудио** — автоматически записывает аудио при срабатывании wake word через `subscribe_voice_assistant(handle_audio=...)`
- **Waveform с зумом** — колесо мыши для зума к курсору, drag для панорамирования
- **Спектрограмма** — визуализация частотного спектра (heatmap)
- **Метрики** — RMS, Peak, SNR, duration
- **DSP фильтры** — 8 параметрических фильтров с side-by-side сравнением
- **ASR распознавание** — интеграция с Wyoming ONNX ASR (gigaam-v3-rnnt, русский язык)
- **Прослушивание** — Play для оригинального и отфильтрованного аудио
- **Настройки** — IP устройства, ASR хост/порт сохраняются в config.json

## Архитектура

```
┌─────────────────┐     aioesphomeapi      ┌──────────────────┐
│   ESP32-S3      │ ◄──────────────────►   │   server.py      │
│   Voice Assist  │   subscribe_voice_     │   (FastAPI)      │
│   ES7210 4ch    │   assistant(audio)     │                  │
└─────────────────┘                        │   Filters:       │
                                           │   scipy.signal    │
                                           │                  │
┌─────────────────┐     Wyoming TCP        │   ASR:           │
│   Wyoming       │ ◄──────────────────►   │   AsyncTcpClient │
│   ONNX ASR      │   Transcribe + Audio   │                  │
│   gigaam-v3     │                        └────────┬─────────┘
└─────────────────┘                                 │
                                                    │ HTTP
                                            ┌───────▼─────────┐
                                            │   Browser UI     │
                                            │   index.html     │
                                            │   waveform zoom  │
                                            │   spectrogram    │
                                            │   filter compare │
                                            └─────────────────┘
```

## Установка

### 1. Зависимости

```bash
pip install -r requirements.txt
```

Требования:
- Python 3.11+
- aioesphomeapi >= 29.0 (захват аудио)
- fastapi + uvicorn (веб-сервер)
- numpy + scipy (DSP фильтры и анализ)
- wyoming >= 1.5 (ASR клиент)

### 2. Wyoming ONNX ASR (опционально, для распознавания речи)

```bash
pip install "onnx-asr[cpu,hub]" wyoming
git clone https://github.com/mitrokun/wyoming_stt_onnxasr.git
cd wyoming_stt_onnxasr
set PYTHONPATH=.
python -m wyoming_onnxasr --model gigaam-v3-rnnt --uri "tcp://0.0.0.0:10306" --quantization int8
```

Модель скачается автоматически (~220MB int8). Дождитесь `Ready`.

### 3. Запуск

```bash
cd esphome-audio-debug
python server.py
```

Откройте http://localhost:8899

## Использование

### Захват аудио

1. Нажмите **Connect** (IP устройства настраивается в Settings)
2. Скажите wake word на устройстве
3. Аудио автоматически запишется (10 сек) и появится в списке Recordings
4. Кликните на запись — увидите waveform, спектрограмму и метрики

### DSP фильтры

Кликайте кнопки фильтров для добавления в цепочку. Активные фильтры подсвечиваются синим. Кликните ещё раз — уберётся.

| Фильтр | Параметры | Описание |
|---------|-----------|----------|
| **High-pass** | cutoff Hz (150) | Убирает низкочастотный гул |
| **Low-pass** | cutoff Hz (4000) | Оставляет только голос (0-4kHz) |
| **Noise Gate** | threshold dB (-40) | Обрезает тихие участки |
| **Spectral Sub** | noise_frames (40) | Вычитает шум по спектру |
| **Normalize** | target dB (-10) | Нормализация громкости |
| **Limiter** | ceiling dB (-1) | Ограничение пиков без клиппинга |
| **Compressor** | threshold dB, ratio | Сжатие динамического диапазона |
| **AGC** | target RMS dB, frame ms | Автоматическая регулировка усиления |

После выбора фильтров нажмите **Apply Chain** — увидите side-by-side сравнение:
- Waveform: оригинал (синий) vs отфильтрованный (зелёный)
- Спектрограмма: до и после
- Метрики: RMS, Peak, SNR с цветной индикацией (зелёный = лучше)

### ASR распознавание

- **ASR Original** — распознаёт исходное аудио
- **ASR** (в filtered panel) — распознаёт отфильтрованное аудио

Результат отображается в блоке ASR (Speech-to-Text).

### Settings

Кнопка **Settings** → поля для:
- ESP32 IP / Port
- ASR Host / Port (Wyoming)
- **Save as Default** — сохраняет в `config.json`, подтягивается при следующем запуске

## DSP фильтры — детали реализации

Все фильтры реализованы через `scipy.signal`:

- **High-pass / Low-pass** — Butterworth IIR 4-го порядка, `sosfilt`
- **Noise Gate** — огибающая через свёртку, порог в dB
- **Spectral Subtraction** — STFT, оценка шума по первым N фреймам, вычитание с oversubtraction factor 1.5
- **Normalize** — peak-based нормализация до target dB
- **Limiter** — жёсткое ограничение на ceiling dB
- **Compressor** — RMS-based покадровое сжатие (10ms фреймы)
- **AGC** — покадровый autogain с ограничением gain ×10

Рекомендуемая цепочка для ESP32-S3 с ES7210:
```
highpass:cutoff=150 → spectral_subtraction:noise_frames=40 → normalize:target_db=-10
```

## Совместимость

Требования к ESP32 устройству:
- ESP32-S3 с voice_assistant и `microphone: channels: 0` (API_AUDIO режим)
- ESPHome 2025.4+
- ES7210 TDM или другой I2S микрофон
- Ethernet или WiFi подключение

## Связанные проекты

- [ES7210-TDM](https://github.com/arturgadelshin/ES7210-TDM) — 4-канальный TDM микрофон для ESPHome
- [wyoming_stt_onnxasr](https://github.com/mitrokun/wyoming_stt_onnxasr) — Wyoming ONNX ASR сервер (gigaam, vosk, parakeet)
- [onnx-asr](https://github.com/istupakov/onnx-asr) — ONNX ASR движок

## Лицензия

MIT
