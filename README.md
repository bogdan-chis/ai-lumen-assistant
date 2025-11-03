# AI Lumen Assistant 🦮🗺️🔊

Real-time **text→speech** from video for assistive use.
Runs **offline** with **EasyOCR** and a **local LLM via LM Studio**.

---

## ✨ Features

* 🧠 Frame gating by sharpness to skip blurry frames
* 🔎 Text box detection + **EasyOCR** recognition
* 🧾 Simple layout ordering → short, spoken summary
* 🗂️ Session memory (planned) + retrieval Q&A (optional)
* 🗣️ Low-latency TTS via **pyttsx3**
* 🤖 Local LLM with **LM Studio** (OpenAI-compatible API)

---

## 📁 Project layout

```
notebooks/            # experiments (optional)
src/
  capture.py          # video read + gating
  detect_text.py      # text region detection (EasyOCR polygons → boxes)
  ocr.py              # OCR runner
  layout.py           # reading order + summary
  memory.py           # timeline storage (planned)
  llm.py              # LM Studio OpenAI-compatible client
  tts.py              # text-to-speech
  ui.py               # draw boxes / small UI helpers
  app.py              # orchestrator
main.py               # CLI entry (file input only)
requirements.txt
README.md
```

---

## ⚙️ Requirements

**Python:** 3.10–3.12 on Windows 11.
**GPU:** optional. CPU works; GPU speeds up EasyOCR/Torch.

`requirements.txt`

```txt
opencv-python
numpy

easyocr
torch
torchvision
torchaudio

pyttsx3

openai>=1.40.0
```
---

## 🧪 Quick start

```powershell
# 1) Create env
python -m venv .venv
. .venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt
```

### 🤖 LM Studio setup

1. Open **LM Studio**, load a chat model, and start the **local server**.
2. Set env vars for the OpenAI-compatible client:

```powershell
$env:OPENAI_BASE_URL="http://localhost:1234/v1"
$env:OPENAI_API_KEY="lm-studio"                    # any non-empty string
$env:OPENAI_MODEL="<<exact model name in LM Studio>>"
```

### ▶️ Run on a video file

```powershell
python .\main.py "D:\data\challenge_color_848x480.mp4"
```

> 🎯 This app processes **files only**. It does not use the webcam.

---

## 🧱 Pipeline (current)

1. **Keyframe gate** → discard blurry frames.
2. **Text region detection** → EasyOCR polygons → `(x1,y1,x2,y2)` boxes.
3. **OCR** → text + confidence for each crop.
4. **Layout** → top-left reading order, short summary.
5. **TTS** → speak the summary.
6. **LLM** → LM Studio formats or answers short queries over captured text.

---

## 🛠️ Configuration

**LM Studio (required for LLM step):**

```
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=<model name as shown in LM Studio>
```

**Gating threshold** (in `src/capture.py`):

```python
is_usable(frame, blur_thresh=80.0)
```

---

## ⌨️ CLI hints

* Quit preview: `Esc` or `q`
* Optional keys can be added in `src/app.py` (e.g., trigger a canned question)

---

## 🧰 Troubleshooting

* ❌ **Video won’t open:** check full path and codec;
* 🔒 **LM Studio errors:** start local server, load a model, confirm env vars.
---
