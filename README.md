
# ** Movie Analytics Assignment – Complete End-To-End README**

## ** Author:** Utkarsh Suryaman

## ** Date:** November 2025

## ** Description:**

This repository contains two fully-automated pipelines for the interview assignment:

---

# ** Project Overview**

```
.
├── Dockerfile
├── requirement.txt
├── ques1
│   ├── character_screen_time_analysis.py
│   ├── movie_clip.mp4
│   └── report_out/
├── ques2
│   ├── subtitle_formatter.py
│   ├── movie_clip.mp4  (optional: Whisper auto-extracts transcript)
│   └── output.srt
└── .env   (created by you)
```

---

# ** TASK 1 — Character Screen-Time Analysis (Automated)**

Located in: `ques1/character_screen_time_analysis.py`

### **What this script does**

✔ Detects faces in the video using **MTCNN + InceptionResnetV1**
✔ Computes **face embeddings**
✔ Clusters characters using **DBSCAN**
✔ Calculates **total screen-time** for top-5 characters
✔ Extracts **top clips** for each character
✔ Generates Whisper transcript
✔ Finds **top punchy dialogues**
✔ Outputs everything into **Excel sheets**

### **Output Files**

Generated inside:

```
ques1/report_out/
    ├── character_screen_time_report.xlsx
    ├── extracted_audio.wav
    ├── clip files (optional)
```

---

# ** TASK 2 — Subtitle Generation + Translation (LLM + Whisper)**

Located in: `ques2/subtitle_formatter.py`

This script:

✔ Extracts transcript using **Whisper** (if no transcript provided)
✔ Sends transcript to LLM (OpenAI or Ollama)
✔ Cleans + formats subtitles with SRT rules
✔ Generates **translations** (Hindi/Telugu/etc.)
✔ Produces a **final bilingual SRT**

---

# ** Installation & Setup**

You can run everything using **Docker** (recommended)
or manually (Python environment).

---

# ** Docker Setup**

## **1. Create `.env` file (Required for OpenAI mode)**

Inside root:

```
OPENAI_API_KEY=your_key_here
OLLAMA_HOST=http://host.docker.internal:11434
```

(If you don't use OpenAI, just leave the key blank.)

---

## **2. Dockerfile (already created)**

Your existing Dockerfile builds the environment with Whisper, PyTorch, facenet-pytorch, OpenCV, moviepy, transformers, etc.

---

## **3. Build Docker Image**

```
docker build -t movie-assignment .
```

---

## **4. Run Container**

### **Option A — If using AVX CPU**

```
docker run -it --rm \
    -v "$(pwd)":/workspace \
    --gpus all \
    movie-assignment
```

### **Option B — Without GPU**

```
docker run -it --rm \
    -v "$(pwd)":/workspace \
    movie-assignment
```

---

# ** Running the Scripts**

---

# **TASK 1 — Screen-Time Analysis**

### **Command**

```
cd ques1
python character_screen_time_analysis.py \
    --video movie_clip.mp4 \
    --out_dir ./report_out \
    --whisper_model large \
    --fps 1.0
```

### **Output Includes**

✔ Top-5 characters
✔ Screen-time tables
✔ Clips per character
✔ Punchy dialogues
✔ Excel workbook with all sheets

---

# **TASK 2 — Subtitle Formatting + Translation**

### **Whisper auto-transcription added to script**

You can now run without transcript!!!

Just give the video file:

```
cd ques2
python subtitle_formatter.py \
    --transcript movie_clip.mp4 \
    --output output.srt \
    --llm-type openai \
    --model gpt-4o-mini \
    --target-lang Telugu
```

or with Ollama:

```
python subtitle_formatter.py \
    --transcript movie_clip.mp4 \
    --output output.srt \
    --llm-type ollama \
    --model llama3.1 \
    --target-lang Hindi
```

### **Features**

✔ Automatically runs Whisper to extract transcript
✔ Automatically formats subtitles (≤42 chars, ≤2 lines)
✔ Translates using chosen LLM
✔ Outputs bilingual SRT

### **Generated File**

```
ques2/output.srt
```

---

# ** Testing Whisper Transcription Only**

```
whisper myvideo.mp4 --model small
```

---

# ** Running Ollama (optional, without API key)**

You must install Ollama **locally (host machine), not inside Docker**.

### **Install Ollama**

Linux:

```
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```
ollama serve
```

Pull a model:

```
ollama pull llama3.1
```

Check status:

```
curl http://localhost:11434/api/tags
```

Your Docker container connects to Ollama using:

```
OLLAMA_HOST=http://host.docker.internal:11434
```

---

# ** requirement.txt**

```
opencv-python
face_recognition
moviepy==1.0.3
scikit-learn
pandas
openpyxl
openai-whisper
openai
tqdm
scikit-image
torchvision
transformers
sentence-transformers
facenet-pytorch
numpy
torch
```

---

# ** Troubleshooting Guide**

### **1. “Connection refused: localhost:11434”**

Ollama is not running on your host machine.

Run:

```
ollama serve
```

### **2. OpenAI key missing**

Ensure `.env` includes:

```
OPENAI_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434

```

### **3. CUDA not detected**

Use CPU mode by removing GPU flags:

```
docker run -it movie-assignment
```

### **4. Whisper too slow**

Use smaller model:

```
--whisper_model small
```

---

# ** Final Notes for Interviewer**

* The entire pipeline is **dockerized**, reproducible, and easy to run.
* Supports **both OpenAI API** and **local Ollama LLM**.
* Face clustering uses **MTCNN + InceptionResnetV1 embeddings + DBSCAN**.
* Subtitle processing uses **LLM formatting + translation**.
* Scripts are extensively logged and easy to follow.
