<<<<<<< HEAD
# 🎬 OptiView - Local Video SEO Auditor

A **100% offline** Video SEO analysis tool powered by local AI models via Ollama. Upload your video, and get an SEO score with actionable improvement tips.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Ollama](https://img.shields.io/badge/Ollama-Local_AI-green)

---

## 🌟 Features

- **🔒 100% Offline**: No cloud APIs, no data leaves your machine
- **👂 Audio Analysis**: Whisper transcribes all speech in your video
- **👁️ Visual Analysis**: LLaVA describes what's shown in key frames
- **🧠 SEO Grading**: Qwen2.5 provides strict relevance scoring and tips
- **📊 Beautiful UI**: Modern Streamlit interface with real-time progress

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Vision Model | LLaVA (via Ollama) |
| Logic Model | Qwen2.5 (via Ollama) |
| Audio Model | OpenAI Whisper (local) |
| Video Processing | OpenCV |

---

## 📋 Prerequisites

### 1. FFmpeg

FFmpeg is required for audio extraction from videos.

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

### 2. Ollama

Install Ollama and pull the required models.

```bash
# Install Ollama (visit https://ollama.ai/download for other methods)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In a new terminal, pull required models
ollama pull llava
ollama pull qwen2.5
```

---

## 🚀 Installation

1. **Clone/Navigate to the project directory**
   ```bash
   cd "Video Seo"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App

1. **Ensure Ollama is running** (in a separate terminal):
   ```bash
   ollama serve
   ```

2. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

---

## 📖 Usage

1. **Enter Target Keyword**: Type your video's target SEO keyword/niche in the sidebar
2. **Upload Video**: Select a `.mp4` or `.mov` file
3. **Click "Analyze Video"**: Wait for the 3-phase analysis:
   - 👂 **Phase A**: Audio transcription with Whisper
   - 👁️ **Phase B**: Visual frame analysis with LLaVA
   - 🧠 **Phase C**: SEO scoring with Qwen2.5
4. **Review Results**: Get your score and actionable tips!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT (.mp4/.mov)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               │
┌─────────────────┐ ┌─────────────────┐   │
│   PHASE A       │ │   PHASE B       │   │
│   THE EARS      │ │   THE EYES      │   │
│                 │ │                 │   │
│  Whisper Model  │ │  OpenCV + LLaVA │   │
│  (Audio → Text) │ │  (Frames → Desc)│   │
└────────┬────────┘ └────────┬────────┘   │
         │                   │            │
         │   ┌───────────────┘            │
         ▼   ▼                            │
┌─────────────────────────────────────────▼───────────────────┐
│                        PHASE C                               │
│                       THE BRAIN                              │
│                                                              │
│   Qwen2.5 (Transcript + Visuals + Keyword → SEO Analysis)   │
│                                                              │
│   Output: Relevance Score (0-100) + 3 Actionable Tips       │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot connect to Ollama" | Make sure `ollama serve` is running |
| "Missing models" | Run `ollama pull llava && ollama pull qwen2.5` |
| FFmpeg errors | Ensure FFmpeg is installed: `ffmpeg -version` |
| Slow processing | LLaVA analysis takes time; frames are extracted every 5s to optimize |

---

## 📄 License

This project is open source under the MIT License.

---

## 🙏 Credits

- [Ollama](https://ollama.ai) - Local LLM runtime
- [LLaVA](https://llava-vl.github.io/) - Vision-language model
- [Qwen](https://github.com/QwenLM/Qwen) - Language model for reasoning
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Streamlit](https://streamlit.io) - App framework
=======
# Sanjay---The-Divya-Drishti
Sanjaya • Divya Drishti: The Content Command Center for Creators. Uses local AI 'Divine Vision' to audit video content, predict virality, and generate strategic SEO roadmaps. 100% Offline &amp; Privacy-focused.
>>>>>>> 68a7311011668cc82c9557e40813f22c23206f4a
