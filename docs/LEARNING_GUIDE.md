# 🎓 Learning Guide: Voice ML for Chinese TTS

This guide covers what you should learn and do before your first training session on your PC.

---

## 📝 Homework: Exercise Book Content (Priority 1)

**This is the most impactful thing you can do right now.** Type out your Chinese exercise book content into text files. The more content you prepare, the more audio you can generate once the model is trained.

### How to add vocab

Create files in `data/vocab/` — one file per lesson:

```
data/vocab/lesson01.txt
data/vocab/lesson02.txt
data/vocab/lesson03.txt
```

Format: `Chinese | Pinyin | English` (pinyin and English are optional, only Chinese text is used for audio)

```
你好 | nǐhǎo | hello
谢谢 | xièxie | thank you
学生 | xuéshēng | student
```

### How to add dialogues

Create files in `data/dialogues/` — one file per dialogue:

```
data/dialogues/lesson01_dialogue.txt
data/dialogues/lesson02_dialogue.txt
```

Format: one sentence per line (each line becomes one audio file)

```
你好！你叫什么名字？
我叫小明。你呢？
我也很高兴认识你。
```

### Tips
- **One sentence per line** — each line becomes a separate audio clip
- Keep sentences **under ~50 characters** for best quality
- For long dialogues, split into natural sentence breaks
- Use **standard simplified Chinese characters** (not traditional)
- You can include punctuation (。！？，) — the model handles it

---

## 📖 Concepts to Understand (Priority 2)

### What is TTS (Text-to-Speech)?
TTS converts written text into spoken audio. Modern TTS uses neural networks to generate natural-sounding speech, far beyond the robotic voices of older systems.

### What is Voice Cloning?
Voice cloning takes a **reference voice** (a few minutes of audio from someone) and learns to generate new speech that sounds like that person. That's what GPT-SoVITS does.

### How GPT-SoVITS Works (High Level)
GPT-SoVITS is a two-stage system:

1. **SoVITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)**
   - Learns the acoustic features of the reference voice
   - Converts text into mel-spectrograms (visual representations of sound)
   - Think of it as learning *how the voice sounds*

2. **GPT (Generative Pre-trained Transformer)**
   - Handles the prosody, rhythm, and natural flow of speech
   - Decides *how to say things naturally* (pauses, emphasis, intonation)
   - This is what makes Chinese tones sound correct

### What is Zero-Shot vs Few-Shot vs Fine-Tuning?

| Approach | Reference Audio | Quality | Our Use |
|----------|----------------|---------|---------|
| **Zero-shot** | 3-10 seconds | Good | Quick test |
| **Few-shot** | 30-60 seconds | Better | Not needed |
| **Fine-tuning** | 3-10 minutes | Best ✅ | Our approach |

We use **fine-tuning** (10 min of speaker A4 from THCHS-30) for maximum quality.

### Key ML Concepts

- **Epoch**: One complete pass through all training data
- **Batch size**: How many samples the GPU processes at once (we use 2 due to VRAM limits on RTX 4060)
- **Learning rate**: How fast the model adjusts its weights (too high = unstable, too low = slow)
- **Loss**: A number measuring how "wrong" the model is — you want this to go down during training
- **Overfitting**: When the model memorizes training data instead of generalizing (watch for loss going up on validation data)
- **Checkpoint**: A saved snapshot of the model at a point during training

---

## 🔧 Tools to Explore (Priority 3)

### 1. Listen to the Reference Audio
Play some of the files in `data/reference_audio/` to hear what speaker A4 sounds like. You can use any audio player:
```bash
# Play a file (on Ubuntu)
aplay data/reference_audio/A4_0.wav
# Or use VLC, Audacity, etc.
```

### 2. Read the Transcript List
Look at `data/reference_audio/transcript.list` to see the text that corresponds to each audio clip. This is what the model uses to learn the connection between text and speech.

### 3. Browse GPT-SoVITS Documentation
- **GitHub**: https://github.com/RVC-Boss/GPT-SoVITS
- **Wiki/Guide**: https://github.com/RVC-Boss/GPT-SoVITS/wiki
- **Video tutorials on YouTube**: Search "GPT-SoVITS Chinese TTS tutorial"
- **Bilibili (Chinese)**: Search "GPT-SoVITS 教程" for Chinese-language guides

### 4. Understand the Training Pipeline
When you're on your PC, the training steps will be:
```
Reference Audio (60 clips, 10 min)
        │
        ▼
    Data Preprocessing
    (audio slicing, denoising, ASR transcription)
        │
        ▼
    Feature Extraction
    (HuBERT, BERT features for each clip)
        │
        ▼
    SoVITS Training (~30-60 min on RTX 4060)
    (learns the voice characteristics)
        │
        ▼
    GPT Training (~30-60 min on RTX 4060)
    (learns natural speech patterns)
        │
        ▼
    Inference / Generation
    (type any Chinese text → get audio in that voice)
```

---

## 📚 Optional Deep Dives

If you want to go deeper, these resources are excellent:

### Audio/Speech Fundamentals
- [But what is a Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) (3Blue1Brown) — Foundational for understanding spectrograms
- [Mel Spectrograms Explained](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53) — What the model actually "sees"

### PyTorch Basics
- [PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA) (Fireship)
- [PyTorch Beginner Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html) (Official)
- You don't need to be a PyTorch expert — just understand tensors, models, and training loops

### Chinese NLP
- How Chinese text processing works (tokenization, pinyin conversion)
- Why Chinese TTS is harder than English (tones, characters, no spaces between words)

---

## ✅ Checklist: What to Do Before Your PC Session

- [ ] **Type out vocab** from your exercise book into `data/vocab/lessonXX.txt` files
- [ ] **Type out dialogues** from your exercise book into `data/dialogues/lessonXX.txt` files
- [ ] **Listen** to a few reference audio clips to hear speaker A4
- [ ] **Read** the GPT-SoVITS GitHub README
- [ ] **Skim** the concepts section above
- [ ] **Git push** your text files so they sync to your PC
- [ ] **Run** `scripts/setup_pc.sh` on your PC (one-time setup)
