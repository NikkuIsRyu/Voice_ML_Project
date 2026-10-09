// Chinese Pronunciation Tutor — Client Application

let currentLessons = [];
let currentWords = [];
let currentWordIndex = 0;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

let dialogueData = null;
let dialogueLines = [];
let dialogueTurnIndex = 0;
let userRole = "";

const audioPlayer = document.getElementById("audio-player");

// DOM Elements
const tabBtns = document.querySelectorAll(".nav-tab");
const tabContents = document.querySelectorAll(".tab-content");

const practiceModeSelect = document.getElementById("practice-mode-select");
const lessonGroup = document.getElementById("lesson-group");
const lessonSelect = document.getElementById("lesson-select");
const wordSelect = document.getElementById("word-select");
const prevWordBtn = document.getElementById("prev-word-btn");
const nextWordBtn = document.getElementById("next-word-btn");

const vocabSyllablesDisplay = document.getElementById("vocab-syllables-display");
const vocabHanzi = document.getElementById("vocab-hanzi");
const vocabPinyin = document.getElementById("vocab-pinyin");
const vocabTrans = document.getElementById("vocab-translation");
const playNativeBtn = document.getElementById("play-native-btn");
const playSlowBtn = document.getElementById("play-slow-btn");
const playUserBtn = document.getElementById("play-user-btn");
const recordBtn = document.getElementById("record-btn");
const recordText = document.getElementById("record-text");
const recordingIndicator = document.getElementById("recording-indicator");

const vocabResults = document.getElementById("vocab-results");
const resOverall = document.getElementById("res-overall");
const resTone = document.getElementById("res-tone");
const resChar = document.getElementById("res-char");
const resPitch = document.getElementById("res-pitch");
const diagnosticsList = document.getElementById("diagnostics-list");
const pitchCanvas = document.getElementById("pitch-canvas");

const vocabErrorBanner = document.getElementById("vocab-error-banner");
const vocabErrorText = document.getElementById("vocab-error-text");
const vocabErrorClose = document.getElementById("vocab-error-close");
const vocabScoringLoader = document.getElementById("vocab-scoring-loader");
let recordedMimeType = "";
let lastUserAudioUrl = null;

function showVocabError(msg) {
  if (vocabErrorText) vocabErrorText.textContent = msg;
  if (vocabErrorBanner) vocabErrorBanner.classList.remove("hidden");
  if (vocabScoringLoader) vocabScoringLoader.classList.add("hidden");
  if (vocabResults) vocabResults.classList.add("hidden");
}

function hideVocabError() {
  if (vocabErrorBanner) vocabErrorBanner.classList.add("hidden");
}

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadLessons();
  setupAudioRecorder();
  setupKeyboardShortcuts();
  setupQueueMode();
  setupDialogueTab();
  loadStats();
  if (vocabErrorClose) {
    vocabErrorClose.addEventListener("click", hideVocabError);
  }
});

// --- Tabs Navigation ---
function setupTabs() {
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${tab}`).classList.add("active");

      if (tab === "stats") loadStats();
      if (tab === "dialogue" && !dialogueData) loadDialogues();
    });
  });
}

// --- Vocabulary Practice ---
async function loadLessons() {
  try {
    const res = await fetch("/api/lessons");
    const data = await res.json();
    currentLessons = data.lessons;
    lessonSelect.innerHTML = "";
    currentLessons.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l.id;
      opt.textContent = `${l.title} (${l.word_count} words)`;
      lessonSelect.appendChild(opt);
    });

    if (currentLessons.length > 0) {
      loadWordsForLesson(currentLessons[0].id);
    }
  } catch (err) {
    console.error("Failed to load lessons:", err);
  }
}

lessonSelect.addEventListener("change", (e) => {
  loadWordsForLesson(e.target.value);
});

async function loadWordsForLesson(lessonId) {
  try {
    const res = await fetch(`/api/lessons/${lessonId}`);
    const data = await res.json();
    currentWords = data.words;
    currentWordIndex = 0;

    wordSelect.innerHTML = "";
    currentWords.forEach((w, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${w.chinese} (${w.pinyin})`;
      wordSelect.appendChild(opt);
    });

    displayCurrentWord();
  } catch (err) {
    console.error("Failed to load words:", err);
  }
}

wordSelect.addEventListener("change", (e) => {
  currentWordIndex = parseInt(e.target.value, 10);
  displayCurrentWord();
});

prevWordBtn.addEventListener("click", () => {
  if (currentWordIndex > 0) {
    currentWordIndex--;
    wordSelect.value = currentWordIndex;
    displayCurrentWord();
  }
});

nextWordBtn.addEventListener("click", () => {
  if (currentWordIndex < currentWords.length - 1) {
    currentWordIndex++;
    wordSelect.value = currentWordIndex;
    displayCurrentWord();
  }
});

function displayCurrentWord() {
  const w = currentWords[currentWordIndex];
  if (!w) return;

  vocabHanzi.textContent = w.chinese;
  vocabPinyin.textContent = w.pinyin;
  vocabTrans.textContent = w.translation;

  vocabResults.classList.add("hidden");
  if (vocabSyllablesDisplay) {
    vocabSyllablesDisplay.classList.add("hidden");
    vocabSyllablesDisplay.innerHTML = "";
  }
  if (playUserBtn) playUserBtn.classList.add("hidden");
  lastUserAudioUrl = null;

  playNativeBtn.disabled = !w.audio_url;
  if (playSlowBtn) playSlowBtn.disabled = !w.audio_url;
}

playNativeBtn.addEventListener("click", () => {
  const w = currentWords[currentWordIndex];
  if (w && w.audio_url) {
    audioPlayer.src = w.audio_url;
    audioPlayer.playbackRate = 1.0;
    audioPlayer.play();
  }
});

if (playSlowBtn) {
  playSlowBtn.addEventListener("click", () => {
    const w = currentWords[currentWordIndex];
    if (w && w.audio_url) {
      audioPlayer.src = w.audio_url;
      audioPlayer.playbackRate = 0.75;
      audioPlayer.play();
      audioPlayer.onended = () => {
        audioPlayer.playbackRate = 1.0;
      };
    }
  });
}

if (playUserBtn) {
  playUserBtn.addEventListener("click", () => {
    if (lastUserAudioUrl) {
      audioPlayer.src = lastUserAudioUrl;
      audioPlayer.playbackRate = 1.0;
      audioPlayer.play();
    }
  });
}

// --- Audio Recording & Scoring ---
async function setupAudioRecorder() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    recordBtn.disabled = true;
    recordText.textContent = "Mic Not Supported";
    return;
  }

  recordBtn.addEventListener("click", async () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });
}

async function startRecording() {
  hideVocabError();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];

    let mimeType = "";
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
      "audio/mp4"
    ];
    for (const t of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) {
        mimeType = t;
        break;
      }
    }
    recordedMimeType = mimeType || "";
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const actualMime = mediaRecorder.mimeType || recordedMimeType || "audio/webm";
      const audioBlob = new Blob(audioChunks, { type: actualMime });
      submitVocabAudio(audioBlob, actualMime);
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add("recording");
    recordText.textContent = "Stop & Score";
    recordingIndicator.classList.remove("hidden");
  } catch (err) {
    showVocabError("Microphone access denied or error: " + err);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordText.textContent = "Record Voice";
    recordingIndicator.classList.add("hidden");
  }
}

async function submitVocabAudio(blob, mimeType = "") {
  const w = currentWords[currentWordIndex];
  if (!w) return;

  hideVocabError();
  vocabResults.classList.add("hidden");
  if (vocabScoringLoader) vocabScoringLoader.classList.remove("hidden");

  let ext = ".webm";
  if (mimeType.includes("ogg")) ext = ".ogg";
  else if (mimeType.includes("mp4")) ext = ".mp4";
  else if (mimeType.includes("wav")) ext = ".wav";

  const formData = new FormData();
  formData.append("audio", blob, `user_recording${ext}`);
  formData.append("item_id", w.id);
  formData.append("target_text", w.chinese);
  formData.append("target_pinyin", w.pinyin);
  formData.append("audio_filename", w.audio_filename || "");

  recordText.textContent = "Scoring Voice...";
  recordBtn.disabled = true;

  try {
    const res = await fetch("/api/score/vocab", {
      method: "POST",
      body: formData
    });

    if (vocabScoringLoader) vocabScoringLoader.classList.add("hidden");

    if (!res.ok) {
      let errorMsg = `Evaluation failed (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) errorMsg = errJson.detail;
      } catch (e) {}
      showVocabError(errorMsg);
      return;
    }

    const result = await res.json();
    renderVocabResults(result);
  } catch (err) {
    if (vocabScoringLoader) vocabScoringLoader.classList.add("hidden");
    showVocabError("Network error connecting to scoring server: " + err);
  } finally {
    recordText.textContent = "Record Voice";
    recordBtn.disabled = false;
  }
}

function renderVocabResults(res) {
  if (!res || typeof res !== "object") {
    showVocabError("Invalid scoring response received from server.");
    return;
  }

  vocabResults.classList.remove("hidden");

  const overall = Number.isFinite(res.overall_score) ? `${Math.round(res.overall_score)}%` : "—";
  const tone = Number.isFinite(res.tone_accuracy) ? `${Math.round(res.tone_accuracy)}%` : "—";
  const char = Number.isFinite(res.char_accuracy) ? `${Math.round(res.char_accuracy)}%` : "—";
  const pitch = Number.isFinite(res.pitch_score) ? `${Math.round(res.pitch_score)}%` : "—";

  resOverall.textContent = overall;
  resTone.textContent = tone;
  resChar.textContent = char;
  resPitch.textContent = pitch;

  // Color overall score
  if (Number.isFinite(res.overall_score)) {
    if (res.overall_score >= 85) resOverall.style.color = "var(--accent-green)";
    else if (res.overall_score >= 65) resOverall.style.color = "var(--accent-gold)";
    else resOverall.style.color = "var(--accent-red)";
  } else {
    resOverall.style.color = "var(--text-muted)";
  }

  // Diagnostics breakdown
  diagnosticsList.innerHTML = "";
  (res.diagnostics || []).forEach(d => {
    const item = document.createElement("div");
    item.className = "diagnostic-item";

    const pillClass = d.status === "correct" ? "correct" : (d.status === "tone_error" ? "tone_error" : "pronunciation_error");
    const pillLabel = d.status === "correct" ? "Correct" : (d.status === "tone_error" ? "Tone Error" : "Mispronounced");

    item.innerHTML = `
      <span class="diag-pill ${pillClass}">${pillLabel}</span>
      <strong>${d.char} (${d.target_pinyin || ""})</strong>:
      <span>${d.comment || ""}</span>
    `;
    diagnosticsList.appendChild(item);
  });

  // Render Syllables with Tone and Pronunciation Highlights
  if (vocabSyllablesDisplay) {
    if (res.syllables && Array.isArray(res.syllables) && res.syllables.length > 0) {
      vocabSyllablesDisplay.innerHTML = "";
      vocabSyllablesDisplay.classList.remove("hidden");
      res.syllables.forEach(s => {
        const card = document.createElement("div");
        let statusClass = "correct";
        if (s.issue === "tone") statusClass = "tone_error";
        else if (s.issue) statusClass = "pronunciation_error";

        card.className = `syllable-card ${statusClass}`;
        const tip = s.issue
          ? `Expected ${s.expected} (T${s.expected_tone}) — heard ${s.heard_tone ? 'T'+s.heard_tone : '?'}: ${s.advice || s.issue}`
          : `${s.expected} (T${s.expected_tone}): Correct!`;

        card.innerHTML = `
          <span class="syl-char">${s.char}</span>
          <span class="syl-pinyin">${s.expected}</span>
          <span class="syl-tooltip">${tip}</span>
        `;
        vocabSyllablesDisplay.appendChild(card);
      });
    } else {
      vocabSyllablesDisplay.classList.add("hidden");
    }
  }

  // Update Hear Me (User Recording playback)
  if (res.user_audio_url) {
    lastUserAudioUrl = res.user_audio_url;
    if (playUserBtn) playUserBtn.classList.remove("hidden");
  }

  // Render Pitch Canvas (hidden until valid data arrives)
  const pitchSec = document.querySelector(".pitch-visualizer-section");
  if (res.pitch_curves && Array.isArray(res.pitch_curves.aligned_ref) && res.pitch_curves.aligned_ref.length > 1) {
    if (pitchSec) pitchSec.classList.remove("hidden");
    drawPitchCanvas(res.pitch_curves.aligned_ref, res.pitch_curves.aligned_user);
  } else {
    if (pitchSec) pitchSec.classList.add("hidden");
  }
}

function drawPitchCanvas(refCurve, userCurve) {
  const ctx = pitchCanvas.getContext("2d");
  const W = pitchCanvas.width;
  const H = pitchCanvas.height;

  ctx.clearRect(0, 0, W, H);

  // Background grid
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  for (let y = 30; y < H; y += 35) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  // Zero-line (mean pitch)
  ctx.strokeStyle = "#334155";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!refCurve || refCurve.length < 2) return;

  const N = Math.max(refCurve.length, (userCurve || []).length);
  const padX = 20;
  const usableW = W - padX * 2;
  const midY = H / 2;
  const scaleY = 16; // 16 pixels per semitone

  function plotCurve(curve, color, width) {
    if (!curve || curve.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();

    curve.forEach((val, idx) => {
      const x = padX + (idx / (curve.length - 1)) * usableW;
      const y = midY - val * scaleY; // Invert: higher pitch is up
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Draw Gold Reference Curve (Native Speaker)
  plotCurve(refCurve, "#fbbf24", 4);

  // Draw Cyan User Curve (Student Microphone)
  if (userCurve && userCurve.length > 1) {
    plotCurve(userCurve, "#38bdf8", 3);
  }
}

// --- Dialogue Role-Play ---
const dialogueSelect = document.getElementById("dialogue-select");
const roleSelect = document.getElementById("role-select");
const startDialogueBtn = document.getElementById("start-dialogue-btn");
const dialogueHistory = document.getElementById("dialogue-history");
const dialogueTurnControls = document.getElementById("dialogue-turn-controls");
const turnSpeaker = document.getElementById("turn-speaker");
const turnText = document.getElementById("turn-text");
const dialogueRecordBtn = document.getElementById("dialogue-record-btn");
const dialogueRecordText = document.getElementById("dialogue-record-text");
const dialogueTryAgainBtn = document.getElementById("dialogue-try-again-btn");
const dialogueListenRefBtn = document.getElementById("dialogue-listen-ref-btn");

let dialogueMediaRecorder = null;
let dialogueAudioChunks = [];
let isDialogueRecording = false;

function setupDialogueTab() {
  dialogueSelect.addEventListener("change", (e) => {
    updateDialogueRoles(e.target.value);
  });

  startDialogueBtn.addEventListener("click", () => {
    startDialogueRoleplay();
  });
}

async function loadDialogues() {
  try {
    const res = await fetch("/api/dialogues");
    const data = await res.json();
    dialogueData = data.dialogues;

    dialogueSelect.innerHTML = "";
    dialogueData.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.id;
      opt.textContent = `${d.title} (${d.lines_count} lines)`;
      dialogueSelect.appendChild(opt);
    });

    if (dialogueData.length > 0) {
      updateDialogueRoles(dialogueData[0].id);
    }
  } catch (err) {
    console.error("Failed to load dialogues:", err);
  }
}

function updateDialogueRoles(diagId) {
  const d = dialogueData.find(x => x.id === diagId);
  roleSelect.innerHTML = "";
  if (d && d.roles) {
    d.roles.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = `Play as ${r}`;
      roleSelect.appendChild(opt);
    });
  }
}

async function startDialogueRoleplay() {
  const diagId = dialogueSelect.value;
  userRole = roleSelect.value;

  const res = await fetch(`/api/dialogues/${diagId}`);
  const data = await res.json();
  dialogueLines = data.lines;
  dialogueTurnIndex = 0;

  dialogueHistory.innerHTML = "";
  dialogueTurnControls.classList.add("hidden");

  advanceDialogueTurn();
}

function advanceDialogueTurn() {
  if (dialogueTurnIndex >= dialogueLines.length) {
    const endMsg = document.createElement("div");
    endMsg.className = "dialogue-bubble partner";
    endMsg.innerHTML = "<strong>Scenario Finished!</strong> Well done completing this conversation role-play.";
    dialogueHistory.appendChild(endMsg);
    dialogueTurnControls.classList.add("hidden");
    return;
  }

  const line = dialogueLines[dialogueTurnIndex];

  if (line.speaker !== userRole) {
    // Partner's turn: Add bubble and speak native audio
    dialogueTurnControls.classList.add("hidden");
    const bubble = document.createElement("div");
    bubble.className = "dialogue-bubble partner";
    bubble.innerHTML = `<strong>${line.speaker}:</strong> ${line.text}`;
    dialogueHistory.appendChild(bubble);
    dialogueHistory.scrollTop = dialogueHistory.scrollHeight;

    if (line.audio_url) {
      audioPlayer.src = line.audio_url;
      audioPlayer.play();
      audioPlayer.onended = () => {
        dialogueTurnIndex++;
        advanceDialogueTurn();
      };
    } else {
      setTimeout(() => {
        dialogueTurnIndex++;
        advanceDialogueTurn();
      }, 2000);
    }
  } else {
    // User's turn
    dialogueTurnControls.classList.remove("hidden");
    turnSpeaker.textContent = `${userRole}:`;
    turnText.textContent = line.text;

    setupDialogueTurnRecording(line);
  }
}

function setupDialogueTurnRecording(line) {
  if (dialogueTryAgainBtn) dialogueTryAgainBtn.classList.add("hidden");
  dialogueRecordBtn.disabled = false;
  dialogueRecordBtn.classList.remove("recording");
  if (dialogueRecordText) dialogueRecordText.textContent = "Speak This Line";

  dialogueListenRefBtn.onclick = () => {
    if (line.audio_url) {
      audioPlayer.src = line.audio_url;
      audioPlayer.playbackRate = 1.0;
      audioPlayer.play();
    }
  };

  dialogueRecordBtn.onclick = async () => {
    if (!isDialogueRecording) {
      await startDialogueRecording();
    } else {
      stopDialogueRecording(line);
    }
  };

  if (dialogueTryAgainBtn) {
    dialogueTryAgainBtn.onclick = () => {
      // Remove last user bubble to re-record
      const lastBubble = dialogueHistory.lastElementChild;
      if (lastBubble && lastBubble.classList.contains("user")) {
        dialogueHistory.removeChild(lastBubble);
      }
      setupDialogueTurnRecording(line);
    };
  }
}

async function startDialogueRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    dialogueAudioChunks = [];

    let mimeType = "";
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
      "audio/mp4"
    ];
    for (const t of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) {
        mimeType = t;
        break;
      }
    }
    dialogueMediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    dialogueMediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) dialogueAudioChunks.push(e.data);
    };

    dialogueMediaRecorder.start();
    isDialogueRecording = true;
    dialogueRecordBtn.classList.add("recording");
    if (dialogueRecordText) dialogueRecordText.textContent = "Stop & Submit";
  } catch (err) {
    alert("Microphone access error: " + err);
  }
}

function stopDialogueRecording(line) {
  if (dialogueMediaRecorder && isDialogueRecording) {
    dialogueMediaRecorder.onstop = async () => {
      const actualMime = dialogueMediaRecorder.mimeType || "audio/webm";
      const audioBlob = new Blob(dialogueAudioChunks, { type: actualMime });
      await submitDialogueAudio(audioBlob, line, actualMime);
    };
    dialogueMediaRecorder.stop();
    dialogueMediaRecorder.stream.getTracks().forEach(t => t.stop());
    isDialogueRecording = false;
    dialogueRecordBtn.classList.remove("recording");
    if (dialogueRecordText) dialogueRecordText.textContent = "Scoring Turn...";
    dialogueRecordBtn.disabled = true;
  }
}

async function submitDialogueAudio(blob, line, mimeType) {
  let ext = ".webm";
  if (mimeType.includes("ogg")) ext = ".ogg";
  else if (mimeType.includes("mp4")) ext = ".mp4";
  else if (mimeType.includes("wav")) ext = ".wav";

  const formData = new FormData();
  formData.append("audio", blob, `dialogue_recording${ext}`);
  formData.append("dialogue_id", dialogueSelect.value);
  formData.append("line_index", dialogueTurnIndex);
  formData.append("target_text", line.text);

  try {
    const res = await fetch("/api/score/dialogue", {
      method: "POST",
      body: formData
    });

    dialogueRecordBtn.disabled = false;
    if (dialogueRecordText) dialogueRecordText.textContent = "Speak This Line";

    if (!res.ok) {
      let errMsg = "Evaluation error";
      try {
        const j = await res.json();
        if (j.detail) errMsg = j.detail;
      } catch (e) {}
      alert(`Scoring failed: ${errMsg}`);
      return;
    }

    const data = await res.json();
    const score = Math.round(data.overall_score || 0);
    const pillClass = score >= 80 ? "correct" : (score >= 60 ? "tone_error" : "pronunciation_error");

    let syllableHtml = "";
    if (data.syllables && data.syllables.length > 0) {
      syllableHtml = `<div style="font-size:12px; margin-top:6px; opacity:0.9;">` +
        data.syllables.map(s => {
          const col = s.issue ? (s.issue === 'tone' ? 'var(--accent-gold)' : 'var(--accent-red)') : 'var(--accent-green)';
          return `<span style="color:${col}; font-weight:600; margin-right:4px;">${s.char}(${s.expected}${s.heard_tone && s.heard_tone !== s.expected_tone ? '→T'+s.heard_tone : ''})</span>`;
        }).join(" ") + `</div>`;
    }

    const bubble = document.createElement("div");
    bubble.className = "dialogue-bubble user";
    bubble.innerHTML = `
      <div><strong>You (${userRole}):</strong> ${line.text} <span class="diag-pill ${pillClass}" style="margin-left:8px;">${score}%</span></div>
      ${syllableHtml}
    `;
    dialogueHistory.appendChild(bubble);
    dialogueHistory.scrollTop = dialogueHistory.scrollHeight;

    if (score < 70) {
      if (dialogueTryAgainBtn) dialogueTryAgainBtn.classList.remove("hidden");
      const continueBtn = document.createElement("button");
      continueBtn.className = "btn btn-secondary";
      continueBtn.id = "dialogue-continue-anyway-btn";
      continueBtn.style.marginLeft = "8px";
      continueBtn.textContent = "Continue ⏭️";
      continueBtn.onclick = () => {
        continueBtn.remove();
        if (dialogueTryAgainBtn) dialogueTryAgainBtn.classList.add("hidden");
        dialogueTurnIndex++;
        advanceDialogueTurn();
      };
      const turnActions = dialogueTurnControls.querySelector(".turn-actions");
      const existingCont = document.getElementById("dialogue-continue-anyway-btn");
      if (existingCont) existingCont.remove();
      if (turnActions) turnActions.appendChild(continueBtn);
    } else {
      const existingCont = document.getElementById("dialogue-continue-anyway-btn");
      if (existingCont) existingCont.remove();
      setTimeout(() => {
        dialogueTurnIndex++;
        advanceDialogueTurn();
      }, 1200);
    }
  } catch (err) {
    alert("Network error scoring dialogue: " + err);
    dialogueRecordBtn.disabled = false;
    if (dialogueRecordText) dialogueRecordText.textContent = "Speak This Line";
  }
}

// --- Practice Flow Keyboard Shortcuts ---
function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;

    if (e.code === "Space") {
      e.preventDefault();
      const vocabTab = document.getElementById("tab-vocab");
      const dialTab = document.getElementById("tab-dialogue");
      if (vocabTab && vocabTab.classList.contains("active")) {
        if (!isRecording) startRecording();
        else stopRecording();
      } else if (dialTab && dialTab.classList.contains("active") && !dialogueTurnControls.classList.contains("hidden")) {
        if (!isDialogueRecording) startDialogueRecording();
        else {
          const line = dialogueLines[dialogueTurnIndex];
          if (line) stopDialogueRecording(line);
        }
      }
    } else if (e.code === "Enter" || e.code === "ArrowRight") {
      const vocabTab = document.getElementById("tab-vocab");
      if (vocabTab && vocabTab.classList.contains("active")) {
        e.preventDefault();
        nextWordBtn.click();
      }
    } else if (e.code === "ArrowLeft") {
      const vocabTab = document.getElementById("tab-vocab");
      if (vocabTab && vocabTab.classList.contains("active")) {
        e.preventDefault();
        prevWordBtn.click();
      }
    }
  });
}

// --- SRS Due Queue Mode ---
function setupQueueMode() {
  if (!practiceModeSelect) return;
  practiceModeSelect.addEventListener("change", async (e) => {
    const mode = e.target.value;
    if (mode === "due") {
      if (lessonGroup) lessonGroup.classList.add("hidden");
      try {
        const res = await fetch("/api/srs/due");
        const data = await res.json();
        const dueWords = data.words || [];
        if (dueWords.length === 0) {
          showVocabError("No SRS words due for review right now! Showing lesson mode.");
          practiceModeSelect.value = "lesson";
          if (lessonGroup) lessonGroup.classList.remove("hidden");
          return;
        }
        currentWords = dueWords.map(w => ({
          id: w.item_id,
          chinese: w.text,
          pinyin: w.pinyin,
          translation: w.translation,
          audio_url: w.audio_url,
          audio_filename: w.audio_filename
        }));
        currentWordIndex = 0;
        wordSelect.innerHTML = "";
        currentWords.forEach((w, idx) => {
          const opt = document.createElement("option");
          opt.value = idx;
          opt.textContent = `${w.chinese} (${w.pinyin})`;
          wordSelect.appendChild(opt);
        });
        displayCurrentWord();
      } catch (err) {
        showVocabError("Failed to fetch SRS due words: " + err);
      }
    } else {
      if (lessonGroup) lessonGroup.classList.remove("hidden");
      if (currentLessons.length > 0) {
        loadWordsForLesson(lessonSelect.value || currentLessons[0].id);
      }
    }
  });
}

// --- SRS & Dashboard Stats ---
async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    document.getElementById("stat-total").textContent = data.total_words;
    document.getElementById("stat-mastered").textContent = data.mastered_words;
    document.getElementById("stat-due").textContent = data.due_words;
    document.getElementById("stat-tone-acc").textContent = `${data.average_tone_accuracy}%`;

    const tbody = document.getElementById("recent-records-body");
    tbody.innerHTML = "";

    if (!data.recent_attempts || data.recent_attempts.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No practice records yet. Record some words!</td></tr>`;
    } else {
      data.recent_attempts.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${r.target_text}</strong></td>
          <td>${r.target_pinyin}</td>
          <td>${r.recognized_text || "-"}</td>
          <td>${r.tone_accuracy}%</td>
          <td><span class="diag-pill ${r.overall_score >= 80 ? 'correct' : (r.overall_score >= 60 ? 'tone_error' : 'pronunciation_error')}">${r.overall_score}%</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Load weakness profile
    const wRes = await fetch("/api/stats/weakness");
    if (wRes.ok) {
      const wData = await wRes.json();
      const confContainer = document.getElementById("tone-confusions-list");
      if (confContainer) {
        confContainer.innerHTML = "";
        if (!wData.top_tone_confusions || wData.top_tone_confusions.length === 0) {
          confContainer.innerHTML = `<p class="text-muted">No tone confusions recorded yet.</p>`;
        } else {
          wData.top_tone_confusions.forEach(c => {
            const item = document.createElement("div");
            item.className = "weakness-item";
            item.innerHTML = `
              <span><strong>Tone ${c.expected_tone}</strong> heard as <strong>Tone ${c.heard_tone}</strong></span>
              <span class="weakness-badge">${c.count}×</span>
            `;
            confContainer.appendChild(item);
          });
        }
      }

      const probContainer = document.getElementById("problem-syllables-list");
      if (probContainer) {
        probContainer.innerHTML = "";
        if (!wData.top_problem_syllables || wData.top_problem_syllables.length === 0) {
          probContainer.innerHTML = `<p class="text-muted">No repeated errors recorded yet.</p>`;
        } else {
          wData.top_problem_syllables.forEach(p => {
            const item = document.createElement("div");
            item.className = "weakness-item";
            item.innerHTML = `
              <span><strong>${p.syllable}</strong> (${p.issue})</span>
              <span class="weakness-badge">${p.count}×</span>
            `;
            probContainer.appendChild(item);
          });
        }
      }
    }
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}
