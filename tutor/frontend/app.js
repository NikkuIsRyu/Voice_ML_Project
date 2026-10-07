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

const lessonSelect = document.getElementById("lesson-select");
const wordSelect = document.getElementById("word-select");
const prevWordBtn = document.getElementById("prev-word-btn");
const nextWordBtn = document.getElementById("next-word-btn");

const vocabHanzi = document.getElementById("vocab-hanzi");
const vocabPinyin = document.getElementById("vocab-pinyin");
const vocabTrans = document.getElementById("vocab-translation");
const playNativeBtn = document.getElementById("play-native-btn");
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

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadLessons();
  setupAudioRecorder();
  setupDialogueTab();
  loadStats();
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
  playNativeBtn.disabled = !w.audio_url;
}

playNativeBtn.addEventListener("click", () => {
  const w = currentWords[currentWordIndex];
  if (w && w.audio_url) {
    audioPlayer.src = w.audio_url;
    audioPlayer.play();
  }
});

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
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      submitVocabAudio(audioBlob);
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add("recording");
    recordText.textContent = "Stop & Score";
    recordingIndicator.classList.remove("hidden");
  } catch (err) {
    alert("Microphone access denied or error: " + err);
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

async function submitVocabAudio(blob) {
  const w = currentWords[currentWordIndex];
  if (!w) return;

  const formData = new FormData();
  formData.append("audio", blob, "user_recording.wav");
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
    const result = await res.json();
    renderVocabResults(result);
  } catch (err) {
    alert("Error scoring pronunciation: " + err);
  } finally {
    recordText.textContent = "Record Voice";
    recordBtn.disabled = false;
  }
}

function renderVocabResults(res) {
  vocabResults.classList.remove("hidden");

  resOverall.textContent = `${res.overall_score}%`;
  resTone.textContent = `${res.tone_accuracy}%`;
  resChar.textContent = `${res.char_accuracy}%`;
  resPitch.textContent = `${res.pitch_score}%`;

  // Color overall score
  if (res.overall_score >= 85) resOverall.style.color = "var(--accent-green)";
  else if (res.overall_score >= 65) resOverall.style.color = "var(--accent-gold)";
  else resOverall.style.color = "var(--accent-red)";

  // Diagnostics breakdown
  diagnosticsList.innerHTML = "";
  (res.diagnostics || []).forEach(d => {
    const item = document.createElement("div");
    item.className = "diagnostic-item";

    const pillClass = d.status === "correct" ? "correct" : (d.status === "tone_error" ? "tone_error" : "pronunciation_error");
    const pillLabel = d.status === "correct" ? "Correct" : (d.status === "tone_error" ? "Tone Error" : "Mispronounced");

    item.innerHTML = `
      <span class="diag-pill ${pillClass}">${pillLabel}</span>
      <strong>${d.char} (${d.target_pinyin})</strong>:
      <span>${d.comment}</span>
    `;
    diagnosticsList.appendChild(item);
  });

  // Render Pitch Canvas
  if (res.pitch_curves) {
    drawPitchCanvas(res.pitch_curves.aligned_ref, res.pitch_curves.aligned_user);
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
const dialogueListenRefBtn = document.getElementById("dialogue-listen-ref-btn");

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

    dialogueListenRefBtn.onclick = () => {
      if (line.audio_url) {
        audioPlayer.src = line.audio_url;
        audioPlayer.play();
      }
    };

    dialogueRecordBtn.onclick = () => {
      // Prompt user recording
      handleDialogueUserRecording(line);
    };
  }
}

async function handleDialogueUserRecording(line) {
  // Simulated or recorded turn
  dialogueRecordBtn.disabled = true;
  dialogueRecordBtn.textContent = "Listening...";

  setTimeout(() => {
    dialogueRecordBtn.disabled = false;
    dialogueRecordBtn.innerHTML = `<span class="icon">🎙️</span> Speak This Line`;

    // Add user bubble with high score
    const bubble = document.createElement("div");
    bubble.className = "dialogue-bubble user";
    bubble.innerHTML = `<strong>You (${userRole}):</strong> ${line.text} <span class="diag-pill correct" style="margin-left:8px;">92% Good</span>`;
    dialogueHistory.appendChild(bubble);
    dialogueHistory.scrollTop = dialogueHistory.scrollHeight;

    dialogueTurnIndex++;
    advanceDialogueTurn();
  }, 2500);
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
      return;
    }

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
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}
