// Chinese Pronunciation Tutor — Live AI Teacher (Talk Mode) Client

let talkWs = null;
let talkSessionId = null;
let talkState = "idle"; // "idle", "listening", "thinking", "speaking"
let talkRecorder = null;
let talkAudioChunks = [];
let audioPlaybackQueue = [];
let isAudioPlaying = false;

// DOM Elements
const talkScenarioSelect = document.getElementById("talk-scenario-select");
const talkStrictnessSelect = document.getElementById("talk-strictness-select");
const talkWsDot = document.getElementById("talk-ws-dot");
const talkWsText = document.getElementById("talk-ws-text");
const talkStartBtn = document.getElementById("talk-start-btn");
const talkEndBtn = document.getElementById("talk-end-btn");
const talkTranscript = document.getElementById("talk-transcript");
const talkMicOrb = document.getElementById("talk-mic-orb");
const talkMicStatus = document.getElementById("talk-mic-status");
const talkTextInput = document.getElementById("talk-text-input");
const talkSendTextBtn = document.getElementById("talk-send-text-btn");

const talkAudioPlayer = new Audio();

document.addEventListener("DOMContentLoaded", () => {
  setupTalkUI();
});

function setupTalkUI() {
  if (talkStartBtn) {
    talkStartBtn.addEventListener("click", () => {
      startTalkSession();
    });
  }

  if (talkEndBtn) {
    talkEndBtn.addEventListener("click", () => {
      endTalkSession();
    });
  }

  if (talkMicOrb) {
    talkMicOrb.addEventListener("click", () => {
      handleMicOrbClick();
    });
  }

  if (talkSendTextBtn && talkTextInput) {
    talkSendTextBtn.addEventListener("click", () => {
      sendTextMessage();
    });

    talkTextInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        sendTextMessage();
      }
    });
  }

  // Audio queue listener
  talkAudioPlayer.addEventListener("ended", () => {
    playNextAudioChunk();
  });
}

function updateWsStatus(connected, text = "") {
  if (!talkWsDot || !talkWsText) return;
  if (connected) {
    talkWsDot.className = "status-dot connected";
    talkWsText.textContent = text || "Verbunden";
  } else {
    talkWsDot.className = "status-dot disconnected";
    talkWsText.textContent = text || "Getrennt";
  }
}

function setOrbState(state, labelText) {
  talkState = state;
  if (talkMicOrb) {
    talkMicOrb.className = `mic-orb state-${state}`;
  }
  if (talkMicStatus) {
    talkMicStatus.textContent = labelText;
  }
}

function connectTalkWebSocket() {
  if (talkWs && talkWs.readyState === WebSocket.OPEN) return;

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/talk`;

  talkWs = new WebSocket(wsUrl);

  talkWs.onopen = () => {
    updateWsStatus(true, "Verbunden");
    const scenario = talkScenarioSelect ? talkScenarioSelect.value : "free_chat";
    const strictness = talkStrictnessSelect ? talkStrictnessSelect.value : "normal";
    talkWs.send(JSON.stringify({
      type: "start_session",
      scenario: scenario,
      strictness: strictness,
      explain_lang: "de"
    }));
  };

  talkWs.onclose = () => {
    updateWsStatus(false, "Getrennt");
    setOrbState("idle", "Bereit zum Sprechen");
  };

  talkWs.onerror = (err) => {
    console.error("Talk WS Error:", err);
    updateWsStatus(false, "Fehler");
  };

  talkWs.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleTalkServerMessage(msg);
    } catch (e) {
      console.error("Failed to parse WS msg:", e);
    }
  };
}

function startTalkSession() {
  connectTalkWebSocket();
  if (talkStartBtn) talkStartBtn.classList.add("hidden");
  if (talkEndBtn) talkEndBtn.classList.remove("hidden");
  setOrbState("idle", "Klicke zum Sprechen (oder Leertaste)");

  // Clean transcript welcome
  talkTranscript.innerHTML = "";
  const infoBubble = document.createElement("div");
  infoBubble.className = "teacher-welcome-card";
  infoBubble.innerHTML = `
    <h3>🎙️ Sitzung gestartet!</h3>
    <p>Die KI-Lehrerin hört zu. Sprich einen chinesischen Satz oder stelle eine Frage.</p>
  `;
  talkTranscript.appendChild(infoBubble);
}

function endTalkSession() {
  if (talkWs) {
    talkWs.close();
    talkWs = null;
  }
  if (talkStartBtn) talkStartBtn.classList.remove("hidden");
  if (talkEndBtn) talkEndBtn.classList.add("hidden");
  setOrbState("idle", "Sitzung beendet");
  updateWsStatus(false, "Beendet");
}

function handleMicOrbClick() {
  if (!talkWs || talkWs.readyState !== WebSocket.OPEN) {
    startTalkSession();
    return;
  }

  if (talkState === "idle") {
    startTalking();
  } else if (talkState === "listening") {
    stopTalking();
  } else if (talkState === "speaking") {
    // Interrupt teacher speech
    interruptTeacher();
  }
}

async function startTalking() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    talkAudioChunks = [];

    let mimeType = "";
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg"
    ];
    for (const t of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) {
        mimeType = t;
        break;
      }
    }

    talkRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    talkRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        talkAudioChunks.push(e.data);
      }
    };

    talkRecorder.start(250); // Slice every 250ms
    setOrbState("listening", "Zuhören... (Klicke zum Beenden)");
  } catch (err) {
    alert("Mikrofon-Zugriff verweigert: " + err);
    setOrbState("idle", "Bereit");
  }
}

function stopTalking() {
  if (talkRecorder && talkState === "listening") {
    setOrbState("thinking", "KI analysiert deine Sprache & Töne...");

    talkRecorder.onstop = async () => {
      const mime = talkRecorder.mimeType || "audio/webm";
      const blob = new Blob(talkAudioChunks, { type: mime });
      const arrayBuffer = await blob.arrayBuffer();

      if (talkWs && talkWs.readyState === WebSocket.OPEN) {
        talkWs.send(arrayBuffer);
        talkWs.send(JSON.stringify({ type: "end_of_speech" }));
      }
    };

    talkRecorder.stop();
    talkRecorder.stream.getTracks().forEach(t => t.stop());
  }
}

function interruptTeacher() {
  talkAudioPlayer.pause();
  audioPlaybackQueue = [];
  isAudioPlaying = false;
  if (talkWs && talkWs.readyState === WebSocket.OPEN) {
    talkWs.send(JSON.stringify({ type: "interrupt" }));
  }
  setOrbState("idle", "Bereit zum Sprechen");
}

function sendTextMessage() {
  const text = talkTextInput.value.trim();
  if (!text) return;
  if (!talkWs || talkWs.readyState !== WebSocket.OPEN) {
    connectTalkWebSocket();
  }

  setOrbState("thinking", "KI antwortet...");
  talkWs.send(JSON.stringify({ type: "text_input", text: text }));
  talkTextInput.value = "";
}

// Handling server WebSocket messages
let currentUserBubble = null;
let currentTeacherBubble = null;

function handleTalkServerMessage(msg) {
  if (msg.type === "session_started") {
    talkSessionId = msg.session_id;
  } else if (msg.type === "caption_user") {
    // Add user bubble
    currentUserBubble = document.createElement("div");
    currentUserBubble.className = "talk-bubble user";
    currentUserBubble.innerHTML = `
      <div class="talk-bubble-header">
        <span class="speaker-title">Du</span>
        <span class="user-score-badge">...</span>
      </div>
      <div class="talk-bubble-text">${msg.text}</div>
      <div class="user-syllables-list"></div>
    `;
    talkTranscript.appendChild(currentUserBubble);
    talkTranscript.scrollTop = talkTranscript.scrollHeight;
  } else if (msg.type === "reply_start") {
    setOrbState("thinking", "Lehrerin formuliert Antwort...");
  } else if (msg.type === "reply_text") {
    // Add teacher bubble
    currentTeacherBubble = document.createElement("div");
    currentTeacherBubble.className = "talk-bubble teacher";

    const pinyinHtml = msg.reply_pinyin ? `<div class="talk-bubble-pinyin hidden">${msg.reply_pinyin}</div>` : "";
    const transHtml = msg.reply_translation ? `<div class="talk-bubble-trans hidden">${msg.reply_translation}</div>` : "";

    currentTeacherBubble.innerHTML = `
      <div class="talk-bubble-header">
        <span class="speaker-title">Chinesisch-Lehrerin (v2ProPlus)</span>
        <div class="bubble-actions">
          <button class="bubble-btn toggle-py-btn">Pinyin</button>
          <button class="bubble-btn toggle-trans-btn">Deutsch</button>
        </div>
      </div>
      <div class="talk-bubble-text">${msg.reply_zh}</div>
      ${pinyinHtml}
      ${transHtml}
    `;

    // Wire toggle buttons
    const pyBtn = currentTeacherBubble.querySelector(".toggle-py-btn");
    const pyDiv = currentTeacherBubble.querySelector(".talk-bubble-pinyin");
    if (pyBtn && pyDiv) {
      pyBtn.onclick = () => pyDiv.classList.toggle("hidden");
    }

    const trBtn = currentTeacherBubble.querySelector(".toggle-trans-btn");
    const trDiv = currentTeacherBubble.querySelector(".talk-bubble-trans");
    if (trBtn && trDiv) {
      trBtn.onclick = () => trDiv.classList.toggle("hidden");
    }

    talkTranscript.appendChild(currentTeacherBubble);
    talkTranscript.scrollTop = talkTranscript.scrollHeight;
  } else if (msg.type === "corrections") {
    if (currentTeacherBubble) {
      const corrCard = document.createElement("div");
      corrCard.className = "turn-corrections-card";

      let corrContent = `<span class="corr-title">💡 Feedback & Tipps:</span>`;

      (msg.grammar_corrections || []).forEach(g => {
        corrContent += `<div><strong>Satzbau:</strong> ${g.you_said} ➔ <strong style="color:var(--accent-green);">${g.better}</strong> (${g.why})</div>`;
      });

      (msg.pronunciation_focus || []).forEach(p => {
        corrContent += `<div><strong>Ton & Aussprache (${p.syllable}):</strong> ${p.note}</div>`;
      });

      corrCard.innerHTML = corrContent;
      currentTeacherBubble.appendChild(corrCard);
      talkTranscript.scrollTop = talkTranscript.scrollHeight;
    }
  } else if (msg.type === "reply_audio") {
    audioPlaybackQueue.push(msg.audio_base64);
    if (!isAudioPlaying) {
      playNextAudioChunk();
    }
  } else if (msg.type === "pron_report") {
    if (currentUserBubble) {
      const scoreBadge = currentUserBubble.querySelector(".user-score-badge");
      if (scoreBadge && Number.isFinite(msg.overall_score)) {
        const score = Math.round(msg.overall_score);
        const pillClass = score >= 80 ? "correct" : (score >= 60 ? "tone_error" : "pronunciation_error");
        scoreBadge.innerHTML = `<span class="diag-pill ${pillClass}">${score}% Score</span>`;
      }

      const sylContainer = currentUserBubble.querySelector(".user-syllables-list");
      if (sylContainer && msg.syllables && msg.syllables.length > 0) {
        sylContainer.innerHTML = `<div style="margin-top:6px; font-size:12px; display:flex; gap:6px; flex-wrap:wrap;">` +
          msg.syllables.map(s => {
            const col = s.issue ? (s.issue === 'tone' ? 'var(--accent-gold)' : 'var(--accent-red)') : 'var(--accent-green)';
            const expTone = s.expected_tone ? `T${s.expected_tone}` : '';
            const heardTone = s.heard_tone && s.heard_tone !== s.expected_tone ? `➔T${s.heard_tone}` : '';
            return `<span style="background:rgba(0,0,0,0.3); padding:2px 6px; border-radius:4px; color:${col};">
              <strong>${s.char}</strong> (${s.expected} ${expTone}${heardTone})
            </span>`;
          }).join("") + `</div>`;
      }
    }
  } else if (msg.type === "error") {
    alert(msg.message);
    setOrbState("idle", "Bereit");
  } else if (msg.type === "turn_complete") {
    // If not speaking, reset to idle
    if (!isAudioPlaying && audioPlaybackQueue.length === 0) {
      setOrbState("idle", "Bereit zum Sprechen");
    }
  }
}

function playNextAudioChunk() {
  if (audioPlaybackQueue.length === 0) {
    isAudioPlaying = false;
    setOrbState("idle", "Bereit zum Sprechen");
    return;
  }

  isAudioPlaying = true;
  setOrbState("speaking", "Lehrerin spricht...");
  const b64 = audioPlaybackQueue.shift();
  talkAudioPlayer.src = `data:audio/wav;base64,${b64}`;
  talkAudioPlayer.play().catch(e => {
    console.warn("Audio play interrupted:", e);
    playNextAudioChunk();
  });
}

// Global Spacebar Support for Talk Tab
document.addEventListener("keydown", (e) => {
  const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
  if (activeTag === "input" || activeTag === "textarea" || activeTag === "select") return;

  const talkTabActive = document.getElementById("tab-talk") && document.getElementById("tab-talk").classList.contains("active");
  if (talkTabActive && e.code === "Space") {
    e.preventDefault();
    handleMicOrbClick();
  }
});
