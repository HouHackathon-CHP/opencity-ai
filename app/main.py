# SPDX-License-Identifier: Apache-2.0
"""OpenCity AI Microservice Entrypoint.

Modular multi-task Vietnamese civic NLP architecture:
- 1 Shared PhoBERT-base-v2 backbone
- Task 1: Civic Category & Municipal Unit Routing (12 categories -> 4 departments)
- Task 2: Urgency Level Estimation (high / normal / low)
- Task 3: Spatiotemporal Deduplication (L2 normalized embeddings + Haversine + time decay)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import analyze_router, health_router, voice_router
from app.services.model_loader import load_model
from app.services.voice_transcriber import load_whisper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("opencity.ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for model pre-warming and graceful shutdown."""
    logger.info("Initializing OpenCity AI Microservice...")
    load_model()
    load_whisper()
    yield
    logger.info("Shutting down OpenCity AI Microservice...")


app = FastAPI(
    title="OpenCity AI Microservice",
    description="Dedicated microservice for Vietnamese civic NLP, triage routing, urgency estimation and deduplication.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular routes
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(voice_router)


@app.get("/", response_class=HTMLResponse)
async def testing_portal():
    """Interactive browser testing playground for PhoWhisper Speech-to-Text and PhoBERT."""
    return """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenCity AI - Trải nghiệm PhoWhisper Tiếng Việt</title>
  <style>
    :root {
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --danger: #dc2626;
      --danger-hover: #b91c1c;
      --success: #16a34a;
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 2rem 1rem; display: flex; justify-content: center; }
    .container { max-width: 720px; width: 100%; }
    .header { text-align: center; margin-bottom: 2rem; }
    .header h1 { font-size: 1.875rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
    .header p { color: var(--muted); font-size: 0.975rem; }
    .badge { display: inline-block; background: #e0e7ff; color: #4338ca; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.75rem; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }
    .controls { display: flex; flex-direction: column; align-items: center; gap: 1.25rem; }
    .mic-btn { width: 88px; height: 88px; border-radius: 50%; border: none; background: var(--primary); color: white; font-size: 2rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; box-shadow: 0 10px 15px -3px rgba(37,99,235,0.3); }
    .mic-btn:hover { background: var(--primary-hover); transform: scale(1.05); }
    .mic-btn.recording { background: var(--danger); animation: pulse 1.5s infinite; }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.6); }
      70% { box-shadow: 0 0 0 20px rgba(220, 38, 38, 0); }
      100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    .status-text { font-size: 1rem; font-weight: 600; color: #334155; }
    .timer { font-size: 1.25rem; font-family: monospace; color: var(--muted); }
    .divider { display: flex; align-items: center; text-align: center; margin: 1.5rem 0; color: var(--muted); font-size: 0.85rem; }
    .divider::before, .divider::after { content: ''; flex: 1; border-bottom: 1px solid var(--border); }
    .divider::before { margin-right: 0.5em; }
    .divider::after { margin-left: 0.5em; }
    .file-upload { display: flex; gap: 0.75rem; width: 100%; align-items: center; justify-content: center; }
    .file-input { border: 1px dashed var(--border); padding: 0.75rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; cursor: pointer; flex: 1; }
    .btn-submit { background: #334155; color: white; border: none; padding: 0.75rem 1.25rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    .btn-submit:hover { background: #1e293b; }
    .result-box { display: none; margin-top: 1.5rem; background: #f1f5f9; border-radius: 0.75rem; padding: 1.25rem; border: 1px solid #cbd5e1; }
    .result-title { font-size: 0.85rem; text-transform: uppercase; font-weight: 700; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .transcription-text { font-size: 1.15rem; color: #0f172a; line-height: 1.6; font-weight: 500; margin-bottom: 1rem; background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
    .meta-item { background: white; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; }
    .meta-label { font-size: 0.75rem; color: var(--muted); font-weight: 600; text-transform: uppercase; }
    .meta-val { font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-top: 0.25rem; }
    .nav-links { display: flex; justify-content: center; gap: 1.5rem; margin-top: 1.5rem; }
    .nav-links a { color: var(--primary); text-decoration: none; font-size: 0.9rem; font-weight: 600; }
    .nav-links a:hover { text-decoration: underline; }
    .spinner { display: none; margin: 1rem auto; width: 36px; height: 36px; border: 3px solid #e2e8f0; border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <span class="badge">OpenCity AI Microservice v2.0</span>
      <h1>Kiểm thử Nhận diện Giọng nói Tiếng Việt</h1>
      <p>Tích hợp mô hình <strong>vinai/PhoWhisper-base</strong> & <strong>PhoBERT Multi-Task</strong></p>
    </div>

    <div class="card">
      <div class="controls">
        <button id="micBtn" class="mic-btn" title="Bấm để ghi âm">🎙️</button>
        <div id="statusText" class="status-text">Bấm vào biểu tượng Micro để bắt đầu nói</div>
        <div id="timer" class="timer">00:00</div>
      </div>

      <div class="divider">HOẶC TẢI LÊN FILE ÂM THANH</div>

      <form id="uploadForm" class="file-upload">
        <input type="file" id="audioFile" class="file-input" accept="audio/*">
        <button type="submit" class="btn-submit">Nhận diện</button>
      </form>

      <div id="spinner" class="spinner"></div>

      <div id="resultBox" class="result-box">
        <div class="result-title">Văn bản Tiếng Việt nhận diện được:</div>
        <div id="transcriptionText" class="transcription-text"></div>
        <div class="meta-grid">
          <div class="meta-item">
            <div class="meta-label">Tiêu đề gợi ý</div>
            <div id="suggestedTitle" class="meta-val">---</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Lĩnh vực phân loại</div>
            <div id="suggestedCat" class="meta-val">---</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Độ khẩn cấp</div>
            <div id="urgency" class="meta-val">---</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Thời lượng âm thanh</div>
            <div id="duration" class="meta-val">---</div>
          </div>
        </div>
      </div>
    </div>

    <div class="nav-links">
      <a href="/docs" target="_blank">📖 Swagger API Docs</a>
      <a href="/health" target="_blank">🩺 Trạng thái Microservice (/health)</a>
      <a href="http://localhost:3000" target="_blank">🌐 Mở OpenCity Frontend</a>
    </div>
  </div>

  <script>
    let mediaRecorder = null;
    let audioChunks = [];
    let recordInterval = null;
    let seconds = 0;

    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');
    const timer = document.getElementById('timer');
    const spinner = document.getElementById('spinner');
    const resultBox = document.getElementById('resultBox');
    const transcriptionText = document.getElementById('transcriptionText');
    const suggestedTitle = document.getElementById('suggestedTitle');
    const suggestedCat = document.getElementById('suggestedCat');
    const urgency = document.getElementById('urgency');
    const duration = document.getElementById('duration');

    function updateTimer() {
      seconds++;
      const m = String(Math.floor(seconds / 60)).padStart(2, '0');
      const s = String(seconds % 60).padStart(2, '0');
      timer.textContent = `${m}:${s}`;
    }

    async function sendAudio(blob) {
      spinner.style.display = 'block';
      statusText.textContent = 'PhoWhisper đang giải mã âm thanh...';
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');

      try {
        const resp = await fetch('/transcribe', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Yêu cầu thất bại (' + resp.status + ')');
        const data = await resp.json();

        resultBox.style.display = 'block';
        transcriptionText.textContent = data.text || 'Không có âm thanh rõ ràng';
        suggestedTitle.textContent = data.suggested_title || '---';
        suggestedCat.textContent = data.suggested_category || '---';
        urgency.textContent = data.urgency === 'high' ? '🔴 Khẩn cấp' : (data.urgency === 'low' ? '🟢 Thấp' : '🟡 Bình thường');
        duration.textContent = data.duration_seconds + ' giây';
        statusText.textContent = 'Hoàn tất!';
      } catch (err) {
        statusText.textContent = 'Lỗi: ' + err.message;
      } finally {
        spinner.style.display = 'none';
      }
    }

    micBtn.addEventListener('click', async () => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        micBtn.classList.remove('recording');
        micBtn.textContent = '🎙️';
        clearInterval(recordInterval);
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        seconds = 0;
        timer.textContent = '00:00';
        recordInterval = setInterval(updateTimer, 1000);

        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = () => {
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(audioChunks, { type: 'audio/webm' });
          sendAudio(blob);
        };

        mediaRecorder.start();
        micBtn.classList.add('recording');
        micBtn.textContent = '⏹️';
        statusText.textContent = 'Đang lắng nghe... Nói sự cố phản ánh của bạn!';
      } catch (err) {
        statusText.textContent = 'Không mở được micro: ' + err.message;
      }
    });

    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('audioFile');
      if (!fileInput.files || fileInput.files.length === 0) {
        alert('Vui lòng chọn file âm thanh!');
        return;
      }
      sendAudio(fileInput.files[0]);
    });
  </script>
</body>
</html>
"""

