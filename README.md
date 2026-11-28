<div align="center">

# OpenCity AI Microservice

**Dịch vụ xử lý ngôn ngữ tự nhiên tiếng Việt (PhoBERT Multi-Task) và nhận dạng giọng nói (Wav2Vec2 Vietnamese) cho nền tảng OpenCity.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](app/main.py)
[![Changelog](https://img.shields.io/badge/Changelog-git--cliff-blue.svg)](CHANGELOG.md)

</div>

---

## Tính năng chính

### 1. PhoBERT Multi-Task (`vinai/phobert-base-v2`)
- **Phân loại đơn vị đô thị**: Ánh xạ 12 danh mục dân sinh về 4 đơn vị tiếp nhận (`Giao thông`, `Hạ tầng & chiếu sáng`, `Môi trường & cây xanh`, `Tiếp nhận chung`).
- **Đánh giá mức độ khẩn cấp**: Dự đoán 3 mức ưu tiên (`high`, `normal`, `low`) dựa trên mức độ cấp bách của phản ánh.
- **Phát hiện trùng lặp không gian - thời gian**: Kết hợp L2-Normalized Sentence Embeddings với khoảng cách địa lý Haversine và độ trễ thời gian (ISO 8601).

### 2. Nhận dạng giọng nói tiếng Việt (Wav2Vec2 Vietnamese ASR)
- Chuyển đổi ghi âm giọng nói phản ánh của công dân sang văn bản tiếng Việt và tự động gợi ý tiêu đề, danh mục, mức độ khẩn cấp qua endpoint `POST /api/opencity/reports/voice-transcribe`.
- Tích hợp sẵn giao diện thử nghiệm nhanh trên trình duyệt tại endpoint gốc (`GET /`).

---

## Cấu trúc thư mục

```text
opencity-ai/
├── app/
│   ├── api/routes/         # Endpoint /health, /analyze, /api/opencity/reports/voice-transcribe
│   ├── core/               # Cấu hình môi trường, CUDA/CPU, xác thực HMAC nội bộ
│   ├── schemas/            # Pydantic schemas cho Request & Response
│   ├── services/           # Model loader, civic classifier, deduplication, ASR voice service
│   ├── models/             # Trọng số mô hình local
│   └── main.py             # Entrypoint FastAPI
├── docs/                   # Tài liệu audit giấy phép và hướng dẫn git-cliff
├── scripts/                # Kiểm tra giấy phép thư viện (check_direct_licenses.py)
├── tests/                  # Bộ kiểm thử pytest
├── training/               # Pipeline huấn luyện GPU và sinh dữ liệu tổng hợp
├── .github/workflows/      # CI/CD kiểm thử, deploy và tự động cập nhật CHANGELOG
├── Dockerfile              # Đóng gói container độc lập
└── LICENSE                 # Giấy phép Apache License 2.0
```

---

## Khởi chạy độc lập (Standalone)

### Cách 1: Chạy bằng Python (Hỗ trợ GPU CUDA / CPU)

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows

# Cài đặt thư viện
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Khởi chạy máy chủ API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Cách 2: Chạy bằng Docker

```bash
docker build -t opencity-ai .
docker run -d -p 8000:8000 --name opencity-ai opencity-ai
```

---

## API Endpoints chính

### 1. `GET /health`
Kiểm tra trạng thái sẵn sàng của dịch vụ và mô hình đang nạp.

```json
{
  "status": "ok",
  "service": "opencity-ai",
  "ai_model": "phobert-base-v2-multitask",
  "model_version": "phobert-base-v2-multitask-civic"
}
```

### 2. `POST /analyze`
Phân tích văn bản phản ánh, tính toán xác suất phân luồng đơn vị và phát hiện trùng lặp hồ sơ.

**Request Body:**
```json
{
  "text": "Cột điện nghiêng nguy hiểm sắp đổ ở phố Chùa Bộc",
  "labels": ["infrastructure", "environment", "transport", "intake"],
  "examples": [],
  "candidates": [
    {
      "id": "rep_123",
      "text": "Đổ trụ điện tại Chùa Bộc",
      "latitude": 21.0078,
      "longitude": 105.8286,
      "created_at": "2026-11-14T06:00:00Z"
    }
  ],
  "latitude": 21.0080,
  "longitude": 105.8285,
  "created_at": "2026-11-14T06:30:00Z"
}
```

**Response:**
```json
{
  "model_version": "phobert-base-v2-multitask-civic",
  "ranked_labels": [
    { "label": "infrastructure", "score": 0.985 },
    { "label": "transport", "score": 0.012 },
    { "label": "environment", "score": 0.002 },
    { "label": "intake", "score": 0.001 }
  ],
  "signals": ["cột", "điện", "nguy_hiểm"],
  "similar": [
    { "id": "rep_123", "similarity": 0.92 }
  ]
}
```

### 3. `POST /api/opencity/reports/voice-transcribe`
Nhận file âm thanh (`WebM`, `WAV`, `MP3`, `OGG`, `M4A`), trả về văn bản tiếng Việt cùng gợi ý phân loại và mức độ khẩn cấp.

---

## Kiểm thử & Rà soát giấy phép

```bash
python -m pytest -q
python scripts/check_direct_licenses.py
```

---

## Nhóm phát triển

| Thành viên | GitHub |
| :--- | :--- |
| **Trương Ngọc Hiểu** | [@hieutnago](https://github.com/hieutnago) |
| **Trần Trọng Chiến** | [@blueMonkey23](https://github.com/blueMonkey23) |
| **Nguyễn Hà Phương** | [@ourstorycomic](https://github.com/ourstorycomic) |

---

## Tài liệu & Quy chuẩn cộng đồng

- [CHANGELOG.md](CHANGELOG.md) — Nhật ký thay đổi tự động qua `git-cliff` ([hướng dẫn](docs/GIT_CLIFF_GUIDE.md))
- [CONTRIBUTING.md](CONTRIBUTING.md) — Hướng dẫn đóng góp dự án
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Quy tắc ứng xử cộng đồng
- [SECURITY.md](SECURITY.md) — Chính sách bảo mật và báo cáo lỗ hổng
- [LICENSE](LICENSE) — Giấy phép mã nguồn mở Apache License 2.0
