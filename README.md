# OpenCity AI Microservice 🧠🇻🇳

Dedicated Vietnamese Civic NLP & Multi-Task AI Engine for the OpenCity Platform.

## 🌟 Kiến trúc mô hình (PhoBERT Multi-Task)

Dịch vụ chạy một kiến trúc đa nhiệm (Multi-Task Learning) trên nền tảng **PhoBERT-base-v2**:
- **Backbone**: `vinai/phobert-base-v2` (768 chiều ẩn, tiền huấn luyện trên 20GB văn bản tiếng Việt).
- **Head 1 (Phân loại đơn vị đô thị)**: 12 danh mục dân sinh -> Ánh xạ 4 đơn vị tiếp nhận chính quyền (`Giao thông`, `Hạ tầng & chiếu sáng`, `Môi trường & cây xanh`, `Tiếp nhận chung`).
- **Head 2 (Đánh giá mức độ khẩn cấp)**: Dự đoán 3 mức ưu tiên giải quyết (`high`, `normal`, `low`) dựa trên tính cấp bách (sập đổ, nguy hiểm tính mạng, ngập lụt, cháy nổ).
- **Head 3 (Phát hiện trùng lặp đa chiều - Spatiotemporal Deduplication)**: L2-Normalized Sentence Embeddings kết hợp công thức Haversine (khoảng cách địa lý GPS) và độ trễ thời gian (ISO 8601).

---

## 🚀 Khởi chạy độc lập (Standalone)

### Cách 1: Chạy bằng Python (Hỗ trợ GPU CUDA / CPU)

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows

# Cài đặt thư viện
pip install -r requirements.txt
pip install torch transformers safetensors sentencepiece protobuf

# Khởi chạy máy chủ API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Cách 2: Chạy bằng Docker

```bash
docker build -t opencity-ai .
docker run -d -p 8000:8000 --name opencity-ai opencity-ai
```

---

## 📡 API Endpoints

### 1. `GET /health`
Kiểm tra trạng thái sẵn sàng của dịch vụ và model.

**Response:**
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
