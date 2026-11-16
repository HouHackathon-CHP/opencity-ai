# Kiểm tra giấy phép nguồn mở (Open Source License Audit)

Dự án **OpenCity AI Microservice** (`opencity-ai`) được cấp phép theo giấy phép mã nguồn mở **Apache License 2.0** (`LICENSE`). Mọi thành phần, thư viện, mô hình ngôn ngữ và tập dữ liệu đều tuân thủ 100% nguyên tắc phần mềm mã nguồn mở (OSI-approved), không sử dụng bất kỳ dịch vụ tính phí, API đóng hay thư viện độc quyền nào.

---

## 1. Thành phần trực tiếp & Thư viện phụ thuộc

| Thành phần / Thư viện | Phiên bản | Giấy phép | Tổ chức phê duyệt | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Mã nguồn `opencity-ai`** | 2.0.0 | **Apache-2.0** | OSI Approved | Tự phát triển cho dự án OpenCity |
| **PhoBERT-base-v2 (Base Model)** | 2.0 | **MIT / Apache-2.0** | OSI Approved | VinAI Research (Mã nguồn mở phi lợi nhuận) |
| **FastAPI** | 0.115.0+ | **MIT** | OSI Approved | Web framework hiệu năng cao |
| **Uvicorn** | 0.32.0+ | **BSD-3-Clause** | OSI Approved | Máy chủ ASGI mã nguồn mở |
| **Pydantic** | 2.9.2+ | **MIT** | OSI Approved | Xác thực dữ liệu |
| **python-multipart** | 0.0.32+ | **Apache-2.0** | OSI Approved | Xử lý file tải lên đa phần (Multipart audio upload) |
| **PyTorch (torch)** | 2.2.0+ | **BSD-style / Modified BSD** | OSI Approved | Meta Open Source / Linux Foundation |
| **Transformers** | 4.40.0+ | **Apache-2.0** | OSI Approved | Hugging Face Open Source |
| **Safetensors** | 0.4.0+ | **Apache-2.0** | OSI Approved | Hugging Face (Định dạng an toàn không dùng pickle) |
| **SentencePiece** | 0.2.0+ | **Apache-2.0** | OSI Approved | Google Open Source |
| **Protobuf** | 4.25.0+ | **BSD-3-Clause** | OSI Approved | Google Open Source |
| **Whisper (STT Base)** | tiny / base | **MIT** | OSI Approved | OpenAI Open Source Speech Model |
| **PyAV (av)** | 12.0.0+ | **BSD-3-Clause / LGPL-2.1+** | OSI Approved | Python binding for FFmpeg audio/video processing |
| **Python Runtime** | `python:3.12-slim` | **PSF License** | OSI Approved | Python Software Foundation |

---

## 2. Cam kết về tính mở & Độc lập chi phí

1. **Không phụ thuộc API trả phí (Zero Paid API Dependency)**:
   - Hệ thống **hoàn toàn tự chủ (Self-Hosted on-premise)**.
   - Không gọi ra OpenAI GPT, Anthropic Claude hay Google Cloud AI tính phí.
   - Toàn bộ quá trình tính toán suy luận (inference) chạy nội bộ trên CPU hoặc GPU của máy chủ triển khai.
2. **Không có mã nguồn đóng hay bản quyền thương mại hạn chế**:
   - Mọi thư viện phụ thuộc đều thuộc nhóm **Giấy phép Tự do (Permissive Licenses: MIT, Apache-2.0, BSD)**.
   - Cho phép chỉnh sửa, sao chép, phân phối và sử dụng tự do theo chuẩn OSI.
3. **Bộ dữ liệu công cộng & tự tổng hợp**:
   - Bộ dữ liệu phản ánh hiện trường 36.000 mẫu (`civic_dataset_multitask_nationwide`) được sinh tự động và tinh chỉnh dựa trên 63 tỉnh thành Việt Nam, cấp phép theo chuẩn **Apache-2.0**.
