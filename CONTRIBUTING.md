# Hướng dẫn Đóng góp Dự án (OpenCity AI)

Xin chân thành cảm ơn sự quan tâm và góp sức của bạn cho **OpenCity AI**!
Chúng tôi luôn chào đón mọi hình thức đóng góp, bao gồm báo cáo lỗi, gợi ý cải tiến, tối ưu mô hình NLP/ASR và gửi Pull Request.

## Thông báo Lỗi / Đề xuất Cải tiến

- Xác minh rằng vấn đề của bạn chưa được ai khác báo cáo trước đó.
- Sử dụng tiêu đề cụ thể và dễ hiểu.
- Cung cấp thông tin đầy đủ: môi trường sử dụng (Python, CUDA/CPU, phiên bản PyTorch), các bước để tái tạo lỗi, kết quả mong muốn và kết quả thực tế.

## Quy trình Pull Request

1. Tạo bản sao (Fork) của repository này.
2. Khởi tạo nhánh làm việc mới từ `dev` hoặc `main` (ví dụ: `fix/sua-loi-abc` hoặc `feature/bo-sung-xyz`).
3. Thực hiện các commit theo chuẩn [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`) để hệ thống `git-cliff` tự động tổng hợp `CHANGELOG.md`.
4. Chạy kiểm thử và kiểm tra giấy phép thư viện trước khi gửi PR:
   ```bash
   python -m pytest -q
   python scripts/check_direct_licenses.py
   ```
5. Đẩy (Push) code lên fork cá nhân và khởi tạo Pull Request kèm mô tả rõ ràng những thay đổi đã thực hiện.

## Tiêu chuẩn Code

- Tuân thủ quy ước Python / FastAPI / Pydantic trong dự án và giữ header `# SPDX-License-Identifier: Apache-2.0` ở các file mã nguồn mới.
- Chỉ sử dụng các thư viện và trọng số mô hình có giấy phép mã nguồn mở tương thích với Apache-2.0 (đã được kiểm tra qua `scripts/check_direct_licenses.py`).
- Bổ sung comment giải thích cho những đoạn xử lý mô hình hoặc logic phức tạp.
- Xây dựng test case trong `tests/` cho các tính năng mới hoặc lỗi đã sửa.

## Nguyên tắc Cộng đồng

- Thể hiện sự tôn trọng đối với mọi thành viên (xem thêm [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).
- Duy trì phong cách giao tiếp văn minh, mang tính xây dựng.
- Luôn ghi nhớ rằng chúng ta đang cùng nhau phát triển cộng đồng OpenCity.

## Liên hệ Hỗ trợ

Khi gặp khó khăn hoặc có thắc mắc, bạn có thể tạo issue mới hoặc liên lạc trực tiếp với team phát triển qua **Email**: trantrongchien05@gmail.com.
