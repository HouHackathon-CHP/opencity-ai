# Nhật ký thay đổi

Tất cả các thay đổi đáng chú ý của dự án này sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án này tuân thủ [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Chưa phát hành

### CI/CD
- **ci**: ci: reconstruct OpenSSH BEGIN/END headers when omitted in secret - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [cf5cb54](https://github.com/HouHackathon-CHP/opencity-ai/commit/cf5cb54ed42634074758241638a9459c79357d6f)
- **ci**: ci: sanitize CR in ssh-keygen error diagnostics - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [67efb8b](https://github.com/HouHackathon-CHP/opencity-ai/commit/67efb8ba6d026c45db1475f432c26c29bc22020b)
- **ci**: ci: auto-normalize SSH deploy key format and report shape diagnostics - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [8dca3d8](https://github.com/HouHackathon-CHP/opencity-ai/commit/8dca3d8be5e11626070d5030fbfb8561e8f3f8e7)
- **ci**: ci: verify CD SSH deploy after secret update - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [e2e871f](https://github.com/HouHackathon-CHP/opencity-ai/commit/e2e871f87d7cf2487ad3bd380c806b20170594ad)
- **ci**: ci: sanitize SSH private key newlines and diagnose CD failures - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [061ce59](https://github.com/HouHackathon-CHP/opencity-ai/commit/061ce59b72cc3ed7f234521e3ec99e098d3a6ed8)
- **ci**: ci: accept OPENCITY_AI repository secret as fallback for SERVER_SSH_KEY - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [31eab6b](https://github.com/HouHackathon-CHP/opencity-ai/commit/31eab6b37b8b6533e93a93ad5005923eabb3d7f2)
- **ci**: ci: fail deploy step clearly if SERVER_SSH_KEY secret is missing - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [f2facd2](https://github.com/HouHackathon-CHP/opencity-ai/commit/f2facd2c33a296313a9da94c8c7e53956db2c674)
- **ci**: ci: add SSH continuous deployment stage to production server - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [7de2c61](https://github.com/HouHackathon-CHP/opencity-ai/commit/7de2c6120e552c39500ee87b7928de82112be971)

## [1.0.0] - 2026-11-27

### Đã thêm
- **voice**: feat(voice): switch ASR engine to Wav2Vec2 Vietnamese (pre-trained on 13k hours) with Apache 2.0 license - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [006cadc](https://github.com/HouHackathon-CHP/opencity-ai/commit/006cadcf46aaf6af740c28f780400b3c516f080b)
- **ui**: feat(ui): add built-in browser testing playground for PhoWhisper voice transcription at root endpoint - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [5ebaef5](https://github.com/HouHackathon-CHP/opencity-ai/commit/5ebaef560316961fad0137d6df4bbbb62cd50eca)
- **api**: feat(api): add CORS support and voice-transcribe route alias for frontend direct testing - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [fdc0bb1](https://github.com/HouHackathon-CHP/opencity-ai/commit/fdc0bb1ddedde2a437056a5ac58f6841642d1eaa)
- **models**: feat(models): replace whisper_tiny with vinai/PhoWhisper-base local weights and refactor to load_phowhisper - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [d3a6a7d](https://github.com/HouHackathon-CHP/opencity-ai/commit/d3a6a7d428fb37554e74ce6c7a5a70f0882d9c76)
- **voice**: feat(voice): upgrade STT model to vinai/PhoWhisper-base with BSD-3-Clause license - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [d38f703](https://github.com/HouHackathon-CHP/opencity-ai/commit/d38f703f779ec3791b5477018afd5dac52fc0390)
- **voice**: feat(voice): bundle local offline weights for Whisper-tiny - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [116074c](https://github.com/HouHackathon-CHP/opencity-ai/commit/116074c5a4419adde7f09a2737fe0922c42a1649)
- **voice**: feat(voice): add Whisper-based Speech-to-Text transcription and auto-routing API - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [4420375](https://github.com/HouHackathon-CHP/opencity-ai/commit/442037540595d71b64032fce6aaace58d8f1fef7)
- **ci**: feat(ci): add Apache 2.0 license, license audit, unit tests, and GitHub Actions CI/CD - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [0f01eb3](https://github.com/HouHackathon-CHP/opencity-ai/commit/0f01eb38fb90edcbe3e62e706a828f07b27c07c9)
- **ai**: feat(ai): initialize standalone opencity-ai microservice with PhoBERT Multi-Task - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [c6044c8](https://github.com/HouHackathon-CHP/opencity-ai/commit/c6044c85fa75e9b0f376f40f6845e500dd67d9d3)

### Đã sửa
- **voice**: fix(voice): add resampler flushing, gain normalization, and num_beams decoding for PhoWhisper - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [bf21069](https://github.com/HouHackathon-CHP/opencity-ai/commit/bf210690947cb3ac196b4cf027494c82a3c02012)
- **deps**: fix(deps): add python-multipart for audio upload handling in FastAPI - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [5f699eb](https://github.com/HouHackathon-CHP/opencity-ai/commit/5f699ebafdd37de0fede2cdb19cf4a0b30281b54)
- **ci**: fix(ci): fix python sys.path and PYTHONPATH in GitHub Actions workflow - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [2a19ccd](https://github.com/HouHackathon-CHP/opencity-ai/commit/2a19ccd6593d79c6ce1733901f53d3e7cca4cb93)

### Đã thay đổi
- **ai**: refactor(ai): modularize architecture into core, schemas, services, and api routes - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [de839dc](https://github.com/HouHackathon-CHP/opencity-ai/commit/de839dc4226fbd4d2a8c59d8face83e879e01cd0)

### Tài liệu
- **training**: docs(training): include GPU training pipeline and nationwide synthetic dataset generator - Trương Ngọc Hiểu ([@hieutnago](https://github.com/hieutnago)) in [c3fbac2](https://github.com/HouHackathon-CHP/opencity-ai/commit/c3fbac25976c6227d55bcf4ee2debd3cbf5b2d64)

### Các thay đổi khác
- Merge pull request #1 from HouHackathon-CHP/dev - Nguyễn Hà Phương ([@ourstorycomic](https://github.com/ourstorycomic)) in [fedacff](https://github.com/HouHackathon-CHP/opencity-ai/commit/fedacffff0b07b41446a7d5dee208e54290884e4)
- Exclude local files from AI Docker build context - Trần Trọng Chiến ([@blueMonkey23](https://github.com/blueMonkey23)) in [056aad9](https://github.com/HouHackathon-CHP/opencity-ai/commit/056aad9bf4a5a1432a0de754ae2a12b9fa16ef6e)
- Document core separation and verify backend AI contract - Trần Trọng Chiến ([@blueMonkey23](https://github.com/blueMonkey23)) in [e07802e](https://github.com/HouHackathon-CHP/opencity-ai/commit/e07802e595549038811de0838b610944804132eb)

---
*Changelog được tạo tự động bởi [git-cliff](https://git-cliff.org). Xem hướng dẫn tại [docs/GIT_CLIFF_GUIDE.md](docs/GIT_CLIFF_GUIDE.md)*
