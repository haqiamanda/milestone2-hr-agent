# Submission Checklist

- [ ] README dan perintah setup telah diuji dari environment bersih.
- [ ] `Dockerfile` lengkap; image API/UI dibangun tanpa API key saat build dan berjalan sebagai user non-root.
- [ ] `.dockerignore` mencegah `.env`, venv, cache, dan output lokal masuk image.
- [ ] Dari checkout bersih, grader dapat menjalankan urutan `docker compose` pada `docker-delivery.md` tanpa Python/pip di host, memakai project name unik.
- [ ] Seed, deterministic tests dalam container, `/healthz`, `/readyz` (`ready`), OpenAPI, dan UI Streamlit berhasil setelah `.env` instruktur disediakan.
- [ ] Diagram arsitektur menunjukkan trust boundary, cache, telemetry, dan confirmation flow.
- [ ] FastAPI terhubung ke Streamlit yang disediakan.
- [ ] Policy indexing menghasilkan chunk versioned dan vector yang sesuai dimensi.
- [ ] Pertanyaan kebijakan menampilkan sitasi valid.
- [ ] Akses lintas pegawai diblokir.
- [ ] Proposal tidak menulis leave request sebelum konfirmasi.
- [ ] Konfirmasi ulang dengan key sama tidak membuat duplicate.
- [ ] Semua request yang dibuat berstatus `pending_manager_review`.
- [ ] `pytest -m "not live"` lulus.
- [ ] Live evaluation report berisi minimal 15 kasus.
- [ ] Risk register memiliki owner, mitigation, dan evidence.
- [ ] Screenshot UI, monitoring, dan hasil test tersedia.
- [ ] Demo 5–7 menit telah dilatih dengan backup output.

