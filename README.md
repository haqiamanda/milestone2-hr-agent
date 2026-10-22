# Milestone P2M2 — Asisten Kebijakan HR dan Cuti

Anda membangun backend FastAPI untuk UI Streamlit yang sudah disediakan. Sistem harus menjawab kebijakan dengan sitasi, membaca saldo cuti milik pengguna terautentikasi, menyiapkan proposal cuti, dan membuat permintaan `pending_manager_review` hanya sesudah konfirmasi. Semua persona dan data adalah sintetis.

**Level:** Intermediate · **Waktu:** tiga sesi mentoring × 180 menit + demo 5–7 menit · **Hasil akhir:** API, test, evaluasi, risk register, dan catatan arsitektur.

Mulai dari [brief.md](brief.md), ikuti [lab.md](lab.md), lalu gunakan [assessment.md](assessment.md), [rubric.md](rubric.md), dan [submission-checklist.md](submission-checklist.md). Instruktur menyimpan kunci dan implementasi referensi di folder `solution/` terpisah.

## Yang Anda bangun

- `src/`: indexing dan hybrid retrieval, repository PostgreSQL, agent empat tool, aturan cuti, FastAPI, telemetry, serta evaluator. Cari marker `TODO(M2)` sebagai titik implementasi.
- `Dockerfile`: image non-root untuk API dan UI. `docker-compose.yml` menyediakan wiring empat service; lihat [kontrak pengiriman Docker](docker-delivery.md).
- Test tambahan, laporan evaluasi 15 kasus, diagram arsitektur, risk register, dan demo.

`app.py`, `ui_contract.py`, `demo_personas.json`, `data/`, dan model Pydantic pada `src/contracts.py` adalah artefak yang disediakan. UI hanya memanggil API. Identitas pegawai ditentukan backend dari bearer token, bukan dari teks pertanyaan atau pilihan model.

## Dokumen submission terbaru

| Kebutuhan mentor | Lokasi resmi |
|---|---|
| Problem, user, scope, ukuran keberhasilan dan production gaps | [Project brief](templates/project-brief.md) |
| Diagram arsitektur, trust boundaries, RAG, confirmation dan ADR | [Architecture submission](templates/architecture.md) |
| Risiko, impact/likelihood, control, evidence, owner dan status | [Risk register CSV](templates/risk-register.csv) |
| Hasil regresi, QA historis, runtime, hard gates, assessment dan release caveats | [Evaluation report](templates/evaluation-report.md) |
| Demo 5–7 menit, backup dan screenshot yang perlu ditambahkan | [Demo script](templates/demo-script.md) |
| Urutan slide, narasi dan batas klaim | [Final presentation outline](templates/final-presentation-outline.md) |

Regresi isolated terbaru: **447 passed, 18 live/PostgreSQL deselected**. Docker **377 PASS** dan PostgreSQL integration **17 PASS** mendahului latest security fix. [Official live report](docs/submission_evidence/artifacts/hr-qa-b21b139719e8/evaluation_sprint_f9132622da1f447e95d64ee9c0cb1c49.md): **12/15 PASS, 3 UNAVAILABLE, 0 FAIL, NO-GO**, recall@3 **100%**, citation validity **100%**, hard gates **3/6 verified**. [Supplemental QA retest](docs/submission_evidence/artifacts/hr-qa-b21b139719e8/hard-gates-a78aa761d31c44ed959abf5b0da56021.json): **3/3 PASS**, memakai instrumented FastAPI runtime terpisah terhadap disposable QA PostgreSQL. Ini bukan worker API/UI attestation atau unified official GO. Lihat [evidence index](docs/submission_evidence/README.md) untuk exact scope dan deliverables tersisa.

`docs/study_audit/` dan `docs/qa_analysis/` tetap dipertahankan sebagai bahan belajar dan riwayat. Beberapa temuan/status di dalamnya mendahului perbaikan. Gunakan enam dokumen resmi di atas untuk snapshot submission terbaru; jangan menafsirkan preview PDF/workbook sebagai screenshot runtime atau angka QA lama sebagai hasil baru. Risk owner merupakan peran yang diusulkan; likelihood/impact adalah penilaian kualitatif.

## Pengembangan lokal pada lingkungan terisolasi

Dari root repository ini. Instruksi setup hanya untuk target baru/terisolasi yang sudah Anda verifikasi; jangan menjalankan seed pada database Neon atau data aktif. Nama project Compose unik tidak mengisolasi database eksternal jika konfigurasi masih menunjuk target yang sama. Fase finalisasi dokumentasi tidak menjalankan perintah berikut.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d postgres redis
python -m pytest -m "not live and not postgres_integration"
```

Konfigurasi lokal mengikuti petunjuk instruktur; jangan memasukkan credential ke repository atau screenshot. Di macOS/Linux, gunakan `source .venv/bin/activate` dan `cp .env.example .env`. Untuk DB baru yang terisolasi, perintah setup yang didukung adalah `python -m src.setup_lab` tanpa mengandalkan flag `--reset`. API dapat dijalankan dengan `python -m uvicorn src.api:create_app --factory --reload`, dan UI dengan `python -m streamlit run app.py` pada terminal kedua. Setup melakukan upsert data sintetis dan bukan operasi aman untuk database aktif. Jalur submission resmi tetap Docker-only pada [docker-delivery.md](docker-delivery.md); subset test di atas belum membuktikan seluruh integration acceptance.

Alur host-Python di atas untuk pengembangan lokal. **Submission dinilai melalui Docker saja**, tanpa instalasi Python pada host grader; ikuti [urutan build, seed, startup, dan test](docker-delivery.md).

## Batas sistem

- Tidak ada tool atau endpoint yang menyetujui cuti. Konfirmasi membuat satu permintaan `pending_manager_review`.
- Saldo, request, dan trace pegawai lain tidak boleh diakses.
- Jawaban kebijakan harus menyertakan sitasi dari dokumen yang ditemukan pada run yang sama; bila bukti tidak cukup, abstain atau eskalasi.
- Jangan mengunggah `.env`, API key, database, atau data pegawai nyata.

Lihat [brief.md](brief.md) untuk kontrak endpoint dan kategori kasus, dan [lab.md](lab.md) untuk langkah, checkpoint, serta troubleshooting.

Dated final-source evidence (UTC 2026-10-22 15:57): [acceptance record](docs/submission_evidence/artifacts/hr-qa-b21b139719e8/final-source-20261022T155716Z-2f9aeadc/acceptance.json), [complete isolated pytest log](docs/submission_evidence/artifacts/hr-qa-b21b139719e8/final-source-20261022T155716Z-2f9aeadc/isolated-pytest.log), dan [PostgreSQL integration NOT RUN](docs/submission_evidence/artifacts/hr-qa-b21b139719e8/final-source-20261022T155716Z-2f9aeadc/postgres-integration-NOT-RUN.md). Hasil isolated **447 passed, 0 failed, 18 deselected**; QA health/readiness **HTTP 200** pada running API image yang masih lebih lama dari source. Integration final-source **17 NOT RUN** karena existing fixtures reset/seed/migrate dan membuat container lain; bukan bukti final-source integration PASS. Official NO-GO dan supplemental scope tidak berubah.
