# Hands-On Lab — HR Policy and Leave Assistant

## 1. Skenario

Anda adalah AI Engineer di perusahaan sintetis **Nusantara Digital**. Tim HR telah menyediakan aplikasi Streamlit, data pegawai sintetis, dan dokumen kebijakan. Tugas Anda adalah membangun backend FastAPI yang dapat menjawab pertanyaan dengan sitasi, membaca saldo milik pengguna aktif, menyiapkan proposal cuti, dan membuat request berstatus `pending_manager_review` hanya setelah konfirmasi eksplisit.

> **Security Note:** Jangan memasukkan data pegawai nyata, API key, screenshot berisi secret, atau informasi medis ke repository dan submission.

## 2. Learning Objectives

Setelah milestone, Anda dapat:

1. Membangun ingestion, chunking, embedding, dan hybrid retrieval berbasis pgvector.
2. Mengimplementasikan agent dengan minimal dua tool dan budget eksekusi.
3. Memvalidasi tool argument, citation, authorization, dan side effect.
4. Menyajikan agent melalui FastAPI yang digunakan aplikasi Streamlit.
5. Membuat evaluation suite, telemetry, monitoring rule, dan risk register.
6. Menjelaskan keputusan RAG vs prompting/fine-tuning dan gap menuju production.
7. Mengemas API dan UI dalam image Docker yang berjalan sebagai user non-root dan lulus pemeriksaan dari checkout bersih.

**Level:** Intermediate  
**Estimasi:** 3 sesi mentoring × 180 menit  
**Artefak utama:** FastAPI AI service, evaluation report, risk register, dan demo 5–7 menit.

## 3. Prerequisites

- Python 3.11+, virtual environment, pytest, REST API, dan Pydantic dasar.
- Dasar PostgreSQL, Docker Compose, RAG, embedding, dan function calling.
- Docker Desktop/Engine, internet, API key, serta model yang ditetapkan instruktur.

## 4. Yang Sudah Disediakan

- `app.py`: Streamlit client lengkap; jangan mengubah kontraknya.
- `data/`: pegawai, saldo, kebijakan HR, 15 evaluation cases, monitoring rules.
- `src/contracts.py` dan `src/config.py`: public schemas dan configuration contract.
- Source lain dengan marker `TODO(M2)`.
- Acceptance tests, template evidence, rubric, dan submission checklist.
- PostgreSQL/pgvector dan Redis pada `docker-compose.yml`.
- Compose empat service dan `.dockerignore`; peserta menyelesaikan `Dockerfile` sesuai [kontrak pengiriman](docker-delivery.md).

Streamlit hanya berkomunikasi melalui FastAPI. UI tidak boleh membaca database langsung.

## 5. Target Architecture

```text
Provided Streamlit
  -> FastAPI + bearer auth
  -> bounded agent loop
       -> PostgreSQL full-text + pgvector + reranking
       -> own leave balance
       -> leave proposal
       -> HR ticket
  -> explicit confirmation -> pending_manager_review
  -> redacted telemetry + monitoring
  -> Redis retrieval cache (degradable)
```

Lengkapi `templates/architecture.md`. Jelaskan mengapa RAG + bounded agent dipilih dan mengapa fine-tuning/multi-agent berada di luar scope.

## 6. Setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Isi `.env`, lalu mulai dependency lokal:

```bash
docker compose up -d postgres redis
```

## 7. Mentoring Session 1 — Scope, Data, dan Retrieval

### Step 1 — Preflight dan desain

1. Jalankan `pytest -m "not live"` dan kelompokkan failure berdasarkan TODO.
2. Lengkapi project brief dan architecture ADR.
3. Tandai trust boundary: user input, retrieved text, model output, tool, storage.

> **Checkpoint 1:** Diagram mencakup identity flow, confirmation boundary, telemetry, cache degradation, dan failure path.

### Step 2 — Database dan seed

Implementasikan `src/setup_lab.py`: extension `vector`; tabel employee, balance, request, proposal, ticket, policy chunk, dan trace; hashed demo token; status constraint; serta idempotent seed.

### Step 3 — Policy indexing

- Parse front matter dan buat chunk stabil dengan document ID, section, version, effective date.
- Embed secara batch dengan `EMBEDDING_MODEL`.
- Validasi dimensi sebelum insert ke pgvector.

### Step 4 — Hybrid retrieval

- Ambil kandidat semantic dan full-text.
- Gabungkan dengan reciprocal-rank fusion dan keyword reranking.
- Cache dengan key yang menyertakan versi index.
- Redis failure menjadi cache miss, bukan request failure.

> **Checkpoint 2:** Query “jatah cuti tahunan” menemukan `HR-LEAVE-001` pada top 3 dengan metadata lengkap.

## 8. Mentoring Session 2 — Agent, FastAPI, dan Safety

### Step 5 — Tool wajib

1. `search_hr_policy(query, top_k)`
2. `get_my_leave_balance()`
3. `prepare_leave_request(start_date, end_date, leave_type, reason)`
4. `create_hr_ticket(category, summary)`

Identity selalu berasal dari bearer token. Validasi proposal mencakup tanggal masa lalu, urutan tanggal, hari kerja, saldo, overlap, tipe cuti, expiry, dan status pending.

### Step 6 — Agent loop

- Responses API dengan strict function schemas.
- Maksimal 4 turn, 6 tool call, 700 output tokens, dan timeout 30 detik.
- Capped retry hanya untuk transient provider failure.
- Jangan retry side effect tanpa idempotency.
- Jangan simpan atau tampilkan chain-of-thought.
- Citation harus berasal dari retrieved documents pada run yang sama.
- Missing/invalid citation menghasilkan abstain.

### Step 7 — API contract

| Method | Endpoint | Requirement |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/readyz` | DB, Redis degradation, model config |
| POST | `/api/v1/chat` | Validated agent response |
| POST | `/api/v1/action-proposals/{id}/confirm` | Atomic, expiring, idempotent, pending-only |
| GET | `/api/v1/leave-requests/me` | Caller-owned records only |
| GET | `/api/v1/traces/{id}` | Owner or monitoring admin |
| GET | `/api/v1/monitoring/summary` | Admin only |
| GET | `/api/v1/monitoring/alerts` | Admin only |

Gunakan consistent problem detail; jangan mengembalikan stack trace, prompt, secret, atau data pengguna lain.

> **Checkpoint 3:** Streamlit dapat policy Q&A dan balance lookup. Bima tidak dapat membaca trace Alya.

### Step 8 — Confirmation boundary

`prepare_leave_request` hanya membuat proposal. Endpoint konfirmasi memverifikasi owner dan expiry, mengunci row dalam transaction, menggunakan idempotency key, membuat tepat satu request, dan selalu menyimpan `pending_manager_review`.

> **Checkpoint 4:** Sebelum konfirmasi tidak ada request. Konfirmasi dua kali dengan key sama menghasilkan request ID sama.

## 9. Mentoring Session 3 — Production Readiness

### Step 9 — Telemetry dan privacy

Catat trace ID, outcome, retrieved document IDs, tool name/status/duration, latency, token, optional cost, dan error tersamarkan. Jangan simpan prompt mentah atau chain-of-thought. Samarkan email, telepon, token-like value, dan rincian medis.

### Step 10 — Evaluation dan monitoring

Implementasikan runner 15 kasus. Report memuat pass rate, hard gates, retrieval recall@3, citation validity, tool selection, p50/p95, token, optional cost, dan GO/NO-GO.

Release gate: hard gates 100%, overall pass minimal 80%, retrieval recall@3 minimal 80%, citation validity 100%, dan tidak ada leave request `approved` yang dibuat sistem.

### Step 11 — Deployment dan demo

- Lengkapi Dockerfile non-root yang mendukung service API, Streamlit, seed, dan test. Compose disediakan; pertahankan service dan port internalnya.
- Ikuti [urutan Docker-only](docker-delivery.md): build tanpa key saat build, seed dari container, start empat service, cek `/healthz`, `/readyz`, OpenAPI, dan UI. Gunakan nama project Compose unik.
- Pastikan `.dockerignore` mengecualikan `.env`, cache, venv, dan laporan lokal dari image.
- Jalankan deterministic tests, live test, dan evaluation suite.
- Lengkapi risk register dan demo script.

> **Checkpoint 5:** Checkout bersih berjalan tanpa Python host; demo Docker mencakup happy path, blocked cross-user request, pending confirmation, failure/abstention, monitoring evidence, dan residual risk.

## 10. Expected Outputs

Grounded response harus memiliki sitasi dan trace. Confirmation menghasilkan `request_id`, status `pending_manager_review`, replay flag, dan audit reference. Bentuk lengkap ditetapkan pada `src/contracts.py` dan Streamlit yang disediakan.

## 11. Testing

```bash
pytest -m "not live"
RUN_LIVE_TESTS=1 pytest -m live -o addopts=""
python -m src.run_evaluation --base-url http://localhost:8000 --token token-alya
```

Tambahkan regression test untuk setiap bug. Jangan melonggarkan acceptance assertion.

## 12. Debugging Guide

| Masalah | Kemungkinan penyebab | Resolusi |
|---|---|---|
| Import error | venv tidak aktif | Aktifkan venv dan install requirements |
| pgvector missing | Image PostgreSQL salah | Gunakan Compose yang disediakan |
| Dimension mismatch | Model dan env tidak selaras | Koreksi dimensi lalu rebuild index lokal |
| API degraded | Schema/seed/model belum siap | Periksa `/readyz`, logs, dan `.env` |
| Citation kosong | Retrieval/prompt/validator gagal | Inspeksi retrieved IDs pada trace |
| Duplicate request | Lock/idempotency salah | Periksa transaction dan proposal state |
| Redis mati | Cache unavailable | Sistem harus lanjut sebagai cache miss |
| UI connection refused | `API_URL` salah | Bedakan hostname Compose dan localhost |

## 13. Stretch Goals

- **Easy:** Filter trace berdasarkan outcome.
- **Intermediate:** Holiday calendar untuk business-day calculation.
- **Intermediate:** Bandingkan dua reranker berdasarkan recall dan latency.
- **Advanced:** OIDC deployment tanpa mengubah tool contract.
- **Advanced:** Asynchronous evaluation worker dan retention job.

## 14. Submission

Gunakan `submission-checklist.md` dan `rubric.md`. Kumpulkan source termasuk Dockerfile, diagram, test output dari container, evaluation report, screenshot UI/monitoring, risk register, demo script, dan presentasi. Jangan sertakan `.env`, database dump, Redis data, API key, atau data pribadi nyata.

