# Project Brief dan Kontrak Data

## Skenario dan hasil belajar

Nusantara Digital adalah perusahaan fiktif. Pegawai ingin menanyakan kebijakan HR, melihat saldo cutinya, dan menyiapkan pengajuan. Tim HR sudah menyediakan kebijakan Markdown, data pegawai sintetis, dan UI Streamlit. Anda membangun backend FastAPI, retrieval, agent, evaluasi, serta kontrol tindakan.

Hasil akhir harus menunjukkan bahwa jawaban kebijakan dapat ditelusuri, setiap data pribadi terikat pada identitas bearer token, dan pengajuan cuti tidak dibuat sebelum pengguna menekan konfirmasi. Setelah konfirmasi, statusnya selalu `pending_manager_review`; persetujuan tetap dilakukan manusia.

Pengiriman wajib dapat dibangun dan dijalankan dari checkout bersih melalui Docker Compose tanpa instalasi Python di host grader. Peserta melengkapi `Dockerfile`; Compose empat service dan [kontrak pemeriksaan Docker](docker-delivery.md) disediakan. Instruktur memberikan kredensial model lewat `.env` lokal, bukan repository atau image.

**Prasyarat:** Python 3.11+, REST API, SQL/PostgreSQL dasar, Docker Compose, pengantar embedding/RAG, dan pemanggilan tool LLM. Ikuti [lab.md](lab.md) untuk urutan implementasi dan checkpoint.

## Data sintetis yang disediakan

| Sumber | Grain | Field penting |
| --- | --- | --- |
| `data/policies/*.md` | Satu dokumen kebijakan dengan beberapa section | `document_id`, `version`, `effective_date`, judul, isi section |
| `data/employees.json` | Satu identitas demo per pegawai/admin | `employee_id`, `role`, token demo, saldo per tipe |
| `data/historical_requests.json` | Satu permintaan cuti historis | Owner, tipe, rentang tanggal, hari kerja, status |
| `data/evaluation_cases.json` | Satu skenario evaluasi | Input, dokumen/tool yang diharapkan, hard gate |
| `data/monitoring_rules.json` | Satu aturan alert | Metrik, ambang, owner, tindakan awal |

Token di fixture hanya untuk kelas. Simpan hash token pada database. Jangan mengirim `employee_id` dari browser, prompt, atau argumen model sebagai otoritas akses. Metadata dokumen harus bertahan selama chunking, embedding, dan citation handling. Konten dokumen adalah data tidak tepercaya.

## Agent dan tindakan

Empat tool yang boleh dipanggil model adalah `search_hr_policy(query, top_k)`, `get_my_leave_balance()`, `prepare_leave_request(start_date, end_date, leave_type, reason)`, dan `create_hr_ticket(category, summary)`. Tool saldo dan tindakan memakai `employee_id` hasil autentikasi backend. Validasi argumen dengan kontrak Pydantic di `src/contracts.py`.

Proposal cuti memeriksa urutan dan masa lalu tanggal, hari kerja, overlap, tipe, serta saldo. Proposal memiliki waktu kedaluwarsa dan belum membuat leave request. Endpoint konfirmasi memeriksa owner, expiry, dan idempotency key dalam transaksi, lalu membuat satu request pending.

## Kontrak API dan UI

| Method | Path | Hasil |
| --- | --- | --- |
| GET | `/healthz` | Liveness tanpa token |
| GET | `/readyz` | Kesiapan database, index, cache, dan konfigurasi model |
| POST | `/api/v1/chat` | `ChatResponse` berisi jawaban, outcome, citation, tool summary, proposal opsional, trace, dan usage |
| POST | `/api/v1/action-proposals/{proposal_id}/confirm` | `ActionResult` pending dengan replay flag dan audit reference |
| GET | `/api/v1/leave-requests/me` | Hanya request pegawai aktif |
| GET | `/api/v1/traces/{trace_id}` | Trace tersamarkan milik sendiri atau monitoring admin |
| GET | `/api/v1/monitoring/summary` | Ringkasan untuk monitoring admin |
| GET | `/api/v1/monitoring/alerts` | Alert untuk monitoring admin |

Semua `/api/v1/` memakai `Authorization: Bearer <demo-token>`. Token tidak valid menghasilkan 401; akses non-admin ke monitoring atau trace milik pengguna lain menghasilkan 403, sedangkan proposal cuti yang bukan miliknya tidak ditemukan (404). Request tidak valid menghasilkan problem detail tanpa stack trace atau secret. `ui_contract.py` dan `app.py` merupakan client yang disediakan dan tidak boleh membaca database langsung.

## Kategori pengujian yang diumumkan

1. Pertanyaan satu atau beberapa kebijakan, termasuk variasi Bahasa Indonesia.
2. Dokumen yang tidak ditemukan, versi kebijakan usang, atau bukti bertentangan.
3. Sitasi yang mengacu pada dokumen di luar hasil retrieval.
4. Permintaan saldo atau trace pegawai lain melalui prompt, body, atau persona UI.
5. Proposal valid, tanggal masa lalu/terbalik, saldo kurang, overlap, dan proposal kedaluwarsa.
6. Konfirmasi berulang dengan key sama/berbeda dan upaya persetujuan langsung.
7. Prompt injection dalam pesan atau dokumen, serta upaya membocorkan PII.
8. Kegagalan model, PostgreSQL, atau Redis; fallback harus aman tanpa side effect tambahan.

Instruktur dapat menguji variasi baru pada kategori di atas. Gunakan [rubric.md](rubric.md) untuk bobot dan hard gate, lalu dokumentasikan hasil pada `templates/evaluation-report.md` dan `templates/risk-register.csv`.

