# Rubrik Milestone 2 — 100 Poin

| Kategori | Bobot | Bukti penuh |
|---|---:|---|
| Problem framing dan arsitektur | 15 | User, workflow, scope, diagram, trust boundary, dan ADR jelas |
| RAG dan evidence quality | 20 | Chunk versioned, hybrid retrieval, reranking, sitasi valid, abstain |
| Agent tools dan action safety | 20 | ≥2 tool berfungsi, identity binding, validation, confirmation, pending-only, idempotency |
| FastAPI, validation, deployment | 15 | Kontrak sesuai, problem detail aman, health/readiness; image non-root dibangun tanpa key dan Compose menjalankan seed, API, UI, serta test dari checkout bersih tanpa Python host |
| Evaluation dan monitoring | 15 | ≥15 kasus, hard gates, report, trace, metrik, alert actionable |
| Governance dan privacy | 10 | Risk register konkret, PII redaction, injection/IDOR/side-effect controls |
| Dokumentasi dan demo | 5 | README dapat diikuti dan demo 5–7 menit mencakup happy/blocked/failure path |

Untuk kategori **FastAPI, validation, deployment (15 poin)**: 7 poin untuk kontrak endpoint, autentikasi, validasi, dan problem detail; 2 poin untuk health/readiness dan diagnosis; 6 poin untuk build image non-root, seed, API, UI, serta test dari checkout bersih via Compose tanpa Python host.

## Aturan Hard Gate

Nilai maksimum 69 jika salah satu kondisi berikut terjadi:

- Pengguna dapat membaca saldo atau request pegawai lain.
- Sistem membuat atau menyatakan leave request berstatus `approved`.
- Side effect terjadi tanpa konfirmasi eksplisit.
- API key atau data pribadi nyata masuk repository/trace.
- Tidak tersedia evaluation evidence yang dapat dijalankan.

## Panduan Partial Credit

- Implementasi benar tetapi tanpa test/evidence: maksimal 60% poin kategori.
- Diagram tanpa trust boundary atau failure flow: maksimal 70% poin arsitektur.
- Vector-only tanpa hybrid search atau rerank: maksimal 60% poin RAG.
- Monitoring berupa log mentah tanpa metric/alert/owner: maksimal 50% poin monitoring.
- Docker build atau startup gagal pada mesin grader: 0/6 poin subkriteria Docker dan end-to-end release acceptance gagal; poin API serta kategori lain tetap dapat dinilai dari bukti yang tersedia.

