# Milestone P2M2 — Asisten Kebijakan HR dan Cuti

Anda membangun backend FastAPI untuk UI Streamlit yang sudah disediakan. Sistem harus menjawab kebijakan dengan sitasi, membaca saldo cuti milik pengguna terautentikasi, menyiapkan proposal cuti, dan membuat permintaan `pending_manager_review` hanya sesudah konfirmasi. Semua persona dan data adalah sintetis.

**Level:** Intermediate · **Waktu:** tiga sesi mentoring × 180 menit + demo 5–7 menit · **Hasil akhir:** API, test, evaluasi, risk register, dan catatan arsitektur.

Mulai dari [brief.md](brief.md), ikuti [lab.md](lab.md), lalu gunakan [assessment.md](assessment.md), [rubric.md](rubric.md), dan [submission-checklist.md](submission-checklist.md). Instruktur menyimpan kunci dan implementasi referensi di folder `solution/` terpisah.

## Yang Anda bangun

- `src/`: indexing dan hybrid retrieval, repository PostgreSQL, agent empat tool, aturan cuti, FastAPI, telemetry, serta evaluator. Cari marker `TODO(M2)` sebagai titik implementasi.
- `Dockerfile`: image non-root untuk API dan UI. `docker-compose.yml` menyediakan wiring empat service; lihat [kontrak pengiriman Docker](docker-delivery.md).
- Test tambahan, laporan evaluasi 15 kasus, diagram arsitektur, risk register, dan demo.

`app.py`, `ui_contract.py`, `demo_personas.json`, `data/`, dan model Pydantic pada `src/contracts.py` adalah artefak yang disediakan. UI hanya memanggil API. Identitas pegawai ditentukan backend dari bearer token, bukan dari teks pertanyaan atau pilihan model.

## Menjalankan starter

Dari folder `participant/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d postgres redis
python -m pytest -m "not live"
```

Isi `.env` dengan kredensial dan model yang ditetapkan instruktur. Di macOS/Linux, gunakan `source .venv/bin/activate` dan `cp .env.example .env`. Sebagian public tests memang gagal sampai TODO backend selesai. Setelah implementasi, jalankan `python -m src.setup_lab --reset`, API dengan `python -m uvicorn src.api:create_app --factory --reload`, lalu UI dengan `python -m streamlit run app.py` pada terminal kedua.

Alur host-Python di atas untuk pengembangan lokal. **Submission dinilai melalui Docker saja**, tanpa instalasi Python pada host grader; ikuti [urutan build, seed, startup, dan test](docker-delivery.md).

## Batas sistem

- Tidak ada tool atau endpoint yang menyetujui cuti. Konfirmasi membuat satu permintaan `pending_manager_review`.
- Saldo, request, dan trace pegawai lain tidak boleh diakses.
- Jawaban kebijakan harus menyertakan sitasi dari dokumen yang ditemukan pada run yang sama; bila bukti tidak cukup, abstain atau eskalasi.
- Jangan mengunggah `.env`, API key, database, atau data pegawai nyata.

Lihat [brief.md](brief.md) untuk kontrak endpoint dan kategori kasus, dan [lab.md](lab.md) untuk langkah, checkpoint, serta troubleshooting.

