# Kontrak Pengiriman Docker

Submission harus dapat dijalankan dari checkout bersih dengan Docker Engine dan Docker Compose saja; grader tidak menginstal dependency Python pada host. Anda melengkapi `Dockerfile`. `docker-compose.yml` disediakan sebagai kontrak PostgreSQL/pgvector, Redis, API, dan Streamlit. Jangan mengubah nama service, port internal, atau `API_URL=http://api:8000`.

Image yang sama dipakai oleh service `api` dan `streamlit`. Karena itu image harus berisi backend, `app.py`, `ui_contract.py`, `demo_personas.json`, data sintetis, dan tests. `docker compose run --rm api python -m src.setup_lab` serta `docker compose run --rm api python -m pytest` harus dapat mengganti perintah default image. Jalankan proses aplikasi sebagai user non-root. Build tidak boleh memerlukan API key atau koneksi ke provider model.

## Prasyarat grader

- Docker Engine dan Compose; internet untuk menarik image/dependency serta untuk seed embedding dan evaluasi live.
- `.env` lokal dibuat dari `.env.example`. Grader mengisi `OPENAI_API_KEY`, `AGENT_MODEL`, `EMBEDDING_MODEL`, dan `EMBEDDING_DIMENSIONS` yang cocok. File ini tidak dikumpulkan atau disalin ke image.
- Gunakan nama project Compose unik per submission. Contoh `p2m2_demo` di bawah harus diganti ketika beberapa submission dinilai pada mesin yang sama.
- Jika port host terpakai, ubah `POSTGRES_PORT`, `REDIS_PORT`, `API_PORT`, atau `STREAMLIT_PORT` di `.env`. Port dan hostname antar-container tidak berubah.

## Urutan dari checkout bersih

Jalankan dari folder `participant/`. Buat `.env` secara lokal (`Copy-Item .env.example .env` pada PowerShell atau `cp .env.example .env` pada macOS/Linux), lalu isi kredensial dari instruktur. Tidak perlu venv atau `pip install` pada host.

```bash
docker compose -p p2m2_demo build api streamlit
docker compose -p p2m2_demo up -d postgres redis
docker compose -p p2m2_demo run --rm api python -m src.setup_lab
docker compose -p p2m2_demo up -d api streamlit
docker compose -p p2m2_demo run --rm api python -m pytest -m "not live"
```

Seed tanpa `--reset` sengaja dipakai agar tidak menghapus data. Project name unik memberi volume PostgreSQL tersendiri untuk tiap submission. Jalankan `docker compose -p p2m2_demo ps` dan `docker compose -p p2m2_demo logs api` untuk diagnosis bila startup gagal.

## Pemeriksaan penerimaan

Dengan port default, buka `http://localhost:8000/healthz`, `http://localhost:8000/readyz`, `http://localhost:8000/openapi.json`, dan `http://localhost:8501`. `/healthz` harus mengembalikan `status: ok`; `/readyz` harus mengembalikan `status: ready` setelah seed dan konfigurasi model benar. Di Streamlit, pilih persona Alya, tanyakan “Berapa jatah cuti tahunan?”, dan pastikan jawaban memuat sitasi kebijakan serta trace. Uji pula saldo pribadi dan konfirmasi yang menghasilkan `pending_manager_review`.

Build image dan pemeriksaan endpoint dasar adalah kewajiban terpisah dari live evaluation. Bila grader belum memberikan kredensial, image tetap harus dapat dibangun, tetapi seed embedding, status `ready`, dan alur model live belum dapat dinilai. Untuk pemeriksaan HTTP dari PowerShell, gunakan `Invoke-RestMethod http://localhost:8000/healthz`; pada macOS/Linux gunakan `curl -fsS http://localhost:8000/healthz`.

Setelah API aktif, live test dan evaluator juga dapat dijalankan tanpa Python host:

```bash
docker compose -p p2m2_demo exec -T -e RUN_LIVE_TESTS=1 -e API_URL=http://localhost:8000 api python -m pytest -m live -o addopts=""
docker compose -p p2m2_demo exec -T api python -m src.run_evaluation --base-url http://localhost:8000 --token token-alya
docker compose -p p2m2_demo cp api:/app/outputs ./outputs
```

Perintah terakhir menyalin report ke folder `outputs/` lokal yang diabaikan Git. Jika Anda mengubah lokasi output evaluator, dokumentasikan perintah ekspor yang setara agar grader dapat mengumpulkan buktinya.

> **Security Note:** `.dockerignore` harus mencegah `.env`, cache, virtual environment, dan laporan lokal masuk ke build context. Jangan memakai build argument untuk mengirim API key. Compose hanya meneruskan kredensial saat container berjalan.
