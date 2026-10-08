# Starter: lengkapi image aplikasi untuk FastAPI DAN Streamlit pada Compose.
# Build tidak boleh membutuhkan OPENAI_API_KEY; konfigurasi model diberikan saat runtime.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
# TODO(M2): Install dependency Python dari requirements.txt.
# TODO(M2): Salin src/, data/, app.py, ui_contract.py, demo_personas.json,
#           dan tests/ ke image. Gunakan .dockerignore agar .env tidak ikut tersalin.
# TODO(M2): Buat user non-root, pastikan file aplikasi dapat dibaca, lalu USER user itu.
# TODO(M2): EXPOSE port 8000 untuk API.
# TODO(M2): CMD jalankan uvicorn src.api:create_app --factory pada 0.0.0.0:8000.
#           Compose mengganti CMD untuk service Streamlit; jangan pakai ENTRYPOINT
#           yang memaksa API sehingga perintah seed/test dan UI tidak dapat dijalankan.

