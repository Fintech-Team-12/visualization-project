FROM python:3.11-slim

# 기본 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Streamlit 기본 포트
EXPOSE 8501

# Streamlit 서버 실행 (docker run -p 8501:8501 과 함께)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]