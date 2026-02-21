FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPER_DOWNLOAD_URL=https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz \
    PIPER_BIN=/opt/piper/piper

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/piper \
  && curl -fsSL "$PIPER_DOWNLOAD_URL" -o /tmp/piper.tar.gz \
  && tar -xzf /tmp/piper.tar.gz -C /opt/piper --strip-components=1 \
  && rm -f /tmp/piper.tar.gz \
  && chmod +x /opt/piper/piper

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY voices ./voices

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9002"]
