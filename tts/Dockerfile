FROM python:3.11-slim

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPER_RELEASE_BASE=https://github.com/rhasspy/piper/releases/download/v1.2.0 \
    PIPER_BIN=/opt/piper/piper

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in \
    "amd64") PIPER_TARBALL="piper_amd64.tar.gz" ;; \
    "arm64") PIPER_TARBALL="piper_arm64.tar.gz" ;; \
    *) echo "Unsupported TARGETARCH: $TARGETARCH" && exit 1 ;; \
  esac \
  && mkdir -p /opt/piper \
  && curl -fsSL "$PIPER_RELEASE_BASE/$PIPER_TARBALL" -o /tmp/piper.tar.gz \
  && tar -xzf /tmp/piper.tar.gz -C /opt/piper --strip-components=1 \
  && rm -f /tmp/piper.tar.gz \
  && chmod +x /opt/piper/piper

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY voices ./voices

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9002"]
