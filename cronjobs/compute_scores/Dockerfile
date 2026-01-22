# Build stage - install dependencies with gcc
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage - clean image without gcc
FROM python:3.11-slim

WORKDIR /app

# Install curl for database download
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download DB-IP Lite City database (updated monthly)
# IMPORTANT: Do not remove this attribution - required by DB-IP CC BY 4.0 license terms
# Attribution: This product includes IP geolocation data created by DB-IP, available from https://db-ip.com
ARG DBIP_DATE
RUN DBIP_DATE=${DBIP_DATE:-$(date +%Y-%m)} && \
    curl -fsSL "https://download.db-ip.com/free/dbip-city-lite-${DBIP_DATE}.mmdb.gz" | gunzip > /app/dbip-city-lite.mmdb && \
    echo "Downloaded DB-IP database for ${DBIP_DATE}"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY *.py .

# Run as non-root user
RUN useradd -m appuser && chown appuser:appuser /app/dbip-city-lite.mmdb
USER appuser

# Entry point
CMD ["python", "main.py"]
