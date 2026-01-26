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

# Download MaxMind GeoLite2 databases
# Attribution: This product includes GeoLite2 data created by MaxMind, available from https://www.maxmind.com
ARG MAXMIND_LICENSE_KEY
RUN if [ -n "$MAXMIND_LICENSE_KEY" ]; then \
    curl -fsSL "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz" | \
    tar -xzf - --strip-components=1 -C /app --wildcards "*/GeoLite2-City.mmdb" && \
    echo "Downloaded GeoLite2-City database" && \
    curl -fsSL "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-ASN&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz" | \
    tar -xzf - --strip-components=1 -C /app --wildcards "*/GeoLite2-ASN.mmdb" && \
    echo "Downloaded GeoLite2-ASN database"; \
    else echo "Error: MAXMIND_LICENSE_KEY required" && exit 1; fi

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY *.py .

# Run as non-root user
RUN useradd -m appuser && chown appuser:appuser /app/*.mmdb
USER appuser

# Entry point
CMD ["python", "main.py"]
