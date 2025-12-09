# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements-cpu.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 9380

# Run the service
CMD ["python", "opea_text2audio_microservice.py"]
