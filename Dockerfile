FROM python:3.11-slim

# Install Tesseract OCR (the actual program, not just the Python wrapper)
# plus a couple of shared libraries OpenCV needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer = faster rebuilds
# when only your .py files change, not your requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Render provides the port to listen on via the $PORT environment variable —
# using the shell form (sh -c) here so $PORT actually gets substituted.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
