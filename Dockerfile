# Use a small Python image with slim-buster
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies needed for NLTK
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Download NLTK data at build time
RUN python -m nltk.downloader punkt stopwords wordnet

# Expose the port (Render automatically sets PORT env)
EXPOSE 5000

# Run app with Gunicorn, using environment PORT variable if set
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:${PORT:-5000}", "app:app"]
