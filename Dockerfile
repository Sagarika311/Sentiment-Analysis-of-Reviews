# Use a small Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Download NLTK data at build time
RUN python -m nltk.downloader punkt stopwords wordnet

# Expose port (Render sets PORT env)
EXPOSE 5000

# Run the app with Gunicorn (expands PORT env)
CMD sh -c "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app"


