# Use official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirements.txt first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Download required NLTK data
RUN python -m nltk.downloader punkt punkt_tab

# Expose port (optional, Railway auto-detects PORT)
EXPOSE 5000

# Run app with Gunicorn (production server)
CMD sh -c "gunicorn -w 4 -b 0.0.0.0:${PORT} app:app"
