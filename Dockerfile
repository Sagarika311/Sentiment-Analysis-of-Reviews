# Use official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements.txt first (for caching)
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Download required NLTK data
RUN python -m nltk.downloader punkt punkt_tab

# Expose port (optional, Railway sets $PORT automatically)
EXPOSE 5000

# Run app with Gunicorn (production WSGI server)
CMD sh -c "gunicorn -w 4 -b 0.0.0.0:${PORT} app:app"
