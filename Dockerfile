# Use small Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Download all required NLTK data at build time
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# Expose port (Render sets PORT env)
EXPOSE 5000

# Run app with Gunicorn
CMD sh -c "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} --timeout 120 app:app"
