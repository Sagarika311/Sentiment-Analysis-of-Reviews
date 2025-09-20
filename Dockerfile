# Use a small Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy only requirements first (caching)
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Download required NLTK data at build time
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# Expose the port (Render automatically sets PORT env)
EXPOSE 5000

# Run app with Gunicorn, using environment PORT variable if set
CMD sh -c "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app"





