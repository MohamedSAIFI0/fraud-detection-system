FROM python:3.9-slim

WORKDIR /app

# Install Java for PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set Java home
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Create directories for checkpoints and models
RUN mkdir -p /app/checkpoints/cassandra /app/checkpoints/kafka /app/models

# Command to run the application
CMD ["python", "src/stream_processor.py"]