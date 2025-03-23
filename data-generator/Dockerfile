FROM python

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy generator code
COPY generator.py

# Command to run the generator
CMD ["python", "generator.py"]