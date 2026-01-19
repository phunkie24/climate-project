FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/ ./data/
COPY config/ ./config/

# Expose port for Streamlit
EXPOSE 8501

# Expose port for API (if needed)
EXPOSE 8000

# Default command (can be overridden)
CMD ["streamlit", "run", "scripts/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
