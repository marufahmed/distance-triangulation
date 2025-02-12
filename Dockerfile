FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libusb-1.0-0 \
    udev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Create video device nodes in container
RUN mknod /dev/video0 c 81 0 && \
    mknod /dev/video1 c 81 1 && \
    chmod 666 /dev/video0 /dev/video1

RUN apt-get update && apt-get install -y v4l-utils

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create directories for calibration data and measurements
RUN mkdir -p /app/data/calibration /app/data/measurements

# Create udev rules for Logitech C922
RUN echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="085c", MODE="0666"' > /etc/udev/rules.d/99-logitech-c922.rules

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "app/main.py"]