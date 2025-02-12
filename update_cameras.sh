#!/bin/bash
# update_cameras.sh

# Find C922 cameras
cameras=$(lsusb | grep "046d:085c" | awk '{print $2 "/" $4}' | sed 's/://')

# Create docker-compose from template
cat > docker-compose.yml << EOF
version: '3.8'

services:
  stereo-vision:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    privileged: true
    devices:
$(while read -r camera; do
  echo "      - \"/dev/bus/usb/$camera:/dev/bus/usb/$camera\""
done <<< "$cameras")
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - CAMERA_LEFT_INDEX=0
      - CAMERA_RIGHT_INDEX=1
    restart: unless-stopped
EOF