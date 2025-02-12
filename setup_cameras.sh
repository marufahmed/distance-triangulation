#!/bin/bash
# setup_cameras.sh

echo "Setting up camera devices..."

# Detach any existing USB devices
wsl.exe --exec usbipd wsl detach --all

# Attach cameras
wsl.exe --exec usbipd wsl attach --busid=1-3
wsl.exe --exec usbipd wsl attach --busid=1-6

# Wait for USB devices to settle
sleep 2

# Remove existing video devices if they're directories
sudo rm -rf /dev/video0 /dev/video1

# Create video device nodes
sudo mknod /dev/video0 c 81 0
sudo mknod /dev/video1 c 81 1

# Set permissions
sudo chmod 666 /dev/video0 /dev/video1

# Verify setup
echo "Checking USB devices:"
lsusb | grep "046d:085c"

echo "Checking video devices:"
ls -l /dev/video*

echo "Setup complete. Starting Docker..."
docker-compose up --build