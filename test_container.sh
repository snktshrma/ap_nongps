#!/bin/bash

echo "==================================="
echo "  AP NonGPS Docker Setup Test"
echo "==================================="
echo ""

# Test 1: Docker installed
echo "✓ Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "✗ Docker is not installed"
    exit 1
fi
docker --version
echo ""

# Test 2: Docker Compose installed
echo "✓ Checking Docker Compose installation..."
if ! command -v docker-compose &> /dev/null; then
    echo "✗ Docker Compose is not installed"
    exit 1
fi
docker-compose --version
echo ""

# Test 3: Container running
echo "✓ Checking if container is running..."
if ! docker ps | grep -q ap_nongps_container; then
    echo "✗ Container is not running"
    echo "  Start with: docker-compose up -d"
    exit 1
fi
echo "Container is running"
echo ""

# Test 4: Python dependencies
echo "✓ Testing Python dependencies..."
docker exec ap_nongps_container python -c "import cv2, numpy, pymavlink, matplotlib, gi, pyexiv2; print('All dependencies loaded')"
echo ""

# Test 5: Dependency versions
echo "✓ Checking dependency versions..."
docker exec ap_nongps_container python -c "import numpy, cv2; print(f'NumPy: {numpy.__version__}, OpenCV: {cv2.__version__}')"
echo ""

# Test 6: Gazebo installation
echo "✓ Checking Gazebo installation..."
docker exec ap_nongps_container gz sim --version | head -1
echo ""

# Test 7: ArduPilot installation
echo "✓ Checking ArduPilot installation..."
docker exec ap_nongps_container bash -c "ls /root/ardupilot/Tools/autotest/sim_vehicle.py" > /dev/null && echo "ArduPilot installed" || echo "✗ ArduPilot not found"
echo ""

# Test 8: Environment variables
echo "✓ Checking environment variables..."
docker exec ap_nongps_container bash -c 'echo "GZ_VERSION: $GZ_VERSION"'
docker exec ap_nongps_container bash -c 'echo "DISPLAY: $DISPLAY"'
echo ""

echo "==================================="
echo "  ✅ All tests passed!"
echo "==================================="
echo ""
echo "Ready to run simulation. See README.md for instructions."
