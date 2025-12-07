#!/bin/bash
# Complete rebuild and test script for fresh setup

echo "========================================="
echo "  AP NonGPS - Complete Setup Test"
echo "========================================="
echo ""
echo "This script will:"
echo "1. Stop any running containers"
echo "2. Rebuild the Docker image from scratch"
echo "3. Start the container"
echo "4. Run all tests"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Step 1: Stop container
echo ""
echo "Step 1/5: Stopping existing container..."
docker-compose down

# Step 2: Rebuild
echo ""
echo "Step 2/5: Rebuilding Docker image (this will take 10-15 minutes)..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi

# Step 3: Enable X11
echo ""
echo "Step 3/5: Enabling X11 forwarding..."
xhost +local:docker

# Step 4: Start container
echo ""
echo "Step 4/5: Starting container..."
docker-compose up -d
sleep 5

# Step 5: Run tests
echo ""
echo "Step 5/5: Running tests..."
./test_container.sh

echo ""
echo "========================================="
echo "  ✅ Setup Complete!"
echo "========================================="
echo ""
echo "To run the simulation, see README.md section:"
echo "'Running with Docker' - step 5"
