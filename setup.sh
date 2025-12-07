#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "ArduPilot Gazebo Non-GPS Setup Script"
echo "=========================================="

# Check if running with sufficient privileges for apt commands
if [ "$EUID" -eq 0 ]; then 
    echo "Warning: Please run this script as a normal user, not as root."
    echo "The script will prompt for sudo password when needed."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking for required tools..."
if ! command_exists git; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

if ! command_exists cmake; then
    echo "Error: cmake is not installed. Please install cmake first."
    exit 1
fi

if ! command_exists python3; then
    echo "Error: python3 is not installed. Please install python3 first."
    exit 1
fi

# Change directory to home
cd ~

# Clone the specified branch from the GitHub repository
if [ ! -d "ardupilot_gazebo_ap" ]; then
    echo "Cloning the ardupilot_gazebo_ap repository..."
    git clone https://github.com/snktshrma/ardupilot_gazebo_ap.git -b gsoc-arena
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone the repository."
        exit 1
    fi
else
    echo "ardupilot_gazebo_ap directory already exists, skipping clone..."
fi

# Update package lists
echo "Updating package lists..."
sudo apt update

# Add Gazebo Harmonic repository
echo "Adding Gazebo Harmonic repository..."
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package lists again after adding Gazebo repo
echo "Updating package lists with Gazebo repository..."
sudo apt update

# Install Gazebo Harmonic
echo "Installing Gazebo Harmonic..."
sudo apt install -y gz-harmonic

# Install necessary dependencies
echo "Installing required libraries for Gazebo and GStreamer..."
sudo apt install -y libgz-sim8-dev rapidjson-dev
sudo apt install -y libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

# Install Python dependencies system packages
echo "Installing Python development dependencies..."
sudo apt-get install -y libgirepository1.0-dev libcairo2-dev gobject-introspection python3-gi python3-gi-cairo gir1.2-gstreamer-1.0

# Set GZ_VERSION environment variable
echo "Setting GZ_VERSION environment variable..."
if ! grep -q "export GZ_VERSION=harmonic" ~/.bashrc; then
    echo 'export GZ_VERSION=harmonic' >> ~/.bashrc
fi

# Move to cloned repository and build
echo "Entering the ardupilot_gazebo_ap directory and building project..."
cd ardupilot_gazebo_ap
mkdir -p build && cd build

echo "Running CMake configuration..."
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Compiling the project (this may take a few minutes)..."
make -j4
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

# Set environment paths for plugins and resources
echo "Setting environment paths for plugins and resources..."
if ! grep -q "GZ_SIM_SYSTEM_PLUGIN_PATH.*ardupilot_gazebo_ap/build" ~/.bashrc; then
    echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo_ap/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
fi

if ! grep -q "GZ_SIM_RESOURCE_PATH.*ardupilot_gazebo_ap/models" ~/.bashrc; then
    echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo_ap/models:$HOME/ardupilot_gazebo_ap/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
fi

if ! grep -q "GZ_SIM_RESOURCE_PATH.*ap_nongps/models" ~/.bashrc; then
    echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ap_nongps/models:$HOME/ap_nongps/worlds:$GZ_SIM_RESOURCE_PATH' >> ~/.bashrc
fi

# Return to ap_nongps directory
cd ~/ap_nongps

# Recommend virtual environment
echo ""
echo "=========================================="
echo "Python Environment Setup"
echo "=========================================="
echo "It is highly recommended to use a virtual environment to avoid dependency conflicts."
read -p "Would you like to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install "numpy<2"  # Install NumPy 1.x first to avoid conflicts
    pip install -r requirements.txt
    
    # Create symlink for system pygobject if pip install fails
    if ! python -c "import gi" 2>/dev/null; then
        echo "Linking system pygobject to virtual environment..."
        PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        ln -sf /usr/lib/python3/dist-packages/gi venv/lib/python${PYTHON_VERSION}/site-packages/ 2>/dev/null || true
    fi
    
    echo ""
    echo "Virtual environment created! To activate it in the future, run:"
    echo "    source ~/ap_nongps/venv/bin/activate"
else
    echo "Installing Python dependencies globally..."
    pip install "numpy<2"  # Install NumPy 1.x first
    pip install -r requirements.txt || {
        echo "Warning: Some Python packages failed to install."
        echo "Please refer to TROUBLESHOOTING.md for solutions."
    }
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
python3 -c "import cv2, numpy, pymavlink, matplotlib; print('✓ Core dependencies loaded successfully')" 2>/dev/null && \
python3 -c "import numpy, cv2; print(f'✓ NumPy: {numpy.__version__}, OpenCV: {cv2.__version__}')" 2>/dev/null || \
echo "⚠ Some dependencies may not be properly installed. Check TROUBLESHOOTING.md"

# Apply changes to current session
echo "Applying environment changes..."
source ~/.bashrc

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "You may need to restart your terminal or run 'source ~/.bashrc' for all changes to take effect."
echo ""
echo "Next steps:"
echo "1. Follow the README.md for running the simulation"
echo "2. If you encounter issues, check TROUBLESHOOTING.md"
echo ""