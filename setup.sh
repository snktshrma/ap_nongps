#!/bin/bash

echo "Starting setup script for ArduPilot Gazebo..."

# Change directory to home
cd ~

# Clone the specified branch from the GitHub repository
echo "Cloning the ardupilot_gazebo_ap repository..."
git clone https://github.com/snktshrma/ardupilot_gazebo_ap.git -b gsoc-arena
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone the repository."
    exit 1
fi

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install necessary dependencies
echo "Installing required libraries for Gazebo and GStreamer..."
sudo apt install -y libgz-sim8-dev rapidjson-dev
sudo apt install -y libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

# Set GZ_VERSION environment variable
echo "Setting GZ_VERSION environment variable..."
echo 'export GZ_VERSION=harmonic' >> ~/.bashrc

# Move to cloned repository and build
echo "Entering the ardupilot_gazebo_ap directory and building project..."
cd ardupilot_gazebo_ap
mkdir -p build && cd build
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
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo_ap/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo_ap/models:$HOME/ardupilot_gazebo_ap/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ap_nongps/models:$HOME/ap_nongps/worlds:$GZ_SIM_RESOURCE_PATH' >> ~/.bashrc

# Apply changes to current session
echo "Applying environment changes..."
source ~/.bashrc

# Install additional dependencies
echo "Installing additional required libraries..."
sudo apt-get install -y libgirepository1.0-dev libcairo2-dev gobject-introspection

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python dependencies."
    exit 1
fi

echo "Setup complete! You may need to restart your terminal for all changes to take effect."
