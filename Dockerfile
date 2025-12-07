FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Gazebo version
ENV GZ_VERSION=harmonic

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    git \
    cmake \
    build-essential \
    pkg-config \
    # Gazebo Harmonic dependencies
    wget \
    lsb-release \
    gnupg \
    # GStreamer and video processing
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    # OpenCV dependencies
    libopencv-dev \
    # Python and pip
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    # PyGObject dependencies
    libgirepository1.0-dev \
    libcairo2-dev \
    gobject-introspection \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    # X11 and display
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    # Additional utilities
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo Harmonic
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && \
    apt-get install -y \
    gz-harmonic \
    libgz-sim8-dev \
    rapidjson-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /root

# Clone and build ardupilot_gazebo_ap
RUN git clone https://github.com/snktshrma/ardupilot_gazebo_ap.git -b gsoc-arena && \
    cd ardupilot_gazebo_ap && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
    make -j$(nproc)

# Set Gazebo environment variables
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=/root/ardupilot_gazebo_ap/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
ENV GZ_SIM_RESOURCE_PATH=/root/ardupilot_gazebo_ap/models:/root/ardupilot_gazebo_ap/worlds:${GZ_SIM_RESOURCE_PATH}
ENV GZ_SIM_RESOURCE_PATH=/root/ap_nongps/models:/root/ap_nongps/worlds:${GZ_SIM_RESOURCE_PATH}

# Install ArduPilot Python dependencies
RUN apt-get update && apt-get install -y \
    python3-matplotlib \
    python3-serial \
    python3-wxgtk4.0 \
    python3-lxml \
    python3-opencv \
    libxml2-dev \
    libxslt1-dev \
    python3-pexpect \
    python3-future \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages via pip
RUN pip3 install \
    MAVProxy \
    pexpect \
    future \
    empy==3.3.4

# Clone and setup ArduPilot (using stable Copter-4.3 instead of 4.5)
RUN git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git && \
    cd ardupilot && \
    git checkout Copter-4.3 && \
    git submodule update --init --recursive

# Add ArduPilot tools to PATH
ENV PATH="/root/ardupilot/Tools/autotest:${PATH}"

# Copy project files
COPY . /root/ap_nongps/

# Install Python dependencies
WORKDIR /root/ap_nongps
RUN pip3 install --upgrade pip && \
    pip3 install "numpy>=1.21.0,<1.25.0" && \
    pip3 install opencv-python==4.6.0.66 && \
    pip3 install -r requirements.txt

# Set display for X11 forwarding
ENV DISPLAY=:0

# Default command
CMD ["/bin/bash"]
