# ap_nongps

## [** Hardware implementation and Transformer & Optimisation based VPS](https://github.com/snktshrma/ngps_flight)

This repository is a work-in-progress and need further modifications. Much of the code variables have been hardcoded to test the algorithm and it'll be solved in upcoming commits. 

## 🚀 Quick Start with Docker (Recommended)

Docker provides the easiest way to get started without worrying about dependency conflicts or system configuration.

### Complete Setup Test (Fresh System)

To test the complete setup from scratch (simulates a fresh system):

```bash
./full_setup_test.sh
```

This script will:
1. Stop any existing containers
2. Rebuild the Docker image from scratch (10-15 minutes)
3. Start the container
4. Run all verification tests

After completion, proceed to "Running with Docker" below.

### Prerequisites

1. **Install Docker and Docker Compose**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Add your user to docker group (logout and login after this)
   sudo usermod -aG docker $USER
   
   # Install Docker Compose
   sudo apt-get install docker-compose-plugin
   ```

2. **Install NVIDIA Container Toolkit** (Required for GPU acceleration with Gazebo)
   ```bash
   # Add NVIDIA package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   
   # Restart Docker
   sudo systemctl restart docker
   ```

3. **Configure X11 Display Forwarding**
   
   Docker containers need access to your X11 display server to show Gazebo GUI.
   
   **Option A: Quick Setup (Less Secure, Good for Testing)**
   ```bash
   # Allow local connections to X server
   xhost +local:docker
   ```
   
   **Option B: Secure Setup (Recommended for Regular Use)**
   ```bash
   # Create a .Xauthority file for Docker
   xauth list
   
   # Allow specific container access
   xhost +local:$(docker inspect --format='{{ .Config.Hostname }}' ap_nongps_container)
   ```
   
   **For Wayland Users:**
   If you're using Wayland instead of X11, you may need to use XWayland:
   ```bash
   # Check if you're using Wayland
   echo $XDG_SESSION_TYPE
   
   # If output is "wayland", ensure XWayland is running
   # Set DISPLAY variable
   export DISPLAY=:0
   ```

### Running with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/snktshrma/ap_nongps.git
   cd ap_nongps
   ```

2. **Build the Docker image**
   ```bash
   docker-compose build
   ```
   This will take 10-15 minutes on the first run as it downloads and compiles all dependencies.

3. **Start the container**
   ```bash
   # Enable X11 forwarding first
   xhost +local:docker
   
   # Start container
   docker-compose up -d
   ```

4. **Verify the setup**
   ```bash
   ./test_container.sh
   ```

5. **Run the simulation** (requires 4 separate terminals)

   **Terminal 1 - Launch Gazebo:**
   ```bash
   docker exec -it ap_nongps_container bash
   ```
   Inside the container:
   ```bash
   cd /root/ap_nongps
   gz sim -v4 -r iris_runway_ngps.sdf
   ```

   **Terminal 2 - Enable Camera Streaming** (wait for Gazebo to fully load first):
   ```bash
   docker exec -it ap_nongps_container bash
   ```
   Inside the container:
   ```bash
   gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p 'data: 1'
   ```

   **Terminal 3 - Start ArduPilot:**
   ```bash
   docker exec -it ap_nongps_container bash
   ```
   Inside the container:
   ```bash
   cd /root/ardupilot
   sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=/root/ardupilot_gazebo_ap/config/gazebo-iris-gimbal-ngps.parm --console --map
   ```

   **Terminal 4 - Run State Estimator:**
   ```bash
   docker exec -it ap_nongps_container bash
   ```
   Inside the container:
   ```bash
   cd /root/ap_nongps/src
   python video_to_feature.py
   ```

   Expected output in Terminal 4:
   ```
   Heartbeat from system (system 1 component 0)
   Offset x, y(in cms): ...
   ```

6. **Stop the container**
   ```bash
   docker-compose down
   ```

### Docker Troubleshooting

**GUI Not Showing:**
```bash
# Verify DISPLAY is set
echo $DISPLAY

# Re-enable X11 forwarding
xhost +local:docker

# Check if container can access display
docker exec -it ap_nongps_container bash -c "echo \$DISPLAY"
```

**GPU Not Working** (Optional - only if you installed NVIDIA Container Toolkit):
```bash
# Verify nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Use GPU-enabled configuration
docker-compose down
docker-compose -f docker-compose.gpu.yml up -d
```

**Rebuild After Changes:**
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

---

## Manual Setup (Alternative to Docker)

> **Note:** Docker setup (above) is the **recommended method** for most users. Manual setup is provided for advanced users or specific use cases.

If you prefer to install dependencies directly on your system, follow these instructions.

### Prerequisites

Before running the setup script, install these basic tools:
```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3-pip python3-venv wget lsb-release
```

### Setup script
You can clone this repo to $HOME and run the ./setup.sh script directly to set it all up at once (give root access if required (sudo)).

    chmod +x setup.sh
    ./setup.sh

## Setup ardupilot_gazebo
The first step assumes you have build the ArduPilotPlugin and got ardupilot_gazebo setup on the system. Follow the instructions provided [here](https://github.com/snktshrma/ardupilot_gazebo_ap/tree/gsoc-arena) if not.

## Configure

Set the Gazebo environment variables in your `.bashrc` or `.zshrc` or in 
the terminal used to run Gazebo.

Assuming that you have cloned the repository to `$HOME/ardupilot_gazebo`:

```bash
export GZ_SIM_RESOURCE_PATH=$HOME/ap_nongps/models:$HOME/ap_nongps/worlds:$GZ_SIM_RESOURCE_PATH
```

#### .bashrc or .zshrc

Assuming that you have cloned the repository to `$HOME/ardupilot_gazebo`:

```bash
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ap_nongps/models:$HOME/ap_nongps/worlds:$GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
```

Reload your terminal with `source ~/.bashrc`.

## Installation and Setup

For Ubuntu:

```bash
sudo apt-get install libgirepository1.0-dev libcairo2-dev
sudo apt-get install gobject-introspection
```

For macOS:

```bash
brew install cairo
brew install gobject-introspection
brew install inih
```

Install python requirements:
```bash
pip install -r requirements.txt
```
**Important:** This project requires NumPy 1.x due to compatibility with OpenCV and matplotlib. NumPy 2.x will cause runtime errors.

**Recommended: Using Virtual Environment**

To avoid dependency conflicts with system packages, we strongly recommend using a virtual environment:
```bash
cd ~/ap_nongps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "numpy<2"  # Install NumPy 1.x first
pip install -r requirements.txt
```

**Troubleshooting:** If you encounter issues during setup, please refer to [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common problems and solutions.

## Terminal 1: 

### For Gazebo Harmonic / Garden: 
    gz sim -v4 -r iris_runway_ngps.sdf
    
#### Start streaming

```bash
gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p "data: 1"
```

The terminal used to launch Gazebo should display the following if the streaming started correctly:

```
[Msg] GstCameraPlugin:: streaming: started
[Dbg] [GstCameraPlugin.cc:407] GstCameraPlugin: creating generic pipeline
[Msg] GstCameraPlugin: GStreamer element set state returned: 2
[Msg] GstCameraPlugin: starting GStreamer main loop
```
    
## Terminal 2:
    cd ardupilot && sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=$HOME/ardupilot_gazebo_ap/config/gazebo-iris-gimbal-ngps.parm --console --map

### Takeoff 10m as it is hardcoded for the time being
    mode GUIDED
    arm throttle force    # because of visual odom the arming checks report the VisOdom is not healthy; we have to force for the initial takeoff
    takeoff 10

#### Now set params:
    # set roll to centre
    Guided> rc 6 1500

    # set pitch directly downwards
    Guided> rc 7 1300

    # set yaw to centre
    Guided> rc 8 1500
    
## Terminal 3:
    # Run the camera based state estimator
    cd src && python video_to_feature.py

If everything's working, then you'll see the following output:

```bash
$ python video_to_feature.py
Heartbeat from system (system 1 component 0)
Offset x, y(in cms):  0.1942269262460972 -0.3884538524921944
Offset x, y(in cms):  0.1942269262460972 -0.3884538524921944
Offset x, y(in cms):  0.1942269262460972 -0.3884538524921944
Offset x, y(in cms):  0.1942269262460972 -0.3884538524921944
```
    
