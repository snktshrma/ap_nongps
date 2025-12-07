# Troubleshooting Guide

This guide covers common issues encountered during setup and runtime of the ap_nongps project.

## Setup Issues

### NumPy Version Conflict

**Symptoms:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
AttributeError: _ARRAY_API not found
```

**Cause:** OpenCV 4.12.x requires NumPy 2.x, but matplotlib and other dependencies may have been compiled against NumPy 1.x.

**Solution:**
```bash
# Downgrade NumPy to 1.x and use compatible OpenCV version
pip install "numpy<2"
pip install opencv-python==4.8.1.78
```

### PyGObject Installation Fails

**Symptoms:**
```
ERROR: Dependency 'girepository-2.0' is required but not found
error: metadata-generation-failed
```

**Cause:** Missing system-level GObject development packages.

**Solution 1 - Install system dependencies:**
```bash
sudo apt-get update
sudo apt-get install libgirepository1.0-dev libcairo2-dev gobject-introspection python3-gi python3-gi-cairo gir1.2-gstreamer-1.0
```

**Solution 2 - Use system package (if pip install still fails):**
```bash
# Install system package
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0

# Create symlink to your virtual environment (adjust path as needed)
ln -s /usr/lib/python3/dist-packages/gi ~/ap_nongps/venv/lib/python3.10/site-packages/
```

### Missing Python Modules

**Symptoms:**
```
ModuleNotFoundError: No module named 'matplotlib'
ModuleNotFoundError: No module named 'gi'
```

**Solution:**
```bash
pip install matplotlib
# For gi module, see PyGObject section above
```

### pyserial Missing (MAVProxy Dependency)

**Symptoms:**
```
mavproxy requires pyserial>=3.0, which is not installed.
```

**Solution:**
```bash
pip install pyserial
```

## Runtime Issues

### VisOdom Not Healthy on Arming

**Symptoms:**
```
AP: PreArm: VisOdom: not healthy
AP: Arm: VisOdom: not healthy
Got COMMAND_ACK: COMPONENT_ARM_DISARM: FAILED
```

**Cause:** This is expected behavior. The visual odometry system reports as unhealthy until the state estimator starts running.

**Solution:** Use force arming in the ArduPilot console:
```bash
mode GUIDED
arm throttle force
takeoff 10
```

The visual odometry should become healthy after you run `video_to_feature.py`.

### Camera Stream Not Starting

**Symptoms:** No video feed or GStreamer errors in Gazebo terminal.

**Cause:** Camera streaming not enabled.

**Solution:** After launching Gazebo, run:
```bash
gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p "data: 1"
```

You should see:
```
[Msg] GstCameraPlugin:: streaming: started
```

### Gazebo Environment Variables Not Set

**Symptoms:**
```
Error: Unable to find model or world files
```

**Solution:** Ensure environment variables are set in your `.bashrc`:
```bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo_ap/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo_ap/models:$HOME/ardupilot_gazebo_ap/worlds:${GZ_SIM_RESOURCE_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/ap_nongps/models:$HOME/ap_nongps/worlds:$GZ_SIM_RESOURCE_PATH
```

Then reload:
```bash
source ~/.bashrc
```

## Virtual Environment Best Practices

### Creating a Virtual Environment

To avoid dependency conflicts with system packages:
```bash
cd ~/ap_nongps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "numpy<2"  # Important: Install NumPy 1.x first
pip install -r requirements.txt
```

### Activating the Virtual Environment

Always activate before running the project:
```bash
source ~/ap_nongps/venv/bin/activate
```

### Deactivating

To exit the virtual environment:
```bash
deactivate
```

## Verification Commands

### Check Dependency Versions
```bash
python -c "import numpy, cv2; print(f'NumPy: {numpy.__version__}, OpenCV: {cv2.__version__}')"
```

Expected output (approximately):
```
NumPy: 1.26.4, OpenCV: 4.8.1
```

### Test All Dependencies
```bash
python -c "import cv2, numpy, pymavlink, matplotlib, gi; print('All dependencies loaded successfully')"
```

## Getting Help

If you encounter issues not covered here:

1. Check the [ArduPilot Gazebo documentation](https://github.com/snktshrma/ardupilot_gazebo_ap/tree/gsoc-arena)
2. Review the [GSoC 2024 discussion thread](https://discuss.ardupilot.org/t/gsoc-2024-wrapping-up-high-altitude-non-gps-navigation/122905)
3. Open an issue on the [GitHub repository](https://github.com/deepak61296/ap_nongps/issues) with:
   - Your OS and Python version
   - Complete error message
   - Steps you've already tried

## System Requirements

- **OS:** Ubuntu 22.04 or later (recommended)
- **Python:** 3.8 or later
- **Gazebo:** Harmonic
- **RAM:** 8GB minimum, 16GB recommended
- **Disk Space:** ~5GB for all dependencies