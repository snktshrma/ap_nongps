# ap_nongps
This repository is a work-in-progress and need further modifications. Much of the code variables have been hardcoded to test the algorithm and it'll be solved in upcoming commits. 

### Pull ardupilot_gazebo gsoc-arena branch from the fork
    git clone https://github.com/snktshrma/ardupilot_gazebo_ap.git -b gsoc-arena
    cd ardupilot_gazebo_ap
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j4
### Ubuntu

#### Garden (apt)

Manual - Gazebo Garden Dependencies:

```bash
sudo apt update
sudo apt install libgz-sim7-dev rapidjson-dev
sudo apt install libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
```

#### Harmonic (apt)

Manual - Gazebo Harmonic Dependencies:

```bash
sudo apt update
sudo apt install libgz-sim8-dev rapidjson-dev
sudo apt install libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
```
Ensure the `GZ_VERSION` environment variable is set to either
`garden` or `harmonic`.

Clone the repo and build:

```bash
git clone https://github.com/snktshrma/ardupilot_gazebo_ap.git -b gsoc-arena
cd ardupilot_gazebo_ap
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4
```

## Configure

Set the Gazebo environment variables in your `.bashrc` or `.zshrc` or in 
the terminal used to run Gazebo.

#### Terminal

Assuming that you have cloned the repository to `$HOME/ardupilot_gazebo_ap`:

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo_ap/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo_ap/models:$HOME/ardupilot_gazebo_ap/worlds:$GZ_SIM_RESOURCE_PATH
```

#### .bashrc or .zshrc

Assuming that you have cloned the repository to `$HOME/ardupilot_gazebo_ap`:

```bash
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo_ap/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo_ap/models:$HOME/ardupilot_gazebo_ap/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
```

Reload your terminal with `source ~/.bashrc`.

## Terminal 1:
    cd ardupilot && sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=$HOME/ardupilot_gazebo_ap/config/gazebo-iris-gimbal-ngps.parm --console --map

### Takeoff 10m as it is hardcoded for the time being
    mode GUIDED
    arm throttle force    # Because of Visual Odometry, we have to force for the initial takeoff
    takeoff 10

#### Now set params:
    rc 6 1500
    rc 7 1340
    rc 8 1500
    

## Terminal 2: 
### For Gazebo Harmonic / Garden: 
    gz sim -v4 -r iris_runway.sdf
    
## Terminal 3:
    # Install dependencies
    sudo apt-get install libgirepository1.0-dev libcairo2-dev
    sudo apt-get install gobject-introspection
    pip install -r requirements.txt

    # Enable camera streaming 
    gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p "data: 1"

    # Run the camera based state estimator
    cd src && python video_to_feature.py

    
