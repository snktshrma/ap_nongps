# ap_nongps

### Pull ardupilot_gazebo (follow rest of the instructions in [readme here](https://github.com/snktshrma/ardupilot_gazebo_ap/tree/gsoc-arena?tab=readme-ov-file#harmonic-apt)):
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
    cd ardupilot && sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=$HOME/ardupilot_gazebo_ap/config/gazebo-iris-gimbal.parm --console --map

## Terminal 2: 
### For Gazebo Harmonic / Garden: 
    gz sim -v4 -r iris_runway.sdf
    
## Terminal 3:
    gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p "data: 1"

    cd src && python video_to_feature.py

    
