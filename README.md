# ap_nongps
This repository is a work-in-progress and need further modifications. Much of the code variables have been hardcoded to test the algorithm and it'll be solved in upcoming commits. 

## Setup ardupilot_gazebo
The first step assumes you have build the ArduPilotPlugin and got ardupilot_gazebo setup on the system. Follow the instructions provided [here](https://github.com/snktshrma/ardupilot_gazebo) if not.

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

## Terminal 1:
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
### For Gazebo Harmonic / Garden: 
    gz sim -v4 -r iris_runway_ngps.sdf
    
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
    
