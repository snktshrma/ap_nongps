# ap_nongps

## T1:
    cd ardupilot && ../Tools/autotest/sim_vehicle.py -f gazebo-iris

## T2: 
### For Gazebo Classic: 
    gazebo worlds/iris_ardupilot_ap.world
### For Gazebo Harmonic / Garden: 
    gz sim -v4 -r worlds/iris_runway_ap.sdf
    
## T3: 
    cd src && python video_to_feature.py
