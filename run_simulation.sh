#!/bin/bash
# Helper script to run the full ap_nongps simulation in Docker

echo "=== AP NonGPS Docker Simulation Helper ==="
echo ""
echo "This script will help you run the simulation step by step."
echo ""

# Check if container is running
if ! docker ps | grep -q ap_nongps_container; then
    echo "❌ Container is not running!"
    echo "Starting container..."
    xhost +local:docker
    docker-compose up -d
    sleep 3
fi

echo "✅ Container is running"
echo ""

echo "=== Instructions ==="
echo ""
echo "You need to run commands in 3 separate terminals:"
echo ""
echo "TERMINAL 1 - Gazebo:"
echo "  docker exec -it ap_nongps_container bash"
echo "  cd /root/ap_nongps"
echo "  gz sim -v4 -r iris_runway_ngps.sdf"
echo ""
echo "TERMINAL 2 - Enable Camera (after Gazebo loads):"
echo "  docker exec -it ap_nongps_container bash"
echo "  gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p 'data: 1'"
echo ""
echo "TERMINAL 3 - ArduPilot:"
echo "  docker exec -it ap_nongps_container bash"
echo "  cd /root/ardupilot"
echo "  sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=/root/ardupilot_gazebo_ap/config/gazebo-iris-gimbal-ngps.parm --console --map"
echo ""
echo "TERMINAL 4 - State Estimator:"
echo "  docker exec -it ap_nongps_container bash"
echo "  cd /root/ap_nongps/src"
echo "  python video_to_feature.py"
echo ""
echo "=== Quick Commands (copy-paste ready) ==="
echo ""
echo "# Terminal 1:"
echo "docker exec -it ap_nongps_container bash -c 'cd /root/ap_nongps && gz sim -v4 -r iris_runway_ngps.sdf'"
echo ""
echo "# Terminal 2 (run after Gazebo loads):"
echo "docker exec -it ap_nongps_container gz topic -t /world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p 'data: 1'"
echo ""
echo "# Terminal 3:"
echo "docker exec -it ap_nongps_container bash -c 'cd /root/ardupilot && sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=/root/ardupilot_gazebo_ap/config/gazebo-iris-gimbal-ngps.parm --console --map'"
echo ""
echo "# Terminal 4:"
echo "docker exec -it ap_nongps_container bash -c 'cd /root/ap_nongps/src && python video_to_feature.py'"
echo ""
