# Contributing to ap_nongps

Thank you for your interest in contributing to `ap_nongps`! We welcome contributions from everyone.

## Getting Started

### Option 1: Using Docker (Recommended)

Docker is the easiest way to get started with development as it handles all dependencies automatically.

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ap_nongps.git
   cd ap_nongps
   ```

2. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/snktshrma/ap_nongps.git
   ```

3. **Set up Docker** (if not already installed):
   Follow the Docker setup instructions in the [README.md](README.md#-quick-start-with-docker-recommended).

4. **Build and run the container**:
   ```bash
   # Enable X11 forwarding
   xhost +local:docker
   
   # Build the image
   docker-compose build
   
   # Start the container
   docker-compose up -d
   
   # Access the container shell
   docker exec -it ap_nongps_container bash
   ```

### Option 2: Manual Setup

If you prefer to install dependencies directly on your system:

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ap_nongps.git
   cd ap_nongps
   ```

2. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/snktshrma/ap_nongps.git
   ```

3. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

Please refer to the [README.md](README.md) for detailed installation and setup instructions.

## Development Workflow

### With Docker

1. **Create a branch**:
   ```bash
   git checkout -b my-feature-branch
   ```

2. **Make changes**: Edit files in the `src/` directory on your host machine. Changes are automatically reflected in the container via volume mounts.

3. **Test your changes**:
   ```bash
   # Access the container
   docker exec -it ap_nongps_container bash
   
   # Run your scripts
   cd /root/ap_nongps/src
   python video_to_feature.py
   ```

4. **Rebuild if needed**: If you modify dependencies in `requirements.txt` or `Dockerfile`:
   ```bash
   docker-compose down
   docker-compose build
   docker-compose up -d
   ```

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature X"
   ```

6. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Without Docker

1. **Create a branch**:
   ```bash
   git checkout -b my-feature-branch
   ```

2. **Make changes**: Implement your feature or fix.

3. **Test your changes**: Follow the instructions in the `README.md` to run Gazebo and ArduPilot.

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature X"
   ```

5. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Testing

### With Docker

Before submitting your changes, ensure that the simulation runs correctly:

```bash
# Inside the container
# Terminal 1: Start Gazebo
gz sim -v4 -r iris_runway_ngps.sdf

# Terminal 2: Start ArduPilot (in a new terminal on host)
docker exec -it ap_nongps_container bash
cd ardupilot && sim_vehicle.py -D -v ArduCopter -f JSON --add-param-file=$HOME/ardupilot_gazebo_ap/config/gazebo-iris-gimbal-ngps.parm --console --map

# Terminal 3: Run the state estimator (in another new terminal on host)
docker exec -it ap_nongps_container bash
cd /root/ap_nongps/src && python video_to_feature.py
```

### Without Docker

Follow the instructions in the `README.md` to run Gazebo and ArduPilot.

## Pull Requests

1. **Push to your fork**:
   ```bash
   git push origin my-feature-branch
   ```

2. **Create a Pull Request**: Go to the original repository and click "New Pull Request".

3. **Description**: Provide a clear description of your changes and link to any relevant issues.

## Docker Development Tips

- **Live Code Editing**: The `src/` directory is mounted as a volume, so you can edit code on your host and run it immediately in the container without rebuilding.

- **Viewing Logs**: 
  ```bash
  docker-compose logs -f
  ```

- **Stopping the Container**:
  ```bash
  docker-compose down
  ```

- **Cleaning Up**:
  ```bash
  # Remove container and volumes
  docker-compose down -v
  
  # Remove image
  docker rmi ap_nongps:latest
  ```

- **Accessing Multiple Shells**: You can open multiple shells in the same container:
  ```bash
  docker exec -it ap_nongps_container bash
  ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and modular

## Getting Help

If you encounter issues:

1. Check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
2. Review the [ArduPilot Gazebo documentation](https://github.com/snktshrma/ardupilot_gazebo_ap/tree/gsoc-arena)
3. Open an issue on GitHub with detailed information

We appreciate your contributions!
