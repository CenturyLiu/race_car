# race_car

This project provides a simple solution for a racecar navigation. The simulation environment is adapted from [eufs_sim](https://github.com/eufsa/eufs_sim)

## Install Prerequisites

- Ubuntu 16.04 + ROS Kinetic

`sudo apt-get install ros-kinetic-ackermann-msgs ros-kinetic-twist-mux ros-kinetic-joy ros-kinetic-controller-manager ros-kinetic-robotnik-msgs ros-kinetic-velodyne-simulator ros-kinetic-effort-controllers ros-kinetic-velocity-controllers ros-kinetic-joint-state-controller ros-kinetic-gazebo-ros-control ros-kinetic-robotnik-msgs`

- Ubuntu 18.04 + ROS Melodic

`sudo apt-get install ros-melodic-ackermann-msgs ros-melodic-twist-mux ros-melodic-joy ros-melodic-controller-manager ros-melodic-robotnik-msgs ros-melodic-velodyne-simulator ros-melodic-effort-controllers ros-melodic-velocity-controllers ros-melodic-joint-state-controller ros-melodic-gazebo-ros-control`

## Install this project

- Download and Compile

  Create a ROS workspace if you don't have one (here, the workspace is called race_car_ws): `mkdir -p ~/race_car_ws/src`.  Copy the contents in this repository into the `src` file you just created.
  
  Navigate to your workspace and build the simulation:
  
      cd ~/race_car_ws/src
      catkin_make
  
  Navigate inside your workspace and give permission to several `.py` or `.sh` files:
  
      cd ~/race_car_ws/src/auto_pilot_ml/scripts
      chmod +x naive_navigation.py
      
      cd ~/race_car_ws/src/eufs_gazebo/nodes
      chmod +x ground_truth_republisher.py
      
      cd ~/race_car/src/sim_simple_controller/nodes
      chmod +x twist_to_ackermannDrive.py
      
      cd ~/race_car_ws/src/scripts
      chmod +x install_python3.sh

- Enable gazebo display

  Put the 3 model packages inside [models](https://github.com/CenturyLiu/race_car/tree/main/models) in this repository into `.gazebo/models`. 
  
- Enable python3 usage

  This project is created for ROS-Kinetic / ROS-Melodic, where python 2.7 is used. However, part of this project's codes are only available for python 3. Several steps for using python 3 along with python 2.7 is listed:
  
  + eliminate any `source` commands in your ~/.bashrc
    `source` commands, for example: `source /opt/ros/melodic/setup.bash` will include python 2.7 directory, and will conflict with python 3. Please delete all `source` commands in "~/.bashrc", and put them in a new file called `~/.bashrc_ros1`. 
  
