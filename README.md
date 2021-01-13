# race_car

This project provides a simple solution for a racecar navigation. The simulation environment is adapted from [eufs_sim](https://github.com/eufsa/eufs_sim)

## Install Prerequisites

- Ubuntu 16.04 + ROS Kinetic

`sudo apt-get install ros-kinetic-ackermann-msgs ros-kinetic-twist-mux ros-kinetic-joy ros-kinetic-controller-manager ros-kinetic-robotnik-msgs ros-kinetic-velodyne-simulator ros-kinetic-effort-controllers ros-kinetic-velocity-controllers ros-kinetic-joint-state-controller ros-kinetic-gazebo-ros-control ros-kinetic-robotnik-msgs`

- Ubuntu 18.04 + ROS Melodic

`sudo apt-get install ros-melodic-ackermann-msgs ros-melodic-twist-mux ros-melodic-joy ros-melodic-controller-manager ros-melodic-robotnik-msgs ros-melodic-velodyne-simulator ros-melodic-effort-controllers ros-melodic-velocity-controllers ros-melodic-joint-state-controller ros-melodic-gazebo-ros-control`

## Install this project

Create a ROS workspace if you don't have one (here, the workspace is called race_car_ws): `mkdir -p ~/race_car_ws/src`.  Copy the contents in this repository into the `src` file you just created.

- 
