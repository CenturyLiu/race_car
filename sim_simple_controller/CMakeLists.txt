cmake_minimum_required(VERSION 2.8.3)
project(sim_simple_controller)


find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  tf
  diagnostic_msgs
  diagnostic_updater
)



catkin_package(
 CATKIN_DEPENDS ackermann_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(sim_simple_controller src/robot_control.cpp)

add_dependencies(sim_simple_controller ackermann_msgs_generate_messages_cpp)

target_link_libraries(sim_simple_controller
   ${catkin_LIBRARIES}
)
