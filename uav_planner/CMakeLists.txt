cmake_minimum_required(VERSION 3.0.2)
project(uav_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

set(PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR}/../../interpolation_method/scripts:${PYTHONPATH})
set(ENV{PYTHONPATH} ${PYTHONPATH})


catkin_install_python(PROGRAMS
  scripts/ocp_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

