cmake_minimum_required(VERSION 3.0.2)
project(interpolation_method)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/wayPoint_fitting.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
