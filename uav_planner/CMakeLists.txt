cmake_minimum_required(VERSION 3.0.2)
project(uav_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
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

catkin_install_python(PROGRAMS
  plot_test/plot_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(FILES
  config/parameters.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

