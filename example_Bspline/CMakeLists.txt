cmake_minimum_required(VERSION 3.0.2)
project(example_Bspline)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
  scripts/symbolic_trajectory.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/optimiser.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/drone_and_ball_interfaces.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/test_trajectory.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



