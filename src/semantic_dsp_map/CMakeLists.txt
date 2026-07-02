cmake_minimum_required(VERSION 3.8)
project(semantic_dsp_map)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_BUILD_TYPE Release)

add_definitions(-std=c++17 -O3 -ftree-vectorize -march=native)

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(mask_kpts_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED)
find_package(yaml-cpp REQUIRED)

add_definitions(${PCL_DEFINITIONS})

add_executable(mapping src/mapping.cpp)

target_include_directories(mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

ament_target_dependencies(mapping
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  cv_bridge
  image_transport
  message_filters
  mask_kpts_msgs
  ament_index_cpp
  pcl_conversions
)

target_link_libraries(mapping
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(TARGETS mapping
  DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY cfg/
  DESTINATION share/${PROJECT_NAME}/cfg
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

install(PROGRAMS
  script/168robotics_related_scripts/lidar_to_depth.py
  script/168robotics_related_scripts/camera_pose_publisher.py
  script/168robotics_related_scripts/superpoint_frontend.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
