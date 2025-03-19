cmake_minimum_required(VERSION 3.10)
project(LiDAR_Visualization)

set(CMAKE_CXX_STANDARD 17)

# Find Python3 (for matplotlib)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${Python3_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} matplotlibcpp nanoflann/include)

# Add the executable
add_executable(lidar_plot read_and_plot_lidar.cpp helpers.cpp noise_reduction.cpp object_detection.cpp)

# Link Python3 library
target_link_libraries(lidar_plot ${Python3_LIBRARIES} pthread)
