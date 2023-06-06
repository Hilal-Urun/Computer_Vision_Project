cmake_minimum_required(VERSION 2.8)

# Set the project name
project(CVfinalProject)

# Find OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

set (includes "include/foodRecognition.h" "include/foodSegmentation.h" "include/leftoverEst.h")

# Create executables
add_executable(Main src/main.cpp src/foodRecognition.cpp src/foodSegmentation.cpp src/leftoverEst.cpp ${includes})

# Link against OpenCV library
target_link_libraries(Main ${OpenCV_LIBS})
