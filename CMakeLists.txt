project(calibration_laptop)
cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )
aux_source_directory(. SRC_LIST)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
set_property(GLOBAL PROPERTY CXX_STANDARD 17)
