cmake_minimum_required(VERSION 3.12)
project(go2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(go2_cnpy STATIC ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy/cnpy.cpp)
target_include_directories(go2_cnpy PUBLIC ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy)
target_link_libraries(go2_cnpy PUBLIC ZLIB::ZLIB)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
target_link_libraries(${PROJECT_NAME}_lib PUBLIC go2_cnpy)
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)

add_executable(go2_torque_recorder_test tests/torque_recorder_test.cpp)
target_link_libraries(go2_torque_recorder_test PRIVATE ${PROJECT_NAME}_lib)
