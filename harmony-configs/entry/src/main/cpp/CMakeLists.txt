cmake_minimum_required(VERSION 3.4.1)
project(ZemoMindSporeNative)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NATIVE_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
  ${NATIVE_ROOT_PATH}
  ${NATIVE_ROOT_PATH}/include
)

add_library(zemo_mslite_native SHARED zemo_mslite_native.cpp)

target_link_libraries(zemo_mslite_native PUBLIC
  libace_napi.z.so
  hilog_ndk.z
  mindspore_lite_ndk
)
