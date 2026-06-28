cmake_minimum_required(VERSION 3.22.3)
project(MSLiteSmoke)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED MSLITE_ROOT)
  set(MSLITE_ROOT "D:/Env/ms/mindspore-lite-2.9.0-win-x64")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
  message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

add_executable(mslite_smoke main.cc)
add_executable(mslite_op_dump op_dump.cc)

target_include_directories(mslite_smoke PRIVATE "${MSLITE_ROOT}/runtime")
target_include_directories(mslite_op_dump PRIVATE
  "${MSLITE_ROOT}/runtime/include"
  "${MSLITE_ROOT}/runtime/include/third_party"
)
target_link_directories(mslite_smoke PRIVATE
  "${MSLITE_ROOT}/runtime/lib"
  "${MSLITE_ROOT}/runtime/third_party/glog"
)

target_link_libraries(mslite_smoke
  libmindspore-lite.dll.a
  pthread
)

if(WIN32)
  target_link_libraries(mslite_smoke ssp)
else()
  target_link_libraries(mslite_smoke dl)
endif()
