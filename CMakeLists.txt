cmake_minimum_required(VERSION 2.8.12)

project(cosc462termproject)

add_executable(cannon
  ${PROJECT_SOURCE_DIR}/src/cannon.cpp
  ${PROJECT_SOURCE_DIR}/src/matrix.cpp
  ${PROJECT_SOURCE_DIR}/src/timer.cpp
)

add_executable(dns
  ${PROJECT_SOURCE_DIR}/src/dns.cpp
  ${PROJECT_SOURCE_DIR}/src/matrix.cpp
  ${PROJECT_SOURCE_DIR}/src/timer.cpp
)

target_include_directories(cannon PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(cannon PUBLIC
  "-std=c++11"
)

target_include_directories(dns PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(dns PUBLIC
  "-std=c++11"
  "-g"
)
