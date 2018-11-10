cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(scaled_variance CXX)

# Tell cmake where to find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_executable(scaled_variance scaled_variance.cc)
include_directories(include)

# Set the relevant generic compiler flags (optimisation + warnings)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -Wmissing-declarations -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wundef -Wcast-align -Wformat=2 -Werror=switch -std=c++11 -mfpmath=sse")
message(STATUS "CXX_FLAGS = " ${CMAKE_CXX_FLAGS})

find_package(GSL 1.15 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(SMASH)

include_directories(${GSL_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(scaled_variance ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})


if(${SMASH_FOUND})
  include_directories(${SMASH_INCLUDE_DIR})
  target_link_libraries(scaled_variance ${SMASH_LIBRARIES})
endif(${SMASH_FOUND})
