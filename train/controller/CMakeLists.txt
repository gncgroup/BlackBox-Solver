cmake_minimum_required(VERSION 2.8)
# This project is designed to be built outside the Shark source tree.
# set Shark_DIR to the proper location
project(EVO)
# Find the Shark libraries and includes
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})
# Executable
add_executable(EVO EVO.cpp)
target_link_libraries(EVO ${SHARK_LIBRARIES})



#C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
