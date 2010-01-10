#######################################################################
# This program is free software; you can redistribute it and/or       #
# modify it under the terms of the GNU General Public License         #
# version 2 as published by the Free Software Foundation.             #
#                                                                     #
# This program is distributed in the hope that it will be useful, but #
# WITHOUT ANY WARRANTY; without even the implied warranty of          #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   #
# General Public License for more details.                            #
#                                                                     #
# Written and (C) by Aurelien Lucchi and Kevin Smith                  #
# Contact aurelien.lucchi (at) gmail.com or kevin.smith (at) epfl.ch  # 
# for comments & bug reports                                          #
#######################################################################

cmake_minimum_required(VERSION 2.4)

PROJECT(rays)
INCLUDE_REGULAR_EXPRESSION("^.*$")

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_C_COMPILER gcc-4.2)
SET(CMAKE_CXX_COMPILER g++-4.2)
SET(EXECUTABLE_OUTPUT_PATH ./bin/
    CACHE PATH "Single output directory for building all executables.")

ADD_LIBRARY(rays MODULE rays.cpp combnk.cpp)

# OpenCv
FIND_PACKAGE(OpenCV REQUIRED)
# add include directories
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(rays ${OPENCV_LIBRARIES})

#IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
#SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
#SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )
#ENDIF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")

set(CMAKE_CXX_FLAGS         "-D WITH_GLEW -D WITH_PNG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -D WITH_GLEW -D WITH_PNG")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -D WITH_GLEW -D WITH_PNG")

set(CMAKE_C_FLAGS         ${CMAKE_CXX_FLAGS})
set(CMAKE_C_FLAGS_DEBUG   ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

SET_TARGET_PROPERTIES(rays PROPERTIES COMPILE_FLAGS -fPIC)
