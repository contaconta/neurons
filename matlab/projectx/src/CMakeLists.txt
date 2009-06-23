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

PROJECT(IntegralImages)
INCLUDE_REGULAR_EXPRESSION("^.*$")

SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_C_COMPILER mex)
#SET(CMAKE_CXX_COMPILER mex)

FIND_PACKAGE(OpenCV REQUIRED)
# add include directories
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})

#INCLUDE_DIRECTORIES(/usr/local/extern/include/)

ADD_EXECUTABLE(IntegralImages integral.cpp loadImage.cpp main.cpp utils.cpp)
#ADD_EXECUTABLE(IntegralImages integral.cpp loadImage.cpp utils.cpp)
TARGET_LINK_LIBRARIES(IntegralImages ${OPENCV_LIBRARIES})

#IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )
#ENDIF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")

SET_TARGET_PROPERTIES(IntegralImages PROPERTIES COMPILE_FLAGS -fPIC)
