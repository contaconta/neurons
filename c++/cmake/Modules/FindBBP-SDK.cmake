#Ecole Polytechnique Federale de Lausanne
#Brain Mind Institute,
#Blue Brain Project
#(c) 2006-2008. All rights reserved.

# _____________________________________________________________________________
#
# BBP-SDK 
# _____________________________________________________________________________
#

cmake_minimum_required(VERSION 2.4)


# PATH ________________________________________________________________________

find_path(BBP-SDK_PATH include/bbp.h
    ${CMAKE_SOURCE_DIR}/../../BBP-SDK/
    "C:/Program Files/BBP-SDK"
    /opt
    /sw/BBP-SDK
    /opt/BBP-SDK
)

if (BBP-SDK_PATH)
    set (BBP-SDK_FOUND TRUE)
endif (BBP-SDK_PATH)

# HEADERS _____________________________________________________________________

if (BBP-SDK_FOUND)
    set (BBP-SDK_INCLUDE_DIRS ${BBP-SDK_PATH}/include)
    mark_as_advanced (BBP-SDK_INCLUDE_DIR)


# DYNAMIC OR STATIC LIBRARY ___________________________________________________

    find_library(BBP-SDK_LIB 
        NAMES BBP-SDK 
        PATHS ${BBP-SDK_PATH}/lib
              ${BBP-SDK_PATH}/lib/Release
    )
    find_library(BBP-SDK_LIB_DEBUG
        NAMES BBP-SDK.dbg
        PATHS ${BBP-SDK_PATH}/lib
              ${BBP-SDK_PATH}/lib/Debug
    )
    mark_as_advanced(BBP-SDK_LIB)  
    mark_as_advanced(BBP-SDK_LIB_DEBUG)    

    find_library(BBP-SDK_CORBA_LIB 
        NAMES BBP-SDK-CORBA
        PATHS ${BBP-SDK_PATH}/lib
              ${BBP-SDK_PATH}/lib/Release
    )
    find_library(BBP-SDK_CORBA_LIB_DEBUG
        NAMES BBP-SDK-CORBA.dbg
        PATHS ${BBP-SDK_PATH}/lib
              ${BBP-SDK_PATH}/lib/Debug
    )
    mark_as_advanced(BBP-SDK_CORBA_LIB)  
    mark_as_advanced(BBP-SDK_CORBA_LIB_DEBUG)

# COMPILE OPTIONS _____________________________________________________________
    # If no build type has been specified we assume the release configuration
    # for BBP-SDK
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set_property(DIRECTORY ${CMAKE_SOURCE_DIR}
                     PROPERTY COMPILE_DEFINITIONS ${BBP-SDK_DEFINITIONS})
    else ("${CMAKE_BUILD_TYPE}" STREQUAL "")
	#    include(${BBP-SDK_PATH}/lib/BBP-SDK.Debug.cmake)
	#    set_property(DIRECTORY ${CMAKE_SOURCE_DIR}
	#                 PROPERTY COMPILE_DEFINITIONS_DEBUG
	#                 ${BBP-SDK_DEFINITIONS})
	    include(${BBP-SDK_PATH}/lib/BBP-SDK.Release.cmake)
	    set_property(DIRECTORY ${CMAKE_SOURCE_DIR}
			 PROPERTY COMPILE_DEFINITIONS_RELEASE
			 ${BBP-SDK_DEFINITIONS})

    endif ("${CMAKE_BUILD_TYPE}" STREQUAL "")

    set(BBP-SDK_INCLUDE_DIRS ${BBP-SDK_INCLUDE_DIRS}
                             ${HDF5_INCLUDE_DIR}
                             ${XML2_INCLUDE_DIR}
                             ${Boost_INCLUDE_DIR})

# FOUND _______________________________________________________________________

   if (NOT BBP-SDK_FIND_QUIETLY)
      message(STATUS "Found BBP-SDK: ${BBP-SDK_PATH}")
   endif (NOT BBP-SDK_FIND_QUIETLY)
else (BBP-SDK_FOUND)
   if (BBP-SDK_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find BBP-SDK")
   endif (BBP-SDK_FIND_REQUIRED)
endif (BBP-SDK_FOUND)
