cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Modules")

project("VIVA - Volumetric Image Visualization and Analysis")

set(CMAKE_C_COMPILER "c++")
#set(CMAKE_BUILD_TYPE debug) #debug none release distribution
set(CMAKE_BUILD_TYPE release) #debug none release distribution

set(CMAKE_CXX_FLAGS         "-D WITH_OPENMP -fopenmp -D WITH_GLEW")
set(CMAKE_CXX_FLAGS_DEBUG   "-D WITH_OPENMP -fopenmp -g -pg -D WITH_GLEW")
set(CMAKE_CXX_FLAGS_RELEASE "-D WITH_OPENMP -fopenmp -O3 -D WITH_GLEW")

set(CMAKE_C_FLAGS         ${CMAKE_CXX_FLAGS})
set(CMAKE_C_FLAGS_DEBUG   ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS "-lgomp ")
set(CMAKE_EXE_LINKER_FLAGS    "-lgomp" )
set(CMAKE_MODULE_LINKER_FLAGS "-lgomp" )

find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${OPENGL_INCLUDE_DIR} )
include_directories(${GLUT_INCLUDE_DIR} )
include_directories(${CMAKE_SOURCE_DIR}/../core)
include_directories(${CMAKE_SOURCE_DIR}/../external/graphCuts)

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../lib/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../bin/)
make_directory(${LIBRARY_OUTPUT_PATH})
make_directory(${EXECUTABLE_OUTPUT_PATH})

#find_package(Boost 1.35.0 COMPONENTS thread filesystem system serialization unit_test_framework REQUIRED)
#set(Boost_USE_MULTITHREAD OFF)
# find_package(BBP-SDK REQUIRED)

add_library (viva SHARED
../core/utils.cpp
../core/Mask.cpp
../core/Axis.cpp
../core/Point.cpp
../core/Point2Do.cpp
../core/Neuron.cpp
../core/ascParser2.cpp
../core/Edge.cpp
../core/Point2Dot.cpp
../core/Cloud.cpp
../core/Point3Dt.cpp
../core/Graph.cpp
../core/EdgeSet.cpp
../core/Point2D.cpp
../core/Point3Dot.cpp
../core/Point3D.cpp
../core/Parser.cpp
../core/Point3Do.cpp
../core/CubeLiveWire.cpp
../core/CubeDijkstra.cpp
../core/Contour.cpp
../core/DoubleSet.cpp
../core/Cube_P.cpp
../core/Cube_T.cpp
../core/CubeFactory.cpp
../core/GraphCut.cpp
)

################################################################################
## Compilation of the viewer
################################################################################

find_package(GTK2 REQUIRED)
INCLUDE_DIRECTORIES( ${GTK2_INCLUDE_DIRS} )
find_package(GtkGLExt REQUIRED)
INCLUDE_DIRECTORIES( ${GTKGLEXT_INCLUDE_DIRS} )
find_package(GLEW REQUIRED)
INCLUDE_DIRECTORIES( ${GLEW_INCLUDE_DIR})

add_executable(vivaView
${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_ascEdit.c
${CMAKE_SOURCE_DIR}/../viewer/src/callbacks.c
${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_draw.c
${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_menu.c
${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_select.c
${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_ui.c
${CMAKE_SOURCE_DIR}/../viewer/src/support.c
${CMAKE_SOURCE_DIR}/../viewer/src/interface.c
${CMAKE_SOURCE_DIR}/../viewer/src/main.c
)

find_package(GSL REQUIRED )
include_directories(${GSL_INCLUDE_DIR})
set(LINK_LIBRARIES_ALL ${BBP-SDK_LIB} ${OPENCV_LIBRARIES}
${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${GTK2_LIBRARY} ${GLEW_LIBRARY}
${GTKGLEXT_LIBS} ${GSL_LIBRARIES} ${GSL_GSLCBLAS_LIBRARY} viva)

target_link_libraries(vivaView ${LINK_LIBRARIES_ALL})

################################################################################
## Compilation of the plugins to the viewer
################################################################################

set(CMAKE_SHARED_MODULE_PREFIX "plugins/")
set(CMAKE_CXX_MODULE_PATH ${CMAKE_SOURCE_DIR}/../lib/plugins)
make_directory(${CMAKE_CXX_MODULE_PATH})

find_package(Gmodule REQUIRED )

add_library (example MODULE ../plugins/src/example/example.cpp)
add_library (CubeDijkstra2 MODULE ../plugins/src/CubeDijkstra2/CubeDijkstra2.cpp)
add_library (CubeLiveWire MODULE ../plugins/src/CubeLiveWire/cubeLiveWire.cpp)
add_library (GraphCuts MODULE ../plugins/src/Graphcuts/GraphCuts.cpp)


################################################################################
## Compilation of the tools
################################################################################

## cube tools

add_executable(cubeAt ${CMAKE_SOURCE_DIR}/../tools/cube/cubeAt.cpp)
target_link_libraries(cubeAt ${LINK_LIBRARIES_ALL})

add_executable(cubeCut ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCut.cpp)
target_link_libraries(cubeCut ${LINK_LIBRARIES_ALL})

add_executable(cubeCalculateAguet ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCalculateAguet.cpp)
target_link_libraries(cubeCalculateAguet ${LINK_LIBRARIES_ALL})

add_executable(cubeDijkstra2 ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDijkstra2.cpp)
target_link_libraries(cubeDijkstra2 ${LINK_LIBRARIES_ALL})

add_executable(cubeDecimate ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDecimate.cpp)
target_link_libraries(cubeDecimate ${LINK_LIBRARIES_ALL})

add_executable(cubeDerivative ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDerivative.cpp)
target_link_libraries(cubeDerivative ${LINK_LIBRARIES_ALL})

add_executable(cubeDerivativesOrder ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDerivativesOrder.cpp)
target_link_libraries(cubeDerivativesOrder ${LINK_LIBRARIES_ALL})

add_executable(cubeStatistics ${CMAKE_SOURCE_DIR}/../tools/cube/cubeStatistics.cpp)
target_link_libraries(cubeStatistics ${LINK_LIBRARIES_ALL})

add_executable(cubeLogistic ${CMAKE_SOURCE_DIR}/../tools/cube/cubeLogistic.cpp)
target_link_libraries(cubeLogistic ${LINK_LIBRARIES_ALL})

add_executable(cubeFromImageStack ${CMAKE_SOURCE_DIR}/../tools/cube/cubeFromImageStack.cpp)
target_link_libraries(cubeFromImageStack ${LINK_LIBRARIES_ALL})


## graph tools
add_executable(graphPrim ${CMAKE_SOURCE_DIR}/../tools/graph/graphPrim.cpp)
target_link_libraries(graphPrim ${LINK_LIBRARIES_ALL})

add_executable(graphFindLeaves ${CMAKE_SOURCE_DIR}/../tools/graph/graphFindLeaves.cpp)
target_link_libraries(graphFindLeaves ${LINK_LIBRARIES_ALL})

add_executable(graphPrune ${CMAKE_SOURCE_DIR}/../tools/graph/graphPrune.cpp)
target_link_libraries(graphPrune ${LINK_LIBRARIES_ALL})

## Test examples
add_executable(tryManyPathsCloud ${CMAKE_SOURCE_DIR}/../tests/tryManyPathsCloud.cpp)
target_link_libraries(tryManyPathsCloud ${LINK_LIBRARIES_ALL})

add_executable(MSTthroughShortestPath ${CMAKE_SOURCE_DIR}/../tests/MSTthroughShortestPath.cpp)
target_link_libraries(MSTthroughShortestPath ${LINK_LIBRARIES_ALL})

add_executable(graphToDisplayList ${CMAKE_SOURCE_DIR}/../tests/graphToDisplayList.cpp)
target_link_libraries(graphToDisplayList ${LINK_LIBRARIES_ALL})

add_executable(gslMinimization ${CMAKE_SOURCE_DIR}/../tests/gslMinimization.cpp)
target_link_libraries(gslMinimization ${LINK_LIBRARIES_ALL})
