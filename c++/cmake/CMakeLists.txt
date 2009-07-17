cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Modules")

project("VIVA - Volumetric Image Visualization and Analysis")

set(CMAKE_C_COMPILER "c++")
set(CMAKE_BUILD_TYPE none) #debug none release distribution

set(CMAKE_CXX_FLAGS         "-openmp -D WITH_OPENMP -D WITH_GLEW")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -pg -openmp -D WITH_OPENMP -D WITH_GLEW")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -openmp -D WITH_OPENMP -D WITH_GLEW")

set(CMAKE_C_FLAGS         ${CMAKE_CXX_FLAGS})
set(CMAKE_C_FLAGS_DEBUG   ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS "-lpthread -lgomp")


find_package(Boost 1.35.0 COMPONENTS thread filesystem system serialization unit_test_framework REQUIRED)
set(Boost_USE_MULTITHREAD OFF)

# find_package(BBP-SDK REQUIRED)

find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR} )
include_directories(${GLUT_INCLUDE_DIR} )

find_package(GTK2 REQUIRED)
INCLUDE_DIRECTORIES( ${GTK2_INCLUDE_DIRS} )

find_package(GtkGLExt REQUIRED)
INCLUDE_DIRECTORIES( ${GTKGLEXT_INCLUDE_DIRS} )

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../lib/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../bin/)
make_directory(${LIBRARY_OUTPUT_PATH})
make_directory(${EXECUTABLE_OUTPUT_PATH})

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
## Error while compiling them - Aurelien's help
#../core/GraphCut.cpp
)

find_package(GLEW REQUIRED)
INCLUDE_DIRECTORIES( ${GLEW_INCLUDE_DIR})
#Include the header files in the compilation of the tools
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/../core)
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

#target_link_libraries(viva ${OPENCV_LIBRARIES} ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
target_link_libraries(vivaView ${BBP-SDK_LIB} ${OPENCV_LIBRARIES}
${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${GTK2_LIBRARY} ${GLEW_LIBRARY} ${GTKGLEXT_LIBS} viva)
