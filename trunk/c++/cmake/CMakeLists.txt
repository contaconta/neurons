cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

site_name(HOST)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Modules")

project("VIVA - Volumetric Image Visualization and Analysis")

################################################################################
## Compiler and build type
################################################################################

set(CMAKE_BUILD_TYPE debug)  #debug none release distribution
#set(CMAKE_BUILD_TYPE release) #debug none release distribution

set(CMAKE_C_COMPILER "g++")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

IF(WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -D _WIN32")
##gray80 LINK_DIRECTORIES(c:/windows/system32 c:/MinGW/lib)
ENDIF(WIN32)

IF(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -D _APPLE")
include_directories(/usr/x11/include/)
include_directories(/sw/include/)
include_directories(/sw2/include/)
##gray80 LINK_DIRECTORIES(c:/windows/system32 c:/MinGW/lib)
ENDIF(APPLE)


## Need to check for openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D WITH_OPENMP")

message("Hostname = " ${HOST} ", using the following CXX compiler: "
  ${CMAKE_CXX_COMPILER})

################################################################################
## External Libraries and Linking Command
################################################################################

## REQUIRED
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})
include_directories(/usr/include/opencv)
message("OpenCV directories = ${OPENCV_INCLUDE_DIR}")
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
message("OPENGL_INCLUDE_DIR=" ${OPENGL_INCLUDE_DIR})
find_package(GLUT REQUIRED)
find_package(TIFF REQUIRED)

#include_directories(${GLUT_INCLUDE_DIR} )

#find_package(BBP-SDK)
#IF(${BBP-SDK_FOUND})
#  find_package(Boost 1.35.0 COMPONENTS thread filesystem system serialization unit_test_framework)
#  IF(${Boost_FOUND})
#        INCLUDE_DIRECTORIES( ${BBP-SDK_INCLUDE_DIRS})
#        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D WITH_BBP")
#        set(Boost_USE_MULTITHREAD OFF)
#  ENDIF(${Boost_FOUND})
#  # message("BBP has been found")
#ELSE(${BBP-SDK_FOUND})
#message("BBP has NOT been found")
#ENDIF(${BBP-SDK_FOUND})

## OPTIONAL
# For the computation of eigenvalues / vectors
find_package(GSL)
IF(${GSL_FOUND})
  include_directories(${GSL_INCLUDE_DIR})
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D WITH_GSL")
ENDIF(${GSL_FOUND})

# For the gtk interface
find_package(GTK2)
IF(${GTK2_FOUND})
  INCLUDE_DIRECTORIES( ${GTK2_INCLUDE_DIRS})
ENDIF(${GTK2_FOUND})
find_package(GtkGLExt)
INCLUDE_DIRECTORIES( ${GTKGLEXT_INCLUDE_DIRS} )

find_package(GLEW)
message("GLEW has been found: ${GLEW_FOUND}")
IF(${GLEW_FOUND})
  include_directories( ${GLEW_INCLUDE_DIR})
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D WITH_GLEW")
ENDIF(${GLEW_FOUND})

find_package(Gmodule)

set(LINK_LIBRARIES_ALL  ${OPENCV_LIBRARIES}
${GLUT_LIBRARIES} ${OPENGL_LIBRARY} ${GTK2_LIBRARY} ${GLEW_LIBRARY}
${GTKGLEXT_LIBS} ${GSL_LIBRARIES} ${GSL_GSLCBLAS_LIBRARY} ${TIFF_LIBRARIES}
viva)

message("Link Libraries: ${LINK_LIBRARIES_ALL}")

################################################################################
## Compilation flags and file organization
################################################################################
set(CMAKE_CXX_FLAGS         ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS} -g -pg ")
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS} -o3 ")
message("CXX flags " ${CMAKE_CXX_FLAGS})

set(CMAKE_C_FLAGS         ${CMAKE_CXX_FLAGS})
set(CMAKE_C_FLAGS_DEBUG   ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

message("GMODULE has been found: ${GMODULE_FOUND}")
IF(${GMODULE_FOUND})
 set(CMAKE_SHARED_LINKER_FLAGS "-lgomp ")
 set(CMAKE_EXE_LINKER_FLAGS    "-lgomp" )
 set(CMAKE_MODULE_LINKER_FLAGS "-lgomp" )
ENDIF(${GMODULE_FOUND})

include_directories(${CMAKE_SOURCE_DIR}/../core)
include_directories(${CMAKE_SOURCE_DIR}/../external/graphCuts)

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../lib/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../bin/)
set(CMAKE_SHARED_MODULE_PREFIX "plugins/")
set(CMAKE_CXX_MODULE_PATH ${CMAKE_SOURCE_DIR}/../lib/plugins)
make_directory(${LIBRARY_OUTPUT_PATH})
make_directory(${EXECUTABLE_OUTPUT_PATH})
make_directory(${CMAKE_CXX_MODULE_PATH})




################################################################################
## VIVA Library - Core of the game
################################################################################

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
../core/Point2Dotw.cpp
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
../core/Cube_C.cpp
../core/CubeFactory.cpp
#../core/GraphCut.cpp
../core/Timer.cpp
../core/TiffFactory.cpp
../core/Configuration.cpp
)

################################################################################
## Compilation of the viewer
################################################################################

IF(${GLEW_FOUND})
 add_executable(vivaView
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_ascEdit.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_draw.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_menu.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_select.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_ui.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/callbacks_load3DImageStack.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/support.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/interface.c
  ${CMAKE_SOURCE_DIR}/../viewer/src/main.c
 )
 target_link_libraries(vivaView ${LINK_LIBRARIES_ALL})
ENDIF(${GLEW_FOUND})

################################################################################
## Compilation of the plugins to the viewer
################################################################################

IF(${GLEW_FOUND})
# add_library (example MODULE ../plugins/src/example/example.cpp)
# add_library (CubeDijkstra2 MODULE ../plugins/src/CubeDijkstra2/CubeDijkstra2.cpp)
# add_library (CubeLiveWire MODULE ../plugins/src/CubeLiveWire/cubeLiveWire.cpp)
# add_library (GraphCuts MODULE ../plugins/src/Graphcuts/GraphCuts.cpp)
# add_library (LoadSeeds MODULE ../plugins/src/LoadSeeds/LoadSeeds.cpp)
ENDIF(${GLEW_FOUND})

################################################################################
## Instalation Rules
################################################################################
install(FILES ../lib/libviva.so    DESTINATION /usr/lib/)
install(PROGRAMS ../bin/vivaView      DESTINATION /usr/bin/)
install(TARGETS viva
  RUNTIME DESTINATION /usr/lib/
  LIBRARY DESTINATION /usr/lib/
  ARCHIVE DESTINATION /usr/lib/
)
install(FILES ../assets/viva.png       DESTINATION /usr/share/icons/)
install(FILES ../assets/viva.png       DESTINATION /usr/share/pixmaps/)
install(FILES ../assets/viva.desktop   DESTINATION /usr/share/applications/)

################################################################################
## Compilation of the tools
################################################################################

################################################################################
## cube tools

set( CUBE_TOOLS
  cubeToCloud
  cubeCalculateAguet
  cubeCalculateFrangi
  cubeLogistic
  cubeMergeAcrossScales
  cubeAverage
  cubeAt
  cubeRenderNeuron
  cubeGaussianPyramid
  cubeGaussianMask
  cubeMultiply
  cubeMax
  cubeMIP
  cubeMaxColor
  cubeThreshold
  cubeOutputValuesAtCloud
  cubeCut
  cubeConvertToUchar
  cubeDijkstra2
  cubeDecimate
  cubeRescale
  cubeDecimateAcrossScales
  cubeDerivative
  cubeDerivativesOrder
  cubeStatistics
  cubeFixNans
  cubeFromImageStack
  cubeFromImageStackRaw
  cubeFromImageStack-16bitTIFF
  cubeCombineFromList
  cubeCombineFromList-16bit
  cubeToImageStack
  cubeCalculateDerivativesSecond
  cubeHermiteMask
  cube-log10
  cubeFixAguetThreshold
)

foreach( cube_tool ${CUBE_TOOLS})
  add_executable(${cube_tool} ${CMAKE_SOURCE_DIR}/../tools/cube/${cube_tool}.cpp)
  target_link_libraries(${cube_tool} ${LINK_LIBRARIES_ALL})
endforeach(cube_tool)



################################################################################
## image tools
################################################################################

set( IMAGE_TOOLS
  imageConvolution2D
  imageRenderSWC
  imageRenderNeuron
  imageMask
  imageFindSoma
  imageGaussianPyramid
  imageMergeAcrossScales
  imageDerivative
  imageThreshold
  imageGradient
  imageDistanceTransform
  imageHistogram
  imageToCloudWithWeights
  imageToCloud
  imageLogistic
  imageCalculateHessian
  imageFindBorders
)

foreach( image_tool ${IMAGE_TOOLS})
  add_executable(${image_tool} ${CMAKE_SOURCE_DIR}/../tools/image/${image_tool}.cpp)
  target_link_libraries(${image_tool} ${LINK_LIBRARIES_ALL})
endforeach(image_tool)




################################################################################
## graph tools
################################################################################
set( GRAPH_TOOLS
  # graphToCloud
  # graphPrim
  graphCompleteChangeWeights
  # graphCompleteAddSoma
  graphAddWeightsFromCubeDijkstra
  graphDijkstra
  # graphPrimFromCptGraph
  # graphCptFromCloud
  # graphFindLeaves
  # graphPrune
  graphAddScale
  # cloudSplit
  # cloudEliminateWidth
  # cloudCombine
  cloudToText
  # cloudEvaluate
  # cloudEliminatePointsFromImage
)

foreach( graph_tool ${GRAPH_TOOLS})
  add_executable(${graph_tool} ${CMAKE_SOURCE_DIR}/../tools/graph/${graph_tool}.cpp)
  target_link_libraries(${graph_tool} ${LINK_LIBRARIES_ALL})
endforeach(graph_tool)



################################################################################
## neuron tools
################################################################################
set( NEURON_TOOLS
  # neuronOutputAllEdgeLengths
  # neuronFixDuplicatePoints
  # neuronToCloud
  neuronToCloudI
  # neuronStatistics
  # neuronFlat
)
foreach( neuron_tool ${NEURON_TOOLS})
  add_executable(${neuron_tool} ${CMAKE_SOURCE_DIR}/../tools/neuron/${neuron_tool}.cpp)
  target_link_libraries(${neuron_tool} ${LINK_LIBRARIES_ALL})
endforeach(neuron_tool)



################################################################################
## Diadem tools
################################################################################
set( DIADEM_TOOLS
  diademCloudFromGt
  diademGraphToMatlabList
  diademGraphToMatlabList-GT
  diademGraphToSWC
  diademSWCToGraph
  diademSWCisBinaryTree
  diademAddScaleFromImage
  diademOutputToCloud
  diademTreeToHighResTree
)

foreach( diadem_tool ${DIADEM_TOOLS})
  add_executable(${diadem_tool} ${CMAKE_SOURCE_DIR}/../tools/diadem/${diadem_tool}.cpp)
  target_link_libraries(${diadem_tool} ${LINK_LIBRARIES_ALL})
endforeach(diadem_tool)

################################################################################
## KMST tools
################################################################################
set( KMST_TOOLS
  kMSTFileToGraph
  kMSTgraphToHighResgraph
  kMSTanalyzeDir
  kMSTevaluateDir
  kMSTgraphAddScale
  kMSTGraphToFile
)

foreach( kmst_tool ${KMST_TOOLS})
  add_executable(${kmst_tool} ${CMAKE_SOURCE_DIR}/../tools/kmst/${kmst_tool}.cpp)
  target_link_libraries(${kmst_tool} ${LINK_LIBRARIES_ALL})
endforeach(kmst_tool)



################################################################################
## Test examples
################################################################################

# add_executable(testSWC ${CMAKE_SOURCE_DIR}/../tests/testSWC.cpp)
# target_link_libraries(testSWC ${LINK_LIBRARIES_ALL})

add_executable(graph ${CMAKE_SOURCE_DIR}/../tests/graph.cpp)
target_link_libraries(graph ${LINK_LIBRARIES_ALL})


# add_executable(tryManyPathsCloud ${CMAKE_SOURCE_DIR}/../tests/tryManyPathsCloud.cpp)
# target_link_libraries(tryManyPathsCloud ${LINK_LIBRARIES_ALL})

# add_executable(cubeMemmoryDeletion ${CMAKE_SOURCE_DIR}/../tests/cubeMemmoryDeletion.cpp)
# target_link_libraries(cubeMemmoryDeletion ${LINK_LIBRARIES_ALL})

# add_executable(cubeTIFF ${CMAKE_SOURCE_DIR}/../tests/cubeTIFF.cpp)
# target_link_libraries(cubeTIFF ${LINK_LIBRARIES_ALL})

# add_executable(idsToVolume ${CMAKE_SOURCE_DIR}/../tests/idsToVolume.cpp)
# target_link_libraries(idsToVolume ${LINK_LIBRARIES_ALL})

# add_executable(checkNonMaxSup ${CMAKE_SOURCE_DIR}/../tests/checkNonMaxSup.cpp)
# target_link_libraries(checkNonMaxSup ${LINK_LIBRARIES_ALL})


# add_executable(convolutionsFast ${CMAKE_SOURCE_DIR}/../tests/convolutionsFast.cpp)
# target_link_libraries(convolutionsFast ${LINK_LIBRARIES_ALL})

# add_executable(convolutionsSlow ${CMAKE_SOURCE_DIR}/../tests/convolutionsSlow.cpp)
# target_link_libraries(convolutionsSlow ${LINK_LIBRARIES_ALL})


# # add_executable(MSTthroughShortestPath ${CMAKE_SOURCE_DIR}/../tests/MSTthroughShortestPath.cpp)
# target_link_libraries(MSTthroughShortestPath ${LINK_LIBRARIES_ALL})

# add_executable(graphToDisplayList ${CMAKE_SOURCE_DIR}/../tests/graphToDisplayList.cpp)
# target_link_libraries(graphToDisplayList ${LINK_LIBRARIES_ALL})

# add_executable(gslMinimization ${CMAKE_SOURCE_DIR}/../tests/gslMinimization.cpp)
# target_link_libraries(gslMinimization ${LINK_LIBRARIES_ALL})

# add_executable(clone ${CMAKE_SOURCE_DIR}/../tests/clone.cpp)
# target_link_libraries(clone ${LINK_LIBRARIES_ALL})

# add_executable(configuration ${CMAKE_SOURCE_DIR}/../tests/configuration.cpp)
# target_link_libraries(configuration ${LINK_LIBRARIES_ALL})

# add_executable(openmp ${CMAKE_SOURCE_DIR}/../tests/openmp.cpp)
# target_link_libraries(openmp ${LINK_LIBRARIES_ALL})

# add_executable(copyFile ${CMAKE_SOURCE_DIR}/../tests/copyFile.cpp)
# target_link_libraries(copyFile ${LINK_LIBRARIES_ALL})

