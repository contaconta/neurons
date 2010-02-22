
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

# set(CMAKE_BUILD_TYPE debug)  #debug none release distribution
set(CMAKE_BUILD_TYPE release) #debug none release distribution

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

IF(${GLEW_FOUND})
add_executable(cubeToCloud ${CMAKE_SOURCE_DIR}/../tools/cube/cubeToCloud.cpp)
target_link_libraries(cubeToCloud ${LINK_LIBRARIES_ALL})
add_executable(cubeCalculateAguet ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCalculateAguet.cpp)
target_link_libraries(cubeCalculateAguet ${LINK_LIBRARIES_ALL})
add_executable(cubeCalculateFrangi ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCalculateFrangi.cpp)
target_link_libraries(cubeCalculateFrangi ${LINK_LIBRARIES_ALL})
add_executable(cubeLogistic ${CMAKE_SOURCE_DIR}/../tools/cube/cubeLogistic.cpp)
target_link_libraries(cubeLogistic ${LINK_LIBRARIES_ALL})

ENDIF(${GLEW_FOUND})

# add_executable(cubeMergeAcrossScales ${CMAKE_SOURCE_DIR}/../tools/cube/cubeMergeAcrossScales.cpp)
# target_link_libraries(cubeMergeAcrossScales ${LINK_LIBRARIES_ALL})

add_executable(cubeAverage ${CMAKE_SOURCE_DIR}/../tools/cube/cubeAverage.cpp)
target_link_libraries(cubeAverage ${LINK_LIBRARIES_ALL})


# add_executable(cubeAt ${CMAKE_SOURCE_DIR}/../tools/cube/cubeAt.cpp)
# target_link_libraries(cubeAt ${LINK_LIBRARIES_ALL})

add_executable(cubeRenderNeuron ${CMAKE_SOURCE_DIR}/../tools/cube/cubeRenderNeuron.cpp)
target_link_libraries(cubeRenderNeuron ${LINK_LIBRARIES_ALL})


add_executable(cubeGaussianPyramid ${CMAKE_SOURCE_DIR}/../tools/cube/cubeGaussianPyramid.cpp)
target_link_libraries(cubeGaussianPyramid ${LINK_LIBRARIES_ALL})

# add_executable(cubeGaussianMask ${CMAKE_SOURCE_DIR}/../tools/cube/cubeGaussianMask.cpp)
# target_link_libraries(cubeGaussianMask ${LINK_LIBRARIES_ALL})

# add_executable(cubeMultiply ${CMAKE_SOURCE_DIR}/../tools/cube/cubeMultiply.cpp)
# target_link_libraries(cubeMultiply ${LINK_LIBRARIES_ALL})

# add_executable(cubeMax ${CMAKE_SOURCE_DIR}/../tools/cube/cubeMax.cpp)
# target_link_libraries(cubeMax ${LINK_LIBRARIES_ALL})

add_executable(cubeMIP ${CMAKE_SOURCE_DIR}/../tools/cube/cubeMIP.cpp)
target_link_libraries(cubeMIP ${LINK_LIBRARIES_ALL})

# add_executable(cubeMaxColor ${CMAKE_SOURCE_DIR}/../tools/cube/cubeMaxColor.cpp)
# target_link_libraries(cubeMaxColor ${LINK_LIBRARIES_ALL})

add_executable(cubeThreshold ${CMAKE_SOURCE_DIR}/../tools/cube/cubeThreshold.cpp)
target_link_libraries(cubeThreshold ${LINK_LIBRARIES_ALL})

# # add_executable(cubeOutputValuesAtCloud ${CMAKE_SOURCE_DIR}/../tools/cube/cubeOutputValuesAtCloud.cpp)
# target_link_libraries(cubeOutputValuesAtCloud ${LINK_LIBRARIES_ALL})

# add_executable(cubeCut ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCut.cpp)
# target_link_libraries(cubeCut ${LINK_LIBRARIES_ALL})

# add_executable(cubeConvertToUchar ${CMAKE_SOURCE_DIR}/../tools/cube/cubeConvertToUchar.cpp)
# target_link_libraries(cubeConvertToUchar ${LINK_LIBRARIES_ALL})


# add_executable(cubeDijkstra2 ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDijkstra2.cpp)
# target_link_libraries(cubeDijkstra2 ${LINK_LIBRARIES_ALL})

add_executable(cubeDecimate ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDecimate.cpp)
target_link_libraries(cubeDecimate ${LINK_LIBRARIES_ALL})

# add_executable(cubeRescale ${CMAKE_SOURCE_DIR}/../tools/cube/cubeRescale.cpp)
# target_link_libraries(cubeRescale ${LINK_LIBRARIES_ALL})

add_executable(cubeDecimateAcrossScales ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDecimateAcrossScales.cpp)
target_link_libraries(cubeDecimateAcrossScales ${LINK_LIBRARIES_ALL})


# add_executable(cubeDerivative ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDerivative.cpp)
# target_link_libraries(cubeDerivative ${LINK_LIBRARIES_ALL})

# # add_executable(cubeDerivativesOrder ${CMAKE_SOURCE_DIR}/../tools/cube/cubeDerivativesOrder.cpp)
# target_link_libraries(cubeDerivativesOrder ${LINK_LIBRARIES_ALL})

add_executable(cubeStatistics ${CMAKE_SOURCE_DIR}/../tools/cube/cubeStatistics.cpp)
target_link_libraries(cubeStatistics ${LINK_LIBRARIES_ALL})

add_executable(cubeFixNans ${CMAKE_SOURCE_DIR}/../tools/cube/cubeFixNans.cpp)
target_link_libraries(cubeFixNans ${LINK_LIBRARIES_ALL})

# add_executable(cubeFromImageStack ${CMAKE_SOURCE_DIR}/../tools/cube/cubeFromImageStack.cpp)
# target_link_libraries(cubeFromImageStack ${LINK_LIBRARIES_ALL})

add_executable(cubeFromImageStackRaw ${CMAKE_SOURCE_DIR}/../tools/cube/cubeFromImageStackRaw.cpp)
target_link_libraries(cubeFromImageStackRaw ${LINK_LIBRARIES_ALL})


# add_executable(cubeFromImageStack-16bitTIFF ${CMAKE_SOURCE_DIR}/../tools/cube/cubeFromImageStack-16bitTIFF.cpp)
# target_link_libraries(cubeFromImageStack-16bitTIFF ${LINK_LIBRARIES_ALL})


# add_executable(cubeCombineFromList ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCombineFromList.cpp)
# target_link_libraries(cubeCombineFromList ${LINK_LIBRARIES_ALL})

# add_executable(cubeCombineFromList-16bit ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCombineFromList-16bit.cpp)
# target_link_libraries(cubeCombineFromList-16bit ${LINK_LIBRARIES_ALL})

add_executable(cubeToImageStack ${CMAKE_SOURCE_DIR}/../tools/cube/cubeToImageStack.cpp)
target_link_libraries(cubeToImageStack ${LINK_LIBRARIES_ALL})

# # add_executable(cubeCalculateDerivativesSecond ${CMAKE_SOURCE_DIR}/../tools/cube/cubeCalculateDerivativesSecond.cpp)
# target_link_libraries(cubeCalculateDerivativesSecond ${LINK_LIBRARIES_ALL})

# add_executable(cubeHermiteMask ${CMAKE_SOURCE_DIR}/../tools/cube/cubeHermiteMask.cpp)
# target_link_libraries(cubeHermiteMask ${LINK_LIBRARIES_ALL})

# add_executable(cube-log10 ${CMAKE_SOURCE_DIR}/../tools/cube/cube-log10.cpp)
# target_link_libraries(cube-log10 ${LINK_LIBRARIES_ALL})


# # add_executable(cubeFixAguetThreshold ${CMAKE_SOURCE_DIR}/../tools/cube/cubeFixAguetThreshold.cpp)
# target_link_libraries(cubeFixAguetThreshold ${LINK_LIBRARIES_ALL})


################################################################################
## image tools
################################################################################

# add_executable(imageConvolution2D ${CMAKE_SOURCE_DIR}/../tools/image/imageConvolution2D.cpp)
# target_link_libraries(imageConvolution2D ${LINK_LIBRARIES_ALL})


# add_executable(imageRenderSWC ${CMAKE_SOURCE_DIR}/../tools/image/imageRenderSWC.cpp)
# target_link_libraries(imageRenderSWC ${LINK_LIBRARIES_ALL})

add_executable(imageRenderNeuron ${CMAKE_SOURCE_DIR}/../tools/image/imageRenderNeuron.cpp)
target_link_libraries(imageRenderNeuron ${LINK_LIBRARIES_ALL})


add_executable(imageMask ${CMAKE_SOURCE_DIR}/../tools/image/imageMask.cpp)
target_link_libraries(imageMask ${LINK_LIBRARIES_ALL})

# add_executable(imageFindSoma ${CMAKE_SOURCE_DIR}/../tools/image/imageFindSoma.cpp)
# target_link_libraries(imageFindSoma ${LINK_LIBRARIES_ALL})

# # add_executable(imageGaussianPyramid ${CMAKE_SOURCE_DIR}/../tools/image/imageGaussianPyramid.cpp)
# target_link_libraries(imageGaussianPyramid ${LINK_LIBRARIES_ALL})

# # add_executable(imageMergeAcrossScales ${CMAKE_SOURCE_DIR}/../tools/image/imageMergeAcrossScales.cpp)
# target_link_libraries(imageMergeAcrossScales ${LINK_LIBRARIES_ALL})

# add_executable(imageDerivative ${CMAKE_SOURCE_DIR}/../tools/image/imageDerivative.cpp)
# target_link_libraries(imageDerivative ${LINK_LIBRARIES_ALL})

# add_executable(imageThreshold ${CMAKE_SOURCE_DIR}/../tools/image/imageThreshold.cpp)
# target_link_libraries(imageThreshold ${LINK_LIBRARIES_ALL})

# add_executable(imageGradient ${CMAKE_SOURCE_DIR}/../tools/image/imageGradient.cpp)
# target_link_libraries(imageGradient ${LINK_LIBRARIES_ALL})

# add_executable(imageDistanceTransform
  # ${CMAKE_SOURCE_DIR}/../tools/image/imageDistanceTransform.cpp)
# target_link_libraries(imageDistanceTransform ${LINK_LIBRARIES_ALL})

# IF(${GLEW_FOUND})
# add_executable(imageHistogram ${CMAKE_SOURCE_DIR}/../tools/image/imageHistogram.cpp)
# target_link_libraries(imageHistogram ${LINK_LIBRARIES_ALL})
# # add_executable(imageToCloudWithWeights ${CMAKE_SOURCE_DIR}/../tools/image/imageToCloudWithWeights.cpp)
# target_link_libraries(imageToCloudWithWeights ${LINK_LIBRARIES_ALL})

# add_executable(imageToCloud ${CMAKE_SOURCE_DIR}/../tools/image/imageToCloud.cpp)
# target_link_libraries(imageToCloud ${LINK_LIBRARIES_ALL})

# add_executable(imageLogistic ${CMAKE_SOURCE_DIR}/../tools/image/imageLogistic.cpp)
# target_link_libraries(imageLogistic ${LINK_LIBRARIES_ALL})

# # add_executable(imageCalculateHessian ${CMAKE_SOURCE_DIR}/../tools/image/imageCalculateHessian.cpp)
# target_link_libraries(imageCalculateHessian ${LINK_LIBRARIES_ALL})
# ENDIF(${GLEW_FOUND})





################################################################################
## graph tools
################################################################################

add_executable(graphToCloud ${CMAKE_SOURCE_DIR}/../tools/graph/graphToCloud.cpp)
target_link_libraries(graphToCloud ${LINK_LIBRARIES_ALL})

add_executable(graphPrim ${CMAKE_SOURCE_DIR}/../tools/graph/graphPrim.cpp)
target_link_libraries(graphPrim ${LINK_LIBRARIES_ALL})

add_executable(graphCompleteChangeWeights ${CMAKE_SOURCE_DIR}/../tools/graph/graphCompleteChangeWeights.cpp)
target_link_libraries(graphCompleteChangeWeights ${LINK_LIBRARIES_ALL})

# # add_executable(graphCompleteAddSoma ${CMAKE_SOURCE_DIR}/../tools/graph/graphCompleteAddSoma.cpp)
# target_link_libraries(graphCompleteAddSoma ${LINK_LIBRARIES_ALL})

add_executable(graphAddWeightsFromCubeDijkstra ${CMAKE_SOURCE_DIR}/../tools/graph/graphAddWeightsFromCubeDijkstra.cpp)
target_link_libraries(graphAddWeightsFromCubeDijkstra ${LINK_LIBRARIES_ALL})

add_executable(graphDijkstra ${CMAKE_SOURCE_DIR}/../tools/graph/graphDijkstra.cpp)
target_link_libraries(graphDijkstra ${LINK_LIBRARIES_ALL})

add_executable(graphPrimFromCptGraph ${CMAKE_SOURCE_DIR}/../tools/graph/graphPrimFromCptGraph.cpp)
target_link_libraries(graphPrimFromCptGraph ${LINK_LIBRARIES_ALL})

add_executable(graphCptFromCloud ${CMAKE_SOURCE_DIR}/../tools/graph/graphCptFromCloud.cpp)
target_link_libraries(graphCptFromCloud ${LINK_LIBRARIES_ALL})

# add_executable(graphFindLeaves ${CMAKE_SOURCE_DIR}/../tools/graph/graphFindLeaves.cpp)
# target_link_libraries(graphFindLeaves ${LINK_LIBRARIES_ALL})

# add_executable(graphPrune ${CMAKE_SOURCE_DIR}/../tools/graph/graphPrune.cpp)
# target_link_libraries(graphPrune ${LINK_LIBRARIES_ALL})

# add_executable(cloudSplit ${CMAKE_SOURCE_DIR}/../tools/graph/cloudSplit.cpp)
# target_link_libraries(cloudSplit ${LINK_LIBRARIES_ALL})

# add_executable(cloudEliminateWidth ${CMAKE_SOURCE_DIR}/../tools/graph/cloudEliminateWidth.cpp)
# target_link_libraries(cloudEliminateWidth ${LINK_LIBRARIES_ALL})


#add_executable(cloudCombine ${CMAKE_SOURCE_DIR}/../tools/graph/cloudCombine.cpp)
#target_link_libraries(cloudCombine ${LINK_LIBRARIES_ALL})

# add_executable(cloudToText ${CMAKE_SOURCE_DIR}/../tools/graph/cloudToText.cpp)
# target_link_libraries(cloudToText ${LINK_LIBRARIES_ALL})


# add_executable(cloudEvaluate ${CMAKE_SOURCE_DIR}/../tools/graph/cloudEvaluate.cpp)
# target_link_libraries(cloudEvaluate ${LINK_LIBRARIES_ALL})

# # add_executable(cloudEliminatePointsFromImage ${CMAKE_SOURCE_DIR}/../tools/graph/cloudEliminatePointsFromImage.cpp)
# target_link_libraries(cloudEliminatePointsFromImage ${LINK_LIBRARIES_ALL})


################################################################################
## neuron tools
################################################################################

# # add_executable(neuronOutputAllEdgeLengths ${CMAKE_SOURCE_DIR}/../tools/neuron/neuronOutputAllEdgeLengths.cpp)
# target_link_libraries(neuronOutputAllEdgeLengths ${LINK_LIBRARIES_ALL})

# # add_executable(neuronFixDuplicatePoints ${CMAKE_SOURCE_DIR}/../tools/neuron/neuronFixDuplicatePoints.cpp)
# target_link_libraries(neuronFixDuplicatePoints ${LINK_LIBRARIES_ALL})

add_executable(neuronToCloud ${CMAKE_SOURCE_DIR}/../tools/neuron/neuronToCloud.cpp)
target_link_libraries(neuronToCloud ${LINK_LIBRARIES_ALL})

# add_executable(neuronToCloudI ${CMAKE_SOURCE_DIR}/../tools/neuron/neuronToCloudI.cpp)
# target_link_libraries(neuronToCloudI ${LINK_LIBRARIES_ALL})

# add_executable(neuronStatistics ${CMAKE_SOURCE_DIR}/../tools/neuron/neuronStatistics.cpp)
# target_link_libraries(neuronStatistics ${LINK_LIBRARIES_ALL})

# add_executable(neuronFlat ${CMAKE_SOURCE_DIR}/../tools/neuron/neuronFlat.cpp)
# target_link_libraries(neuronFlat ${LINK_LIBRARIES_ALL})


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
)

foreach( diadem_tool ${DIADEM_TOOLS})
  add_executable(${diadem_tool} ${CMAKE_SOURCE_DIR}/../tools/diadem/${diadem_tool}.cpp)
  target_link_libraries(${diadem_tool} ${LINK_LIBRARIES_ALL})
endforeach(diadem_tool)

# add_executable(diademGraphToSWC ${CMAKE_SOURCE_DIR}/../tools/diadem/diademGraphToSWC.cpp)
# target_link_libraries(diademGraphToSWC ${LINK_LIBRARIES_ALL})

# add_executable(diademSWCToGraph ${CMAKE_SOURCE_DIR}/../tools/diadem/diademSWCToGraph.cpp)
# target_link_libraries(diademSWCToGraph ${LINK_LIBRARIES_ALL})

# # add_executable(diademSWCisBinaryTree ${CMAKE_SOURCE_DIR}/../tools/diadem/diademSWCisBinaryTree.cpp)
# target_link_libraries(diademSWCisBinaryTree ${LINK_LIBRARIES_ALL})


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


# add_executable(kMSTFileToGraph ${CMAKE_SOURCE_DIR}/../tests/kMSTFileToGraph.cpp)
# target_link_libraries(kMSTFileToGraph ${LINK_LIBRARIES_ALL})

# # add_executable(kMSTgraphToHighResgraph ${CMAKE_SOURCE_DIR}/../tests/kMSTgraphToHighResgraph.cpp)
# target_link_libraries(kMSTgraphToHighResgraph ${LINK_LIBRARIES_ALL})

# add_executable(kMSTanalyzeDir ${CMAKE_SOURCE_DIR}/../tests/kMSTanalyzeDir.cpp)
# target_link_libraries(kMSTanalyzeDir ${LINK_LIBRARIES_ALL})

# add_executable(kMSTevaluateDir ${CMAKE_SOURCE_DIR}/../tests/kMSTevaluateDir.cpp)
# target_link_libraries(kMSTevaluateDir ${LINK_LIBRARIES_ALL})

# add_executable(kMSTgraphAddScale ${CMAKE_SOURCE_DIR}/../tests/kMSTgraphAddScale.cpp)
# target_link_libraries(kMSTgraphAddScale ${LINK_LIBRARIES_ALL})

# add_executable(kMSTGraphToFile ${CMAKE_SOURCE_DIR}/../tests/kMSTGraphToFile.cpp)
# target_link_libraries(kMSTGraphToFile ${LINK_LIBRARIES_ALL})


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

