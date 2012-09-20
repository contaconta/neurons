/************************************************************************/
/* (c) 2009-2011 Ecole Polytechnique Federale de Lausanne               */
/* All rights reserved.                                                 */
/*                                                                      */
/* EPFL grants a non-exclusive and non-transferable license for non     */
/* commercial use of the Software for education and research purposes   */
/* only. Any other use of the Software is expressly excluded.           */
/*                                                                      */
/* Redistribution of the Software in source and binary forms, with or   */
/* without modification, is not permitted.                              */
/*                                                                      */
/* Written by Engin Turetken.                                           */
/* Adapted to a matlab struct by Fethallah Benmansour                   */
/* For the Sinergia project                                             */
/*                                                                      */
/* http://cvlab.epfl.ch/research/body/surv                              */
/* Contact <pom@epfl.ch> for comments & bug reports.                    */
/************************************************************************/

#include "ksp_graph.h"
#include <stdio.h>

KShorthestPathGraph::KShorthestPathGraph(const mxArray* Cells,
        const mxArray* CellsList,
        int temporal_windows_size,
        double spatial_windows_size) {
    // Declarations
    typedef std::pair<int, int> Edge;
    std::vector< Edge > vEdges; vEdges.clear();
    int nNoOfNodes;
    int nNoOfEdges;
    std::vector<float> edgeWeights; edgeWeights.clear();
    int numberOfFrames;
    
    // Initializations
    
    float min_prob_log = log( MIN_OCCUR_PROB / (1 - MIN_OCCUR_PROB) );
    float max_prob_log = log( MAX_OCCUR_PROB / (1 - MAX_OCCUR_PROB) );
    
    numberOfFrames = mxGetNumberOfElements(CellsList);
    
    nNoOfNodes = mxGetNumberOfElements(Cells) + 2;
    m_nSrcNodeIndx = nNoOfNodes - 2;
    m_nDstNodeIndx = nNoOfNodes - 1;
    // field indices
    int nuclei_centroid_idx = mxGetFieldNumber(Cells, "NucleusCentroid");
    int nuclei_red_idx      = mxGetFieldNumber(Cells, "NucleusMeanRedIntensity");
    int nuclei_green_idx    = mxGetFieldNumber(Cells, "NucleusMeanGreenIntensity");
    
    // Filling in the edge array
    // Filling the edges outgoing from the source and incoming to terminal
    // Filling the edges between the source node and all the nodes in the first frame
    // and between the terminal node and all the nodes in the last frame
    
    mxArray* firstFrame = mxGetCell(CellsList, 0);
    mxArray* lastFrame  = mxGetCell(CellsList, numberOfFrames-1);
    
    int numberOfDetectionsFirst = mxGetNumberOfElements(firstFrame);
    int numberOfDetectionsLast  = mxGetNumberOfElements(lastFrame);
    
    double* firstFrameDetections = mxGetPr(firstFrame);
    for(int i = 0; i < numberOfDetectionsFirst; i++) {
        Edge e;
        e.first  = m_nSrcNodeIndx;
        if ((int) (firstFrameDetections[i]-1) < 0 ) {
            mexPrintf("from source to first frame");
        }
        e.second = (int) (firstFrameDetections[i]-1);
        vEdges.push_back(e);
        edgeWeights.push_back(0.0);
    }
    double* lastFrameDetections = mxGetPr(lastFrame);
    for(int i = 0; i < numberOfDetectionsLast; i++) {
        Edge e;
        e.first  = (int) (lastFrameDetections[i]-1);
        e.second = m_nDstNodeIndx;
        vEdges.push_back(e);
        edgeWeights.push_back(0.0);
    }
    //todo, maybe by taking only close to boundary detections
    for(int k = 1; k < numberOfFrames-1; k++) {
        mxArray* currentFrame = mxGetCell(CellsList, k);
        double* currentFrameDetections = mxGetPr(currentFrame);
        for(int i = 0; i < mxGetNumberOfElements(currentFrame); i++) {
            Edge e;
            e.first  = m_nSrcNodeIndx;
            e.second = (int) (currentFrameDetections[i]-1);
            vEdges.push_back(e);
            edgeWeights.push_back(0.0);
            
            e.first  = (int) (currentFrameDetections[i]-1);
            e.second = m_nDstNodeIndx;
            vEdges.push_back(e);
            edgeWeights.push_back(0.0);
        }
    }
    
    for( int i = 1; i < numberOfFrames; i++) {
        mxArray* currentFrame = mxGetCell(CellsList, i);
        double* currentFrameDetections = mxGetPr(currentFrame);
        int min_t = std::max(0, i - temporal_windows_size);
        for( int k = 0; k < mxGetNumberOfElements(currentFrame); k++) {
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_centroid_idx));
            double  currentDetectionRed      = mxGetScalar(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_red_idx));
            for( int j = min_t; j < i; j++) {
                mxArray* previousFrame = mxGetCell(CellsList, j);
                double*  previousFrameDetections = mxGetPr(previousFrame);
                for( int l = 0; l < mxGetNumberOfElements(previousFrame); l++) {
                    double* previousDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_centroid_idx));
                    double  previousDetectionRed      = mxGetScalar(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_red_idx));
                    // compute distance
                    double distance = sqrt((currentDetectionCentroid[0]-previousDetectionCentroid[0])*(currentDetectionCentroid[0]-previousDetectionCentroid[0])
                    +(currentDetectionCentroid[1]-previousDetectionCentroid[1])*(currentDetectionCentroid[1]-previousDetectionCentroid[1]));
                    if(distance < spatial_windows_size) {
                        Edge e;
                        e.first  = (int) (previousFrameDetections[l]-1);
                        e.second = (int) (currentFrameDetections[k]-1);
                        vEdges.push_back(e);
                        // first spatial distance
                        float prob_dist = distance / spatial_windows_size;
                        float dist_log;
                        if ( prob_dist < MIN_OCCUR_PROB )           dist_log = min_prob_log;
                        else if ( prob_dist > MAX_OCCUR_PROB )      dist_log = max_prob_log;
                        else                                        dist_log = log( prob_dist / (1 - prob_dist) );
                        
                        // second color distance
                        double intensityDiff = (float)fabs(previousDetectionRed - currentDetectionRed);
                        float intensity_log;
                        
                        if ( intensityDiff < MIN_OCCUR_PROB )       intensity_log = min_prob_log;
                        else if ( intensityDiff > MAX_OCCUR_PROB )  intensity_log = max_prob_log;
                        else                                        intensity_log = log( intensityDiff / (1 - intensityDiff) );
                        
                        // thirs, time distance
//                         float time_log = -float(temporal_windows_size)/2.0 - float(j-i);
                        
                        edgeWeights.push_back(dist_log+intensity_log);
                    }
                }
            }
        }
    }
    
    nNoOfEdges = vEdges.size();
    float *pfEdgeWeights = new float[nNoOfEdges];
    for(int i = 0; i < nNoOfEdges; i++) 
    {
        pfEdgeWeights[i] = edgeWeights[i];
    }
    
    m_pG = new KShorthestPathGraph::BaseGraphType(
            vEdges.begin(),
            vEdges.end(),
            pfEdgeWeights, nNoOfNodes);
    
    //Deallocations
    vEdges.clear();
    edgeWeights.clear();
    delete[] pfEdgeWeights;
}

KShorthestPathGraph::~KShorthestPathGraph() {
    delete m_pG;
}
