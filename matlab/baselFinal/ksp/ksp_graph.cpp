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

/// constructor using Euclidean distance
KShorthestPathGraph::KShorthestPathGraph(const mxArray* Cells,
                                         const mxArray* CellsList,
                                         int temporal_windows_size,
                                         double spatial_windows_size,
                                         double *imagesize,
                                         double distanceToBoundary)
{
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
    int nuclei_green_idx    = mxGetFieldNumber(Cells, "SomaMeanGreenIntensity");
    
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
    
    for(int k = 1; k < numberOfFrames-1; k++) 
    {
        mxArray* currentFrame = mxGetCell(CellsList, k);
        double* currentFrameDetections = mxGetPr(currentFrame);
        for(int i = 0; i < mxGetNumberOfElements(currentFrame); i++) 
        {
            Edge e;
            e.first  = (int) (currentFrameDetections[i]-1);
            e.second = m_nDstNodeIndx;
            vEdges.push_back(e);
            edgeWeights.push_back(0.0);
            
            // only for close to boundary detections
            double distToBoundary = std::min(imagesize[0], imagesize[1]);
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[i]-1), nuclei_centroid_idx));
            for(unsigned int d = 0; d <2; d++)
            {
                distToBoundary = std::min(distToBoundary, std::min(fabs(currentDetectionCentroid[0]),
                                                                   fabs(currentDetectionCentroid[0] - imagesize[0])));
            }   
            if(distToBoundary < distanceToBoundary)
            {
                e.first  = m_nSrcNodeIndx;
                e.second = (int) (currentFrameDetections[i]-1);
                vEdges.push_back(e);
                edgeWeights.push_back(0.0);
            }
        }
    }
    
    for( int i = 1; i < numberOfFrames; i++) {
        mxArray* currentFrame = mxGetCell(CellsList, i);
        double* currentFrameDetections = mxGetPr(currentFrame);
        int min_t = std::max(0, i - temporal_windows_size);
        for( int k = 0; k < mxGetNumberOfElements(currentFrame); k++) {
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_centroid_idx));
            double  currentDetectionGreen    = mxGetScalar(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_green_idx));
            for( int j = min_t; j < i; j++) {
                mxArray* previousFrame = mxGetCell(CellsList, j);
                double*  previousFrameDetections = mxGetPr(previousFrame);
                for( int l = 0; l < mxGetNumberOfElements(previousFrame); l++) {
                    double* previousDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_centroid_idx));
                    double  previousDetectionGreen    = mxGetScalar(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_green_idx));
                    // compute distance
                    double distance = sqrt((currentDetectionCentroid[0]-previousDetectionCentroid[0])*(currentDetectionCentroid[0]-previousDetectionCentroid[0])
                    +(currentDetectionCentroid[1]-previousDetectionCentroid[1])*(currentDetectionCentroid[1]-previousDetectionCentroid[1]));
                    if(distance < spatial_windows_size) {
                        Edge e;
                        e.first  = (int) (previousFrameDetections[l]-1);
                        e.second = (int) (currentFrameDetections[k] -1);
                        vEdges.push_back(e);
                        // first spatial distance
                        float prob_dist = distance / spatial_windows_size;
                        float dist_log;
                        if ( prob_dist < MIN_OCCUR_PROB )           dist_log = min_prob_log;
                        else if ( prob_dist > MAX_OCCUR_PROB )      dist_log = max_prob_log;
                        else                                        dist_log = log( prob_dist / (1 - prob_dist) );

                        edgeWeights.push_back(dist_log);//
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

/// constructor using color distance
KShorthestPathGraph::KShorthestPathGraph(const mxArray* Cells,
                                         const mxArray* CellsList,
                                         int temporal_windows_size,
                                         double spatial_windows_size,
                                         double *imagesize,
                                         double distanceToBoundary,
                                         double* intensityRange   )
{
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
    int nuclei_green_idx    = mxGetFieldNumber(Cells, "SomaMeanGreenIntensity");
    
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
    
    for(int k = 1; k < numberOfFrames-1; k++) 
    {
        mxArray* currentFrame = mxGetCell(CellsList, k);
        double* currentFrameDetections = mxGetPr(currentFrame);
        for(int i = 0; i < mxGetNumberOfElements(currentFrame); i++) 
        {
            Edge e;
            e.first  = (int) (currentFrameDetections[i]-1);
            e.second = m_nDstNodeIndx;
            vEdges.push_back(e);
            edgeWeights.push_back(0.0);
            
            // only for close to boundary detections
            double distToBoundary = std::min(imagesize[0], imagesize[1]);
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[i]-1), nuclei_centroid_idx));
            for(unsigned int d = 0; d <2; d++)
            {
                distToBoundary = std::min(distToBoundary, std::min(fabs(currentDetectionCentroid[0]),
                                                                   fabs(currentDetectionCentroid[0] - imagesize[0])));
            }   
            if(distToBoundary < distanceToBoundary)
            {
                e.first  = m_nSrcNodeIndx;
                e.second = (int) (currentFrameDetections[i]-1);
                vEdges.push_back(e);
                edgeWeights.push_back(0.0);
            }
        }
    }
    
    for( int i = 1; i < numberOfFrames; i++) {
        mxArray* currentFrame = mxGetCell(CellsList, i);
        double* currentFrameDetections = mxGetPr(currentFrame);
        int min_t = std::max(0, i - temporal_windows_size);
        for( int k = 0; k < mxGetNumberOfElements(currentFrame); k++) {
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_centroid_idx));
            double  currentDetectionGreen    = mxGetScalar(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_green_idx));
            for( int j = min_t; j < i; j++) {
                mxArray* previousFrame = mxGetCell(CellsList, j);
                double*  previousFrameDetections = mxGetPr(previousFrame);
                for( int l = 0; l < mxGetNumberOfElements(previousFrame); l++) {
                    double* previousDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_centroid_idx));
                    double  previousDetectionGreen    = mxGetScalar(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_green_idx));
                    // compute distance
                    double distance = sqrt((currentDetectionCentroid[0]-previousDetectionCentroid[0])*(currentDetectionCentroid[0]-previousDetectionCentroid[0])
                    +(currentDetectionCentroid[1]-previousDetectionCentroid[1])*(currentDetectionCentroid[1]-previousDetectionCentroid[1]));
                    if(distance < spatial_windows_size) {
                        Edge e;
                        e.first  = (int) (previousFrameDetections[l]-1);
                        e.second = (int) (currentFrameDetections[k] -1);
                        vEdges.push_back(e);
                        
                        // color distance
                        double intensityDiff = 10.0*(double)fabs(previousDetectionGreen - currentDetectionGreen) / (intensityRange[3] - intensityRange[2]);
                        float intensity_log;
                        
                        if ( intensityDiff < MIN_OCCUR_PROB )       intensity_log = min_prob_log;
                        else if ( intensityDiff > MAX_OCCUR_PROB )  intensity_log = max_prob_log;//TODO
                        else                                        intensity_log = log( intensityDiff / (1 - intensityDiff) );
                        
                        edgeWeights.push_back(intensity_log);//
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


/// constructor using EMD between histograms of green somata 
/// and learnt sigmoid fitting parameters
KShorthestPathGraph::KShorthestPathGraph(const mxArray* Cells,
                      const mxArray* CellsList,
                      int temporal_windows_size,
                      double spatial_windows_size,
                      double *imagesize,
                      double distanceToBoundary,
                      const mxArray* penaltyMatrix,
                      double *sigmoidParams)
{
    // Declarations
    typedef std::pair<int, int> Edge;
    std::vector< Edge > vEdges; vEdges.clear();
    int nNoOfNodes;
    int nNoOfEdges;
    std::vector<float> edgeWeights; edgeWeights.clear();
    int numberOfFrames;
    
    // Initializations
    
    float min_prob_log = -log( MIN_OCCUR_PROB / (1 - MIN_OCCUR_PROB) );
    float max_prob_log = -log( MAX_OCCUR_PROB / (1 - MAX_OCCUR_PROB) );
    
    numberOfFrames = mxGetNumberOfElements(CellsList);
    
    nNoOfNodes = mxGetNumberOfElements(Cells) + 2;
    m_nSrcNodeIndx = nNoOfNodes - 2;
    m_nDstNodeIndx = nNoOfNodes - 1;
    // field indices
    int nuclei_centroid_idx = mxGetFieldNumber(Cells, "NucleusCentroid");
    int nuclei_red_idx      = mxGetFieldNumber(Cells, "NucleusMeanRedIntensity");
    int nuclei_green_idx    = mxGetFieldNumber(Cells, "SomaMeanGreenIntensity");
    int soma_greenHist_idx  = mxGetFieldNumber(Cells, "SomaHistGreen");
    
    // Filling in the edge array
    // Filling the edges outgoing from the source and incoming to terminal
    // Filling the edges between the source node and all the nodes in the first frame
    // and between the terminal node and all the nodes in the last frame
    
    mxArray* firstFrame = mxGetCell(CellsList, 0);
    mxArray* lastFrame  = mxGetCell(CellsList, numberOfFrames-1);
    
    // mxArray* containing the -1 as a scalar value
    mxArray *minusOne = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *ptr = mxGetPr(minusOne);
    ptr[0] = -1;
    // done with dummy minusOne
    
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
    
    for(int k = 1; k < numberOfFrames-1; k++) 
    {
        mxArray* currentFrame = mxGetCell(CellsList, k);
        double* currentFrameDetections = mxGetPr(currentFrame);
        for(int i = 0; i < mxGetNumberOfElements(currentFrame); i++) 
        {
            Edge e;
            e.first  = (int) (currentFrameDetections[i]-1);
            e.second = m_nDstNodeIndx;
            vEdges.push_back(e);
            edgeWeights.push_back(0.0);
            
            // only for close to boundary detections
            double distToBoundary = std::min(imagesize[0], imagesize[1]);
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[i]-1), nuclei_centroid_idx));
            for(unsigned int d = 0; d <2; d++)
            {
                distToBoundary = std::min(distToBoundary, std::min(fabs(currentDetectionCentroid[0]),
                                                                   fabs(currentDetectionCentroid[0] - imagesize[0])));
            }   
            if(distToBoundary < distanceToBoundary)
            {
                e.first  = m_nSrcNodeIndx;
                e.second = (int) (currentFrameDetections[i]-1);
                vEdges.push_back(e);
                edgeWeights.push_back(0.0);
            }
        }
    }
    double min_emd_dist = 1e9;
    double max_emd_dist  = -1e9;
    for( int i = 1; i < numberOfFrames; i++) {
        mxArray* currentFrame = mxGetCell(CellsList, i);
        double* currentFrameDetections = mxGetPr(currentFrame);
        int min_t = std::max(0, i - temporal_windows_size);
        for( int k = 0; k < mxGetNumberOfElements(currentFrame); k++) {
            double* currentDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_centroid_idx));
            double  currentDetectionGreen    = mxGetScalar(mxGetFieldByNumber(Cells, int(currentFrameDetections[k]-1), nuclei_green_idx));
            for( int j = min_t; j < i; j++) {
                mxArray* previousFrame = mxGetCell(CellsList, j);
                double*  previousFrameDetections = mxGetPr(previousFrame);
                for( int l = 0; l < mxGetNumberOfElements(previousFrame); l++) {
                    double* previousDetectionCentroid = mxGetPr(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_centroid_idx));
                    double  previousDetectionGreen    = mxGetScalar(mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), nuclei_green_idx));
                    // compute distance
                    double distance = sqrt((currentDetectionCentroid[0]-previousDetectionCentroid[0])*(currentDetectionCentroid[0]-previousDetectionCentroid[0])
                    +(currentDetectionCentroid[1]-previousDetectionCentroid[1])*(currentDetectionCentroid[1]-previousDetectionCentroid[1]));
                    if(distance < spatial_windows_size) {
                        Edge e;
                        e.first  = (int) (previousFrameDetections[l]-1);
                        e.second = (int) (currentFrameDetections[k]-1);
                        vEdges.push_back(e);
                        // Here, call EMD distance stuff
                        mxArray *rhs[4], *lhs[1];
                        rhs[0] = mxGetFieldByNumber(Cells, int(previousFrameDetections[l]-1), soma_greenHist_idx);
                        rhs[1] = mxGetFieldByNumber(Cells, int(currentFrameDetections[k] -1), soma_greenHist_idx);
                        rhs[2] = const_cast<mxArray* >( penaltyMatrix );
                        rhs[3] = minusOne;
                        if ( mexCallMATLAB(1, lhs, 4, rhs, "emd_hat_gd_metric_mex"))
                            mexErrMsgTxt("Problem calling EMD distance. Make sure FastEMD has been compiled !!");
                        
                        double emd_dist = mxGetScalar(lhs[0]);
                        if(emd_dist < min_emd_dist)
                            min_emd_dist = emd_dist;
                        if(emd_dist > max_emd_dist)
                            max_emd_dist = emd_dist;
                        
                        double prob_emd = 1.0 / (1.0 + exp(-(sigmoidParams[0] + sigmoidParams[1]*emd_dist)));
                        double emd_weight = 0.0;
                        if      ( prob_emd < MIN_OCCUR_PROB )       emd_weight = min_prob_log;
                        else if ( prob_emd > MAX_OCCUR_PROB )       emd_weight = max_prob_log;
                        else                                        emd_weight= -log( prob_emd / (1 - prob_emd) );
                        
                        edgeWeights.push_back(emd_weight);
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
