/**
 * Copyright (C) 2010 Engin Turetken
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation (http://www.gnu.org/licenses/gpl.txt )
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 * @version 1.0 24/08/2010
 * @author Engin Turetken <engin.turetken@epfl.ch>
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include "SWCFileIO.h"
#include "SWCGraphVertex.h"
#include "SWCTreeDownsampler.h"
#include "SWCTreeSmoother.h"


bool ReadParameters(int argc, char **argv, 
					std::string& sInSWCFileName, 
					std::string& sOutSWCFileName,
					double& fEuclideanDistThr,
					int& nSlidingWndSizeForSmoothing)
{
	int iarg = 1;
	bool bsInSWCFileName = false; 
	bool bsOutSWCFileName = false;
	bool bfEuclideanDistThr = false;
	bool bnSlidingWndSizeForSmoothing = false;
	
	while (iarg < argc)
    {
		if (strcmp(argv[iarg],"-i")==0)
        {
			sInSWCFileName = argv[++iarg];
			bsInSWCFileName = true;
        }
		else if (strcmp(argv[iarg],"-o")==0)
        {
			sOutSWCFileName = argv[++iarg];
			bsOutSWCFileName = true;
        }
		else if (strcmp(argv[iarg],"-ws")==0)
        {
			nSlidingWndSizeForSmoothing = atoi(argv[++iarg]);
			bnSlidingWndSizeForSmoothing = true;
        }
		else if (strcmp(argv[iarg],"-t")==0)
        {
			fEuclideanDistThr = atof(argv[++iarg]);
			bfEuclideanDistThr = true;
        }
		iarg++;
    }
	
	if( !bsInSWCFileName )
    {
		std::cout << "Input swc file is not specified." << std::endl;
		return false;
    }
	
	if( !bsOutSWCFileName )
    {
		std::cout << "Output swc file is not specified." << std::endl;
		return false;
    }
	
	if( !bfEuclideanDistThr )
    {
		std::cout << "Maximum accepteple Euclidean error threshold is not specified." << std::endl;
		return false;
    }
	
	if( bnSlidingWndSizeForSmoothing && (nSlidingWndSizeForSmoothing < 1) )
	{
		std::cout << "Sliding window size can't be smaller than 1." << std::endl;
		return false;
	}

	if( bfEuclideanDistThr && (fEuclideanDistThr < 0) )
	{
		std::cout << "Distance error threshold size can't be negative." << std::endl;
		return false;
	}
	
	return true;
}

void DisplayUsage()
{
	std::cout << "Usage: " << std::endl;
	std::cout << "-i	: Input swc file path." << std::endl;
	std::cout << "-o	: Output swc file path." << std::endl;
	std::cout << "-t	: Maximum accepteple Euclidean error threshold for the error between a line linking two adjacent vertices and the path between them." << std::endl;
	std::cout << "-ws	: (Optional. Default 1 voxels, i.e., no smoothing) Sliding window size to smooth the input tree breanches independently in X, Y and Z axis." << std::endl;
	std::cout << std::endl;
}

int main(int argc, char** argv)
{
	// Declerations
	std::vector<SWCGraphVertex*> OrigVertexList;
	std::vector<SWCGraphVertex*> DownsampledVertexList;
	std::vector<SWCGraphVertex*> SmoothedVertexList;
	std::vector<SWCGraphVertex*>::iterator VertexIter;
	
	// Arguments and initializations
	std::string sInSWCFileName;
	std::string sOutSWCFileName;
	double fEuclideanDistThr;
	int nSlidingWndSizeForSmoothing = 1;
	
	// Read the parameters
	if( !ReadParameters(argc,argv, 
						sInSWCFileName, 
						sOutSWCFileName,
						fEuclideanDistThr,
						nSlidingWndSizeForSmoothing) )
    {
		DisplayUsage();
		return -1;
    }
	
	
	std::cout << "Reading the swc file ..." << std::endl;
	if( !SWCFileIO::Read(sInSWCFileName, OrigVertexList) )
	{
		std::cout << "Could not read the swc file: " << sInSWCFileName << std::endl;;
		return -1;
	}
	std::cout << "Done!" << std::endl;	

	if( nSlidingWndSizeForSmoothing > 1 )
	{
		std::cout << "Smoothing the tree with a sliding window size of " << nSlidingWndSizeForSmoothing << std::endl;
		SWCTreeSmoother::Smooth(OrigVertexList,
								SmoothedVertexList,
								nSlidingWndSizeForSmoothing);
		std::cout << "Done!" << std::endl;
	}
	else
	{
		SmoothedVertexList = OrigVertexList;
	}
	
	
	std::cout << "Downsampling the tree ..." << std::endl;
	SWCTreeDownsampler::Downsample(SmoothedVertexList,
								   DownsampledVertexList, 
								   fEuclideanDistThr);
	std::cout << "Done!" << std::endl;
	std::cout << "Original graph has " << SmoothedVertexList.size() - 1 << " vertices." << std::endl;
	std::cout << "Downsampled graph has " << DownsampledVertexList.size() - 1 << " vertices." << std::endl;

	
	std::cout << "Writing the downsampled swc file ..." << std::endl;
	if( !SWCFileIO::Write(sOutSWCFileName, DownsampledVertexList) )
	{
		std::cout << "Could not write the swc file: " << sOutSWCFileName << std::endl;;
		return -1;
	}
	std::cout << "Done!" << std::endl;	
	
	// Deallocations
	for(VertexIter = OrigVertexList.begin();
		VertexIter != OrigVertexList.end();
		VertexIter++)
	{
		if((*VertexIter) != NULL)
		{
			delete (*VertexIter);
		}
	}
	for(VertexIter = DownsampledVertexList.begin();
		VertexIter != DownsampledVertexList.end();
		VertexIter++)
	{
		if((*VertexIter) != NULL)
		{
			delete (*VertexIter);
		}
	}
	if( nSlidingWndSizeForSmoothing > 1 )
	{
		for(VertexIter = SmoothedVertexList.begin();
			VertexIter != SmoothedVertexList.end();
			VertexIter++)
		{
			if((*VertexIter) != NULL)
			{
				delete (*VertexIter);
			}
		}
	}	
	return 0;
}
