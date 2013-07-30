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

#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <float.h>

#include "SWCGraphVertex.h"

#include "SWCTreeDownsampler.h"
	
void SWCTreeDownsampler::Downsample(std::vector<SWCGraphVertex*>& rOrigVertexList,
									std::vector<SWCGraphVertex*>& rDownVertexList, 
									double fEuclideanDistThr)
{
	// Declerations
	SWCGraphVertex* pOrigRootVertex;
	SWCGraphVertex* pDownRootVertex;
	std::vector<SWCGraphVertex*>::iterator VertexIter;
	
	// Find the tree root and start growing from it.
	for(VertexIter = rOrigVertexList.begin();
		VertexIter != rOrigVertexList.end();
		VertexIter++)
	{
		if( (*VertexIter) == NULL )
		{
			continue;
		}
		
		pOrigRootVertex = (*VertexIter);
		
		if( pOrigRootVertex->ParentID < 0 )
		{
			break;
		}
	}
	
	// Adding the first NULL vertex to the list.
	rDownVertexList.push_back(NULL);
	
	// Adding the root vertex of the downsampled tree.
	pDownRootVertex = new SWCGraphVertex();
	pDownRootVertex->ID = 1;
	pDownRootVertex->Type = pOrigRootVertex->Type;
	pDownRootVertex->X = pOrigRootVertex->X;
	pDownRootVertex->Y = pOrigRootVertex->Y;
	pDownRootVertex->Z = pOrigRootVertex->Z;
	pDownRootVertex->Radius = pOrigRootVertex->Radius;
	pDownRootVertex->ParentID = -1;
	pDownRootVertex->pParent = NULL;
	rDownVertexList.push_back(pDownRootVertex);
	
	// Recursively downsample edges starting from the root node.
	for(VertexIter = pOrigRootVertex->Children.begin();
		VertexIter != pOrigRootVertex->Children.end();
		VertexIter++)
	{
		DownsampleRecursively(rOrigVertexList, rDownVertexList, 
							  pOrigRootVertex, (*VertexIter),
							  pDownRootVertex, fEuclideanDistThr);
	}
}

void SWCTreeDownsampler::DownsampleRecursively(std::vector<SWCGraphVertex*>& rOrigVertexList,
											   std::vector<SWCGraphVertex*>& rDownVertexList, 
											   SWCGraphVertex* pParentOrigVertex,
											   SWCGraphVertex* pChildOrigVertex,
											   SWCGraphVertex* pParentDownVertex,
											   double fEuclideanDistThr)
{
	// Declerations.
	SWCGraphVertex* pOptimOrigVertex;
	SWCGraphVertex* pChildDownVertex;
	std::vector<SWCGraphVertex*>::iterator VertexIter;
	
	// Continue on downsampling if this is not a bifurcation or an end point.
	do
	{
		pOptimOrigVertex = FindNextOptimalVertex(rOrigVertexList,
												 pParentOrigVertex,
												 pChildOrigVertex,
												 fEuclideanDistThr);
		
		// Create an edge between the sampled parent and the child vertices. 
		pChildDownVertex = new SWCGraphVertex();
		
		pChildDownVertex->ID = rDownVertexList.size();
		pChildDownVertex->Type = pOptimOrigVertex->Type;
		pChildDownVertex->X = pOptimOrigVertex->X;
		pChildDownVertex->Y = pOptimOrigVertex->Y;
		pChildDownVertex->Z = pOptimOrigVertex->Z;
		pChildDownVertex->Radius = pOptimOrigVertex->Radius;
		pChildDownVertex->ParentID = pParentDownVertex->ID;
		pChildDownVertex->pParent = pParentDownVertex;
		pParentDownVertex->Children.push_back(pChildDownVertex);
		
		rDownVertexList.push_back(pChildDownVertex);
		
		pParentOrigVertex = pOptimOrigVertex;
		pParentDownVertex = pChildDownVertex;
		
		if(pParentOrigVertex->Children.size() == 1)
		{
			pChildOrigVertex = pParentOrigVertex->Children.front();
		}
		
	}while( pParentOrigVertex->Children.size() == 1 );
	
	// Stop if this is an end point. If it is a bifurcation,
	// call the same function for each children of this node.
	if( pParentOrigVertex->Children.size() > 1 )
	{
		for(VertexIter = pParentOrigVertex->Children.begin();
			VertexIter != pParentOrigVertex->Children.end();
			VertexIter++)
		{
			DownsampleRecursively(rOrigVertexList, rDownVertexList, 
								  pParentOrigVertex, (*VertexIter),
								  pParentDownVertex, fEuclideanDistThr);
		}
	}
}

SWCGraphVertex* SWCTreeDownsampler::FindNextOptimalVertex(std::vector<SWCGraphVertex*>& rOrigVertexList,
														  SWCGraphVertex* pParentVertex,
														  SWCGraphVertex* pChildVertex,
														  double fEuclideanDistThr)
{
	// Declerations.
	SWCGraphVertex* pCurrentVertex;
	SWCGraphVertex* pBufferVertex;
	double fParentToBufferX;
	double fParentToBufferY;
	double fParentToBufferZ;
	double fParentToCurrentX;
	double fParentToCurrentY;
	double fParentToCurrentZ;
	double fDotProd;
	double fBufferDist;
	double fError;
	double fErrorSum;
	
	pCurrentVertex = pChildVertex;
	
	// Continue on downsampling if this is not a bifurcation or an end point.
	do
	{
		pBufferVertex = pCurrentVertex->pParent;
		
		fParentToCurrentX = pCurrentVertex->X - pParentVertex->X;
		fParentToCurrentY = pCurrentVertex->Y - pParentVertex->Y;
		fParentToCurrentZ = pCurrentVertex->Z - pParentVertex->Z;
		
		// Compute error from the current vertex to parent vertex
		fErrorSum = 0;
		while((pBufferVertex != pParentVertex) && 
			  !((fabs(fParentToCurrentX) < DBL_EPSILON) &&
				(fabs(fParentToCurrentY) < DBL_EPSILON) &&
				(fabs(fParentToCurrentZ) < DBL_EPSILON)))
		{
			fParentToBufferX = pBufferVertex->X - pParentVertex->X;
			fParentToBufferY = pBufferVertex->Y - pParentVertex->Y;
			fParentToBufferZ = pBufferVertex->Z - pParentVertex->Z;
			
			fDotProd = fParentToCurrentX * fParentToBufferX +
						fParentToCurrentY * fParentToBufferY +
						fParentToCurrentZ * fParentToBufferZ;
			fDotProd /= sqrt(fParentToCurrentX * fParentToCurrentX + 
							fParentToCurrentY * fParentToCurrentY +
							fParentToCurrentZ * fParentToCurrentZ);
			
			fBufferDist = sqrt(fParentToBufferX * fParentToBufferX + 
							   fParentToBufferY * fParentToBufferY + 
							   fParentToBufferZ * fParentToBufferZ);
			
			fError = sqrt((fBufferDist * fBufferDist) - 
						  (fDotProd * fDotProd));
			
			fErrorSum += fError;
			
			pBufferVertex = pBufferVertex->pParent;
		}
		
		if((pCurrentVertex->Children.size() == 1) && 
		   (fErrorSum < fEuclideanDistThr) )
		{
			pCurrentVertex = pCurrentVertex->Children.front();
		}
		
	}while((pCurrentVertex->Children.size() == 1) && 
		   (fErrorSum < fEuclideanDistThr) );
	
	// Return the parent of the current vertex if the 
	// error (with the current vertex) is greater than 
	// the threshold. Otherwise (if the current vertex is a 
	// bifurcation or an end point), return the current vertex.
	if( pCurrentVertex->Children.size() != 1 )
	{
		return pCurrentVertex;
	}
	else
	{
		return pCurrentVertex->pParent;
	}
}
