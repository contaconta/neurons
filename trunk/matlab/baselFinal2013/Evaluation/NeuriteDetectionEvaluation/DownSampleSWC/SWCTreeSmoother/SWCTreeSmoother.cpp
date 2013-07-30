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
 * @version 1.0 26/08/2010
 * @author Engin Turetken <engin.turetken@epfl.ch>
 */

#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <float.h>

#include "SWCGraphVertex.h"

#include "SWCTreeSmoother.h"
	
void SWCTreeSmoother::Smooth(std::vector<SWCGraphVertex*>& rOrigVertexList,
							 std::vector<SWCGraphVertex*>& rSmoothVertexList, 
							 int nSlidingWndSizeForSmoothing)
{
	// Declerations
	SWCGraphVertex* pOrigRootVertex;
	SWCGraphVertex* pSmoothRootVertex;
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
	rSmoothVertexList.push_back(NULL);
	
	// Adding the root vertex of the smooth tree.
	pSmoothRootVertex = new SWCGraphVertex();
	pSmoothRootVertex->ID = 1;
	pSmoothRootVertex->Type = pOrigRootVertex->Type;
	pSmoothRootVertex->X = pOrigRootVertex->X;
	pSmoothRootVertex->Y = pOrigRootVertex->Y;
	pSmoothRootVertex->Z = pOrigRootVertex->Z;
	pSmoothRootVertex->Radius = pOrigRootVertex->Radius;
	pSmoothRootVertex->ParentID = -1;
	pSmoothRootVertex->pParent = NULL;
	rSmoothVertexList.push_back(pSmoothRootVertex);
	
	// Recursively smooth starting from the root node.
	for(VertexIter = pOrigRootVertex->Children.begin();
		VertexIter != pOrigRootVertex->Children.end();
		VertexIter++)
	{
		SmoothRecursively(rOrigVertexList, rSmoothVertexList, 
						  pOrigRootVertex, (*VertexIter),
						  pSmoothRootVertex, nSlidingWndSizeForSmoothing);
	}
}

void SWCTreeSmoother::SmoothRecursively(std::vector<SWCGraphVertex*>& rOrigVertexList,
										std::vector<SWCGraphVertex*>& rSmoothVertexList, 
										SWCGraphVertex* pParentOrigVertex,
										SWCGraphVertex* pChildOrigVertex,
										SWCGraphVertex* pParentSmoothVertex,
										int nSlidingWndSizeForSmoothing)
{
	// Declerations.
	SWCGraphVertex* pChildSmoothVertex;
	std::vector<SWCGraphVertex*>::iterator VertexIter;
	
	// Continue on downsampling if this is not a bifurcation or an end point.
	do
	{
		pChildSmoothVertex = AddInterpolatedVertex(rOrigVertexList,
												   rSmoothVertexList,
												   pChildOrigVertex,
												   pParentSmoothVertex,
												   nSlidingWndSizeForSmoothing);
		
		pParentOrigVertex = pChildOrigVertex;
		pParentSmoothVertex = pChildSmoothVertex;
		
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
			SmoothRecursively(rOrigVertexList, rSmoothVertexList, 
							  pParentOrigVertex, (*VertexIter),
							  pParentSmoothVertex, 
							  nSlidingWndSizeForSmoothing);
		}
	}
}


SWCGraphVertex* SWCTreeSmoother::AddInterpolatedVertex(std::vector<SWCGraphVertex*>& rOrigVertexList,
													   std::vector<SWCGraphVertex*>& rSmoothVertexList,
													   SWCGraphVertex* pChildOrigVertex,
													   SWCGraphVertex* pParentSmoothVertex,
													   int nSlidingWndSizeForSmoothing)
{
	// Declerations and interpolations
	SWCGraphVertex* pChildSmoothVertex;
	double fLeftInterpX = 0;
	double fLeftInterpY = 0;
	double fLeftInterpZ = 0;
	double fRightInterpX = 0;
	double fRightInterpY = 0;
	double fRightInterpZ = 0;
	double fInterpX;
	double fInterpY;
	double fInterpZ;
	int nSlidingWndHalfSize = (nSlidingWndSizeForSmoothing - 1) / 2;
	int nEffectiveHalfWndSize;
	int nLeftVertexCntr = 0;
	int nRightVertexCntr = 0;
	SWCGraphVertex* pBufferChild;
	SWCGraphVertex* pBufferParent;

	// Check whether or not the given original child 
	// vertex is a bifurcation. If so, do not smooth it.
	if(pChildOrigVertex->Children.size() <= 1)
	{
		// Traverse in the direction of the parent.
		pBufferParent = pChildOrigVertex->pParent;
		while((pBufferParent != NULL) && 
			  (nLeftVertexCntr < nSlidingWndHalfSize) )
		{
			fLeftInterpX += pBufferParent->X;
			fLeftInterpY += pBufferParent->Y;
			fLeftInterpZ += pBufferParent->Z;
			
			nLeftVertexCntr++;
			
			// Stop here if this is a bifurcation point
			if( pBufferParent->Children.size() > 1 )
			{	
				break;
			}
			
			pBufferParent = pBufferParent->pParent;
		}
		if( nLeftVertexCntr != 0 )
		{
			fLeftInterpX /= ((double)nLeftVertexCntr);
			fLeftInterpY /= ((double)nLeftVertexCntr);
			fLeftInterpZ /= ((double)nLeftVertexCntr);
		}
		
		// Traverse in the direction of the child if the current 
		// vertex is not a terminal vertex.
		pBufferChild = pChildOrigVertex;
		while((pBufferChild->Children.size() == 1) &&
			  (nRightVertexCntr < nSlidingWndHalfSize))
		{
			pBufferChild = pChildOrigVertex->Children.front();
			
			fRightInterpX += pBufferChild->X;
			fRightInterpY += pBufferChild->Y;
			fRightInterpZ += pBufferChild->Z;
			
			nRightVertexCntr++;	
		}
		if( nRightVertexCntr != 0 )
		{
			fRightInterpX /= ((double)nRightVertexCntr);
			fRightInterpY /= ((double)nRightVertexCntr);
			fRightInterpZ /= ((double)nRightVertexCntr);
		}
	}
	
	nEffectiveHalfWndSize = std::min(nLeftVertexCntr, nRightVertexCntr);
	
	// Find an approximation to the average coordinates values in the 
	// effective window.
	fInterpX = (fRightInterpX * nEffectiveHalfWndSize + 
				fLeftInterpX * nEffectiveHalfWndSize + 
				pChildOrigVertex->X) / (2 * nEffectiveHalfWndSize + 1);
	
	fInterpY = (fRightInterpY * nEffectiveHalfWndSize + 
				fLeftInterpY * nEffectiveHalfWndSize + 
				pChildOrigVertex->Y) / (2 * nEffectiveHalfWndSize + 1);
	
	fInterpZ = (fRightInterpZ * nEffectiveHalfWndSize + 
				fLeftInterpZ * nEffectiveHalfWndSize + 
				pChildOrigVertex->Z) / (2 * nEffectiveHalfWndSize + 1);
	
	
	// Create an edge between the smoothed parent and the child vertices.
	pChildSmoothVertex = new SWCGraphVertex();

	pChildSmoothVertex->ID = rSmoothVertexList.size();


	pChildSmoothVertex->Type = pChildOrigVertex->Type;
	pChildSmoothVertex->X = fInterpX;
	pChildSmoothVertex->Y = fInterpY;
	pChildSmoothVertex->Z = fInterpZ;
	pChildSmoothVertex->Radius = pChildOrigVertex->Radius;
	pChildSmoothVertex->ParentID = pParentSmoothVertex->ID;
	pChildSmoothVertex->pParent = pParentSmoothVertex;
	pParentSmoothVertex->Children.push_back(pChildSmoothVertex);

	rSmoothVertexList.push_back(pChildSmoothVertex);
	
	return pChildSmoothVertex;
}
