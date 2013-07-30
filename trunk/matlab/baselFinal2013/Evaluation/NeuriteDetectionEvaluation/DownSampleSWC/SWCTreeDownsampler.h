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

#ifndef SWC_TREE_DOWNSAMPLER_H
#define SWC_TREE_DOWNSAMPLER_H

#include<vector>
#include<string.h>

class SWCGraphVertex;

class SWCTreeDownsampler
{
	
public:
	
	static void Downsample(std::vector<SWCGraphVertex*>& rOrigVertexList,
						   std::vector<SWCGraphVertex*>& rDownVertexList,
						   double fEuclideanDistThr);
	
	
private:
	
	static void DownsampleRecursively(std::vector<SWCGraphVertex*>& rOrigVertexList,
									  std::vector<SWCGraphVertex*>& rDownVertexList, 
									  SWCGraphVertex* pParentOrigVertex,
									  SWCGraphVertex* pChildOrigVertex,
									  SWCGraphVertex* pParentDownVertex,
									  double fEuclideanDistThr);
	
	static SWCGraphVertex* FindNextOptimalVertex(std::vector<SWCGraphVertex*>& rOrigVertexList,
												 SWCGraphVertex* pParentVertex,
												 SWCGraphVertex* pChildVertex,
												 double fEuclideanDistThr);
	
		
};

#endif
