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

#ifndef SWC_TREE_SMOOTHER_H
#define SWC_TREE_SMOOTHER_H

#include<vector>

class SWCGraphVertex;

class SWCTreeSmoother
{
	
public:
	
	static void Smooth(std::vector<SWCGraphVertex*>& rOrigVertexList,
					   std::vector<SWCGraphVertex*>& rSmoothVertexList,
					   int nSlidingWndSizeForSmoothing);
	
	
private:
	
	static void SmoothRecursively(std::vector<SWCGraphVertex*>& rOrigVertexList,
								  std::vector<SWCGraphVertex*>& rSmoothVertexList, 
								  SWCGraphVertex* pParentOrigVertex,
								  SWCGraphVertex* pChildOrigVertex,
								  SWCGraphVertex* pParentSmoothVertex,
								  int nSlidingWndSizeForSmoothing);
	
	static SWCGraphVertex* AddInterpolatedVertex(std::vector<SWCGraphVertex*>& rOrigVertexList,
												 std::vector<SWCGraphVertex*>& rSmoothVertexList,
												 SWCGraphVertex* pChildOrigVertex,
												 SWCGraphVertex* pParentSmoothVertex,
												 int nSlidingWndSizeForSmoothing);
		
};

#endif