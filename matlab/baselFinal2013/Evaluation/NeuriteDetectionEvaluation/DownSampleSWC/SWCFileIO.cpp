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
#include <limits.h>

#include "SWCGraphVertex.h"

#include "SWCFileIO.h"


bool SWCFileIO::Read(std::string sFileFullPath, 
					 std::vector<SWCGraphVertex*>& rSWCVertexList)
{
	// Declerations
	std::fstream SWCFile;
	SWCGraphVertex* pBufferVertex;
	SWCGraphVertex* pBufferParentVertex;
	int nCommentChar;
	
	// Read the file
	SWCFile.open(sFileFullPath.c_str(), std::ios::in);
	
	if( SWCFile.fail() )
	{
		SWCFileReadFailed(sFileFullPath);
		return false;
	}
	
	// Add a null element to the beginning of the list, since vertex ID's start from 1.
	rSWCVertexList.push_back(NULL);
	
	while( SWCFile.good() )
	{
		// If this is no the first point, ignore the rest of the line 
		// before reading the next character.
		if( rSWCVertexList.size() != 1 )
		{
			SWCFile.ignore(INT_MAX, '\n');
		}
		
		if( !SWCFile.good() )
		{
			break;
		}
		
		nCommentChar = SWCFile.get();
		if( !SWCFile.good() )
		{
			break;
		}
		
		SWCFile.unget();
		
		if( nCommentChar == ((int)'#') )
		{
			SWCFile.ignore(INT_MAX, '\n');
		}
		
		
		pBufferVertex = new SWCGraphVertex();
		
		SWCFile >> pBufferVertex->ID;
		SWCFile >> pBufferVertex->Type;
		SWCFile >> pBufferVertex->X;
		SWCFile >> pBufferVertex->Y;
		SWCFile >> pBufferVertex->Z;
		SWCFile >> pBufferVertex->Radius;
		SWCFile >> pBufferVertex->ParentID;
		
		rSWCVertexList.push_back(pBufferVertex);
		
		if( pBufferVertex->ParentID == -1 )
		{
			pBufferVertex->pParent = NULL;
		}
		else
		{
			pBufferParentVertex = rSWCVertexList[pBufferVertex->ParentID];
			
			pBufferVertex->pParent = pBufferParentVertex;
			pBufferParentVertex->Children.push_back(pBufferVertex);
			
		}
	}

	SWCFile.close();
	if( !SWCFile.eof() )
	{
		SWCFileReadFailed(sFileFullPath);
		return false;
	}
	else
	{
		return true;
	}
}

bool SWCFileIO::Write(std::string sFileFullPath, 
					  std::vector<SWCGraphVertex*>& rSWCVertexList)
{
	// Declerations
	std::fstream SWCFile;
	SWCGraphVertex* pBufferVertex;
	std::vector<SWCGraphVertex*>::iterator SWCVertexListIter;
	
	// Read the file
	SWCFile.open(sFileFullPath.c_str(), std::ios::out);
	
	if( SWCFile.fail() )
	{
		SWCFileWriteFailed(sFileFullPath);
		return false;
	}
	
	for(SWCVertexListIter = rSWCVertexList.begin();
		SWCVertexListIter != rSWCVertexList.end();
		SWCVertexListIter++)
	{	
		pBufferVertex = (*SWCVertexListIter);
		
		if( pBufferVertex == NULL )			
		{
			continue;
		}
		
		SWCFile << pBufferVertex->ID << " ";
		SWCFile << pBufferVertex->Type << " ";
		SWCFile << pBufferVertex->X << " ";
		SWCFile << pBufferVertex->Y << " ";
		SWCFile << pBufferVertex->Z << " ";
		SWCFile << pBufferVertex->Radius << " ";
		SWCFile << pBufferVertex->ParentID;
		SWCFile << std::endl;
	}
	
	SWCFile.close();
	if( SWCFile.fail() )
	{
		SWCFileWriteFailed(sFileFullPath);
		return false;
	}
	else
	{
		return true;
	}
}

void SWCFileIO::SWCFileReadFailed(std::string sFileFullPath)
{
	std::cout << "In class SWCFileIO: Error in reading the swc file: " << sFileFullPath << std::endl;
}

void SWCFileIO::SWCFileWriteFailed(std::string sFileFullPath)
{
	std::cout << "In class SWCFileIO: Error in writing the swc file " << sFileFullPath << std::endl;
}