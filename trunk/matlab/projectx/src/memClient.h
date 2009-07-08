/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by Aurelien Lucchi and Kevin Smith                  //
// Contact aurelien.lucchi (at) gmail.com or kevin.smith (at) epfl.ch  // 
// for comments & bug reports                                          //
/////////////////////////////////////////////////////////////////////////

/*
 * shm-client - client program to demonstrate shared memory.
 */

enum eDataType{TYPE_ROW, TYPE_COLUMN};

// @param : either "row" or "column"
// TODO : overload function to have more types for dataDst
int storeWeakLearnerResponses(int index_x, int index_y, unsigned int* dataSrc, eDataType dataType, int dataSize);

// @return a pointer on the data required
int getWeakLearnerResponses(int index_x, int index_y, unsigned int* dataDst, eDataType dataType);
