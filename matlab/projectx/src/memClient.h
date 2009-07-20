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
 * shm-client - client program to manage shared memory.
 */

#include "common.h"

enum eDataFormat{FORMAT_ROW, FORMAT_COLUMN};

// get memory size
int getMemSize(int &width, int &height, int shm_key_id = SHMKEYID);

// Store the weak learner responses in the shared memory
// The responses are stored as 32 bit integers
// @param : either "row" or "column"
// TODO : overload function to have more types for dataDst
int storeWeakLearnerResponses(int* dataSrc, eDataFormat dataFormat, eDataType dataType, int index, int dataSize, int shm_key_id = SHMKEYID);

// @return a pointer on the data required
int getWeakLearnerResponses(int* dataDst, eDataFormat dataFormat, eDataType dataType, int index, int shm_key_id = SHMKEYID);
