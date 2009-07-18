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

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "memClient.h"

using namespace std;

#define DEBUG_M

int getMemSize(int &width, int &height, int shm_key_id)
{
  int rc;  
  key_t shmkey;
  int shmid;
  char *shm;

  // Generate IPC keys
  // Those keys are the same as the ones used by the Deamon
  shmkey = ftok(SHMKEYPATH,SHMKEYID); //shm_key_id);
  if ( shmkey == (key_t)-1 )
    {
      printf("main: ftok() for shm failed\n");
      return -1;
    }

  /*
   * Retrieve the shared memory segment.
   */
  if ((shmid = shmget(shmkey, 0, 0666)) < 0) {
    printf("main: shmget() initialization failed\n");
    return -1;
  }

  /*
   * Now we attach the segment to our data space.
   */
  if ((shm = (char*) shmat(shmid, NULL, 0)) == (char*) -1) {
    printf("main: shmat() initialization failed\n");
    return -1;
  }

#ifdef DEBUG_M
  printf("shm_key %d %x shm_id %d\n",shmkey, shmkey, shmid);
#endif

  /*
    key_t semkey;
  semkey = ftok(SEMKEYPATH,SEMKEYID);
  if ( semkey == (key_t)-1 )
    {
      printf("main: ftok() for sem failed\n");
      return -1;
    }
  // Retrieve semaphore id
  semid = semget(semkey, NUMSEMS, 0666);
  if ( semid == -1 )
    {
      printf("main: semget() failed\n");
      return -1;
    }
  */

  struct header_mem_responses* hmr = (struct header_mem_responses*) shm;
  int* dataHmr = (int*) shm + sizeof(struct header_mem_responses);

#ifdef DEBUG_M
  printf("shm %x\n",shm);
  printf("w %d h %d\n", hmr->width, hmr->height);
#endif

  width = hmr->width;
  height = hmr->height;

  if(shm != 0)
    {
      // Detach from shared memory segment
      rc = shmdt((const void *) shm);
      if (rc != 0) {
        printf("Unable to detach from shared memory segment (rc=%d)\n", rc);
        return -1;
      }
    }

  return 0;
}

int storeWeakLearnerResponses(unsigned int* dataSrc, eDataFormat dataFormat,
                              eDataType dataType, int index, int dataSize, int shm_key_id)
{
  int rc;  
  key_t shmkey;
  int shmid;
  char *shm;
  int index_x, index_y;

  // Generate IPC keys
  // Those keys are the same as the ones used by the Deamon
  shmkey = ftok(SHMKEYPATH,SHMKEYID); //shm_key_id);
  if ( shmkey == (key_t)-1 )
    {
      printf("main: ftok() for shm failed\n");
      return -1;
    }

  /*
   * Retrieve the shared memory segment.
   */
  if ((shmid = shmget(shmkey, 0, 0666)) < 0) {
    printf("main: shmget() initialization failed\n");
    return -1;
  }

  /*
   * Now we attach the segment to our data space.
   */
  if ((shm = (char*) shmat(shmid, NULL, 0)) == (char*) -1) {
    printf("main: shmat() initialization failed\n");
    return -1;
  }

#ifdef DEBUG_M
  printf("shm_key %d %x\n",shmkey, shmkey);
#endif

  /*
    key_t semkey;
  semkey = ftok(SEMKEYPATH,SEMKEYID);
  if ( semkey == (key_t)-1 )
    {
      printf("main: ftok() for sem failed\n");
      return -1;
    }
  // Retrieve semaphore id
  semid = semget(semkey, NUMSEMS, 0666);
  if ( semid == -1 )
    {
      printf("main: semget() failed\n");
      return -1;
    }
  */

  struct header_mem_responses* hmr = (struct header_mem_responses*) shm;
  int* dataHmr = (int*) shm + sizeof(struct header_mem_responses);

#ifdef DEBUG_M
  printf("shm %x\n",shm);
  printf("w %d h %d\n", hmr->width, hmr->height);
#endif

  // Search for the required type in memory
  bool typeFound = false;
  int iLearner;
  for(iLearner=0;iLearner<MAX_NUM_WEAK_LEANER_TYPE;iLearner++)
    {
      if((int)hmr->wlp[iLearner].dataType == 0)
        {
          break;
        }
      else if(hmr->wlp[iLearner].dataType == dataType)
        {
          typeFound = true;
          break;
        }
    }

  // If type was not found, create a new one in memory if we have enough space
  // If we don't have enough space in memory, the data will be recorded to a file
  if(!typeFound)
    {
      if(iLearner<MAX_NUM_WEAK_LEANER_TYPE-1)
        {
#ifdef DEBUG_M
          printf("Creating data type %d at index %d pointing to %d\n",dataType,iLearner,hmr->w_index);
#endif
          // Create new type in memory
          hmr->wlp[iLearner].dataType = dataType;
          hmr->wlp[iLearner].index = hmr->w_index;
        }
      else
        {
          string filename(sDataType[dataType - 1]);
#ifdef DEBUG_M
          printf("Writing data to %s\n",filename.c_str());
#endif
          ofstream ofs(filename.c_str(), ios::out);
          ofs.write((char*)dataSrc,dataSize*sizeof(int));
          ofs.close();
        }
    }

#ifdef DEBUG_M
          printf("Index found : %d\n",iLearner);
#endif

  int hmrSize;
  int iStep;
  if(dataFormat == FORMAT_ROW)
    {
      index_y = hmr->wlp[iLearner].index + index;
      index_x = 0;
      hmrSize = hmr->width;
      iStep = 1;
    }
  else
    {
      index_x = hmr->wlp[iLearner].index + index;
      index_y = 0;
      hmrSize = hmr->height;
      iStep = hmr->width;
    }

#ifdef DEBUG_M
  printf("Indices %d %d %d\n", iStep, index_x, index_y);
#endif

  if(hmrSize > dataSize)
    {
      printf("The size of the source (=%d) is smaller than the size of the area in shared memory (=%d)\n",dataSize,hmrSize);
    }
  else
    {
      // Store data
      int data_index = index_y * hmr->width + index_x;
      for(int i=0;i<hmrSize;i+=iStep)
        {
          //printf("Data %d %d\n", data_index + i, dataSrc[i]);
          dataHmr[data_index + i] = dataSrc[i];
        }
    }

  if(shm != 0)
    {
      // Detach from shared memory segment
      rc = shmdt((const void *) shm);
      if (rc != 0) {
        printf("Unable to detach from shared memory segment (rc=%d)\n", rc);
        return -1;
      }
    }

  return 0;
}

int getWeakLearnerResponses(double* dataDst, eDataFormat dataFormat,
                            eDataType dataType, int index, int shm_key_id)
{
  int rc;  
  key_t shmkey;
  int shmid;
  char *shm;
  int index_x, index_y;

  // Generate IPC keys
  // Those keys are the same as the ones used by the Deamon
  shmkey = ftok(SHMKEYPATH,SHMKEYID); //shm_key_id);
  if ( shmkey == (key_t)-1 )
    {
      printf("main: ftok() for shm failed\n");
      return -1;
    }

  /*
   * Retrieve the shared memory segment.
   */
  if ((shmid = shmget(shmkey, 0, 0666)) < 0) {
    printf("main: shmget() initialization failed\n");
    return -1;
  }

  /*
   * Now we attach the segment to our data space.
   */
  if ((shm = (char*) shmat(shmid, NULL, 0)) == (char*) -1) {
    printf("main: shmat() initialization failed\n");
    return -1;
  }

  /*
    key_t semkey;
  semkey = ftok(SEMKEYPATH,SEMKEYID);
  if ( semkey == (key_t)-1 )
    {
      printf("main: ftok() for sem failed\n");
      return -1;
    }
  // Retrieve semaphore id
  semid = semget(semkey, NUMSEMS, 0666);
  if ( semid == -1 )
    {
      printf("main: semget() failed\n");
      return -1;
    }
  */

  struct header_mem_responses* hmr = (struct header_mem_responses*) shm;
  // TODO : Probably need a template for dataHmr
  int* dataHmr = (int*) shm + sizeof(struct header_mem_responses);

  bool typeFound = false;
  int iLearner;
  for(iLearner=0;iLearner<MAX_NUM_WEAK_LEANER_TYPE;iLearner++)
    {
      if((int)hmr->wlp[iLearner].dataType == 0)
        {
          break;
        }
      else if(hmr->wlp[iLearner].dataType == dataType)
        {
          typeFound= true;
          break;
        }
    }

#ifdef DEBUG_M
          printf("Index found : %d\n",iLearner);
#endif

  if(!typeFound)
    {
      // Load data from of a file
      string filename(sDataType[dataType - 1]);
#ifdef DEBUG_M
      printf("Loading data from %s\n",filename.c_str());
#endif
      ofstream ifs(filename.c_str(), ios::in);
      if(ifs.fail())
        {
          printf("Failed to load data from %s\n",filename.c_str());


          printf("Detaching from shared memory segment\n");
          if(shm != 0)
            {
              // Detach from shared memory segment
              rc = shmdt((const void *) shm);
              if (rc != 0) {
                printf("Unable to detach from shared memory segment (rc=%d)\n", rc);
                return -1;
              }
            }

          return -1;
        }
      //ifs.read((char*)dataDst,dataSize*sizeof(int));
      ifs.close();
    }

  int hmrSize;
  int iStep;
  if(dataFormat == FORMAT_ROW)
    {
      index_y = hmr->wlp[iLearner].index + index;
      index_x = 0;
      hmrSize = hmr->width;
      iStep = 1;
    }
  else
    {
      index_x = hmr->wlp[iLearner].index + index;
      index_y = 0;
      hmrSize = hmr->height;
      iStep = hmr->width;
    }

#ifdef DEBUG_M
  printf("Indices %d %d %d\n", iStep, index_x, index_y);
#endif

  // Copy data to the destination buffer
  int data_index = index_y * hmr->width + index_x;
  for(int i=0;i<hmrSize;i+=iStep)
    {
      //printf("Data %d %d\n", data_index + i, dataHmr[data_index + i]);
      dataDst[i] = (double)dataHmr[data_index + i];
    }

  if(shm != 0)
    {
      // Detach from shared memory segment
      rc = shmdt((const void *) shm);
      if (rc != 0) {
        printf("Unable to detach from shared memory segment (rc=%d)\n", rc);
        return -1;
      }
    }

  return 0;
}

