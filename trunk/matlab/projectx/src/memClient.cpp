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
#include "common.h"
#include "memClient.h"

using namespace std;

int storeWeakLearnerResponses(int index_x, int index_y, unsigned int* dataSrc, eDataType dataType, int dataSize)
{
  int rc;  
  key_t shmkey;
  int shmid;
  char *shm;

  // Generate IPC keys
  // Those keys are the same as the ones used by the Deamon
  shmkey = ftok(SHMKEYPATH,SHMKEYID);
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

  printf("shm_key %d %x\n",shmkey, shmkey);

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

  printf("shm %x\n",shm);

  struct header_mem_responses* hmr = (struct header_mem_responses*) shm;
  int* dataHmr = (int*) shm + sizeof(struct header_mem_responses);

  printf("w %d h %d nbAccess %d\n", hmr->width, hmr->height, hmr->nbAccess);

  if(index_x >= hmr->width || index_y >= hmr->height)
    {
      printf("Incorrect index values. Should be smaller than (%d,%d)\n",hmr->width, hmr->height);
      return -1;
    }

  int data_index = index_y * hmr->width + index_x;
  int hmrSize;
  if(dataType == TYPE_ROW)
    hmrSize = hmr->width;
  else
    hmrSize = hmr->height;

  if(hmrSize > dataSize)
    {
      printf("The size of the source is too small\n");
      return -1;
    }

  // Store data
  for(int i=0;i<hmrSize;i++)
    {
      printf("Data %d %d\n", i, dataSrc[i]);
      dataHmr[data_index + i] = dataSrc[i];
    }

  return 0;
}

int getWeakLearnerResponses(int index_x, int index_y, unsigned int* dataDst, eDataType dataType)
{
  int rc;  
  key_t shmkey;
  int shmid;
  char *shm;

  printf("getWeakLearnerResponses\n");

  // Generate IPC keys
  // Those keys are the same as the ones used by the Deamon
  shmkey = ftok(SHMKEYPATH,SHMKEYID);
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

  printf("getWeakLearnerResponses2\n");

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

  printf("getWeakLearnerResponses3\n");

  if(index_x >= hmr->width || index_y >= hmr->height)
    {
      printf("Incorrect index values. Should be smaller than (%d,%d)\n",hmr->width, hmr->height);
      return -1;
    }

  printf("getWeakLearnerResponses4\n");

  int data_index = index_y * hmr->width + index_x;
  int data_size;
  if(dataType == TYPE_ROW)
    data_size = hmr->width;
  else
    data_size = hmr->height;

  printf("Storing data\n");

  // Store data
  for(int i=0;i<data_size;i++)
    {
      printf("Data %d %d\n", data_index + i, dataHmr[data_index + i]);
      dataDst[i] = dataHmr[data_index + i];
    }

  hmr->nbAccess++;

  return 0;
}

