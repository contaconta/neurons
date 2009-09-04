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

#include "memClient.h"

int getMemType(eMemType& type, int shm_key_id)
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
      printf("getMemSize: ftok() for shm failed\n");
      return -1;
    }

  /*
   * Retrieve the shared memory segment.
   */
  if ((shmid = shmget(shmkey, 0, 0666)) < 0) {
    printf("getMemSize: shmget() initialization failed\n");
    return -1;
  }

  /*
   * Now we attach the segment to our data space.
   */
  if ((shm = (char*) shmat(shmid, NULL, 0)) == (char*) -1) {
    printf("getMemSize: shmat() initialization failed\n");
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
  int* dataHmr = (int*) shm + (sizeof(struct header_mem_responses)/sizeof(int));

#ifdef DEBUG_M
  printf("shm %x\n",shm);
  printf("%d\n", hmr->type);
#endif

  type = hmr->type;

  if(shm != 0)
    {
      // Detach from shared memory segment
      rc = shmdt((const void *) shm);
      if (rc != 0) {
        printf("Unable to detach from shared memory segment (rc=%d)\n", rc);
        return -1;
      }
    }

  return type;
}

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
      printf("getMemSize: ftok() for shm failed\n");
      return -1;
    }

  /*
   * Retrieve the shared memory segment.
   */
  if ((shmid = shmget(shmkey, 0, 0666)) < 0) {
    printf("getMemSize: shmget() initialization failed\n");
    return -1;
  }

  /*
   * Now we attach the segment to our data space.
   */
  if ((shm = (char*) shmat(shmid, NULL, 0)) == (char*) -1) {
    printf("getMemSize: shmat() initialization failed\n");
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
  int* dataHmr = (int*) shm + (sizeof(struct header_mem_responses)/sizeof(int));

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
