#include "common.h"

#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>

using namespace std;

static char *shm = 0;
int shmid = -1;
int semid = -1;

string getExtension(string path){
  return path.substr(path.find_last_of(".")+1);
}

// List files in directory
int getdir (string dir, vector<string> &files)
{
  DIR *dp;
  struct dirent *dirp;
  string path;
  if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL) {
    path = string(dirp->d_name);
    cout << getExtension(path) << endl;
    if(getExtension(path)=="dat")
      files.push_back(dir + path);
  }
  closedir(dp);
  return 0;
}


// Load data from files
int loadData()
{
  int total_size = 0;
  vector<string> files;
  getdir(DIR_DATA,files);
  cout << "Size: " << files.size() << endl;
  for(vector<string>::iterator it = files.begin();
      it != files.end(); it++)
    {
      // get file size
      struct stat filestatus;
      stat(it->c_str(), &filestatus);
      cout << filestatus.st_size << " bytes\n";

      if((filestatus.st_size + total_size) < SHMSZ)
        {
          total_size += filestatus.st_size;
        }
    }
}

/* Clean up the environment by removing the semid structure,
 * detaching the shared memory segment, and then performing
 * the delete on the shared memory segment ID.
 */
void exit_program(int sig)
{
  int rc;

  printf("Catched signal: %d !!\n", sig);

#ifdef USE_SEM
  rc = semctl(semid, 1, IPC_RMID);
  if (rc==-1)
    {
      printf("main: semctl() remove id failed\n");
    }
#endif

  rc = shmctl(shmid, IPC_RMID, 0); //&shmid_struct);
  if (rc==-1)
    {
      printf("main: shmctl() failed\n");
    }

  if(shm != 0)
    {
      // Detach from shared memory segment
      rc = shmdt((const void *) shm);
      if (rc != 0) {
        printf("Unable to detach from shared memory segment (rc=%d)\n", rc);
      }
    }
  printf("Exit\n");
  exit(0);
}

#ifdef USE_SEM
int init_sem()
{
  key_t semkey;
  struct sembuf operations[1];

  /* Generate an IPC key for the semaphore set
   * Typically, an application specific path and
   * id would be used to generate the IPC key.
   */
  semkey = ftok(SEMKEYPATH,SEMKEYID);
  if ( semkey == (key_t)-1 )
    {
      printf("main: ftok() for sem failed\n");
      return -1;
    }

  /* Create a semaphore set using the IPC key.  The number of
   * semaphores in the set is two.  If a semaphore set already
   * exists for the key, return an error. The specified permissions
   * give everyone read/write access to the semaphore set.
   */
  semid = semget( semkey, NUMSEMS, 0666 | IPC_CREAT | IPC_EXCL );
  if ( semid == -1 )
    {
      printf("main: semget() failed\n");
      return -1;
    }

  /* Initialize the first semaphore in the set to 0
   * The first semaphore in the sem set means:
   *        '1' --  The shared memory segment is being used
   *        '0' --  The shared memory segment is freed.
   */
  short  sarray[1];
  sarray[0] = 1;

  /* The '1' on this command is a no-op, because the SETALL command
   * is used.
   */
  rc = semctl( semid, 1, SETALL, sarray);
  if(rc == -1)
    {
      printf("main: semctl() initialization failed\n");
      exit_program(-1);
      return -1;
    }
}
#endif

int init(int width, int height, char* data_type)
{
  int rc;
  key_t shmkey;
  
  signal(SIGINT, exit_program);
  signal(SIGTERM, exit_program);

  int data_unit_size = 0;
  if(strcmp(data_type,"char")==0)
    data_unit_size = sizeof(char);
  else if(strcmp(data_type,"int")==0)
    data_unit_size = sizeof(int);
  else if(strcmp(data_type,"double")==0)
    data_unit_size = sizeof(double);
  else
    {
      printf("Unknown type : %s\n", data_type);
      return -1;
    }

  unsigned int memory_size = sizeof(struct header_mem_responses) + data_unit_size*width*height;
  struct header_mem_responses hmr;
  memset(&hmr,0,sizeof(hmr));
  hmr.width = width;
  hmr.height = height;

  printf("Server is initialising\n");
  printf("Memory required : %u\n", memory_size);

  // TODO : Check if this is less than max size available

  /* Generate an IPC key for the shared memory segment
   * Typically, an application specific path and
   * id would be used to generate the IPC key.
   */
  shmkey = ftok(SHMKEYPATH,SHMKEYID);
  if ( shmkey == (key_t)-1 )
    {
      printf("main: ftok() for shm failed\n");
      return -1;
    }

  printf("shm_key %d %x\n",shmkey, shmkey);

  /*
   * Create the shared memory segment.
   */
  if (shmid = shmget(shmkey, memory_size, IPC_CREAT | 0666) == -1) {
    printf("main: shmget() initialization failed\n");
    exit_program(-1);
    return -1;
  }

  /*
   * Now we attach the segment to our data space.
   */
  if ((shm = (char*) shmat(shmid, NULL, 0)) == (char*) -1) {
    printf("main: shmat() initialization failed\n");
    exit_program(-1);
    return -1;
  }

  *((struct header_mem_responses *)shm) = hmr;

  printf("shm %x\n",shm);

  /*
  // Wait for the semaphore to be available
  operations[0].sem_num = 0;
  operations[0].sem_op = -1; // Decrement the semval by one
  operations[0].sem_flg = 0; // Allow a wait to occur

  rc = semop(semid, operations, 1);
  if (rc == -1)
    {
      printf("main: semop(-1) failed\n");
      exit_program(-1);
      return -1;
    }

  loadData();

  // Signal the first semaphore to free the shared memory.
  operations[0].sem_num = 0;
  operations[0].sem_op  = 1;
  operations[0].sem_flg = IPC_NOWAIT;

  rc = semop( semid, operations, 1 );
  if (rc == -1)
    {
      printf("main: semop(1) failed\n");
      exit_program(-1);
      return -1;
    }

  */


  printf("File has been loaded. Server is ready\n");

  /*
   * Finally, we wait until the other process 
   * changes the first character of our memory
   * to '*', indicating that it has read what 
   * we put there.
   */
  while (*shm != '*')
    sleep(1);

  return 0;
}

int main(int argc, char* argv[])
{
  if(argc<4)
    {
      printf("Usage : width height data_type\n");
      return -1;
    }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  char* data_type = argv[3];

  return init(width, height, data_type);
}
