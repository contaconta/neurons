
#ifndef COMMON_H
#define COMMON_H

#define DIR_DATA "data/"

#define SEMKEYPATH "/dev/null"  // Path used on ftok for semget key
#define SEMKEYID 1              // Id used on ftok for semget key
#define SHMKEYPATH "/dev/null"  // Path used on ftok for shmget key
#define SHMKEYID 2              // Id used on ftok for shmget key

#define NUMSEMS 1               // Num of sems in created sem set

#define SHMSZ 2147483648 // Size of the shared mem segment
// Change value of 'shmmax' to 1gb
// cat 1073741824 > /proc/sys/kernel/shmmax
// sudo sysctl -w kernel.shmmax=1073741824
// sudo emacs /etc/sysctl.conf

#define MAX_NUM_WEAK_LEANER_TYPE 40 // Haar, HOG, ray...

enum eDataType{
  TYPE_HAAR=1,
  TYPE_RAY
};

extern char sDataType[][5];

struct weak_learner_param
{
  eDataType dataType;
  int index; // first index in memory
};

struct header_mem_responses
{
  int width;
  int height;
  struct weak_learner_param wlp[MAX_NUM_WEAK_LEANER_TYPE];  
  //int wl_indices[MAX_NUM_WEAK_LEANER_TYPE];
  int w_index; // index pointing to the place to write in memory
};

// unique id in memory that identifies the end of the metadata
#define END_METADATA_WEAK_LEARNER = 0xFFFFFFFE


#endif //COMMON_H
