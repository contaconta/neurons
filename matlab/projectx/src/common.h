
#ifndef COMMON_H
#define COMMON_H

#define DIR_DATA "data/"

#define SEMKEYPATH "/dev/null"  // Path used on ftok for semget key
#define SEMKEYID 1              // Id used on ftok for semget key
#define SHMKEYPATH "/dev/null"  // Path used on ftok for shmget key
#define SHMKEYID 2              // Id used on ftok for shmget key

#define NUMSEMS 1               // Num of sems in created sem set

#define SHMSZ 2147483648 // Size of the shared mem segment
// Change value of 'shmmax' 
// cat 432306583 > /proc/sys/kernel/shmmax
// sudo sysctl -w kernel.shmmax=432306583
// sudo emacs /etc/sysctl.conf

struct header_mem_responses
{
  int width;
  int height;
  int nbAccess; // number of access to the weak learner in memory
};

// unique id in memory that identifies the end of the metadata
#define END_METADATA_WEAK_LEARNER = 0xFFFFFFFE


#endif //COMMON_H
