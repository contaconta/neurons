// Timer.h
//

#ifndef Timer_h
#define Timer_h

#include <time.h>
#include <sys/time.h>

class Timer
{
private:
  struct timeval start;
  clock_t zeroClock;

public:
  Timer();
  ~Timer();

  void reset();
  unsigned long getMilliseconds();
  unsigned long getMicroseconds();
  unsigned long getMillisecondsCPU();
  unsigned long getMicrosecondsCPU();
};

#endif
