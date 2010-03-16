#include "Timer.h"

Timer::Timer()
{
  reset();
}

Timer::~Timer()
{
}

void Timer::reset()
{
  zeroClock = clock();
  gettimeofday(&start, NULL);
}

unsigned long Timer::getMilliseconds()
{
  struct timeval now;
  gettimeofday(&now, NULL);
  return (now.tv_sec-start.tv_sec)*1000+(now.tv_usec-start.tv_usec)/1000;
}

unsigned long Timer::getMicroseconds()
{
  struct timeval now;
  gettimeofday(&now, NULL);
  return (now.tv_sec-start.tv_sec)*1000000+(now.tv_usec-start.tv_usec);
}

unsigned long Timer::getMillisecondsCPU()
{
  clock_t newClock = clock();
  return (unsigned long)((float)(newClock-zeroClock) / ((float)CLOCKS_PER_SEC/1000.0)) ;
}

unsigned long Timer::getMicrosecondsCPU()
{
  clock_t newClock = clock();
  return (unsigned long)((float)(newClock-zeroClock) / ((float)CLOCKS_PER_SEC/1000000.0)) ;
}
