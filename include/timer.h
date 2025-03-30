#ifndef TIMER_H
#define TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>

// Timer function
double get_time(clock_t start);

#ifdef __cplusplus
}
#endif

#endif // TIMER_H