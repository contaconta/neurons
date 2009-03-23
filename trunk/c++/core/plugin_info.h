#include <Object.h>

typedef const bool (* plugin_init) (void);
typedef const bool (* plugin_run) (vector<Object*>& objects);
typedef const bool (* plugin_quit) (void);
