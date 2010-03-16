#ifndef PLUGIN_FUNCTIONS_H_
#define PLUGIN_FUNCTIONS_H_

#include <Object.h>
#include <gtk/gtk.h>

typedef const bool (* plugin_init) (void);
typedef const bool (* plugin_run) (vector<Object*>& objects);
typedef const bool (* plugin_quit) (void);
typedef const void (* plugin_key_press_event) (GtkWidget* widget,
                                         GdkEventKey* event, gpointer user_data);

#endif

