#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gtk/gtk.h>
#include <fstream>

#include "callbacks.h"
#include "interface.h"
#include "support.h"
#include "globalsE.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <fstream>




void
on_new_neuron_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_open_neuron_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}

void
on_save_neuron_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_save_as1_activate                   (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_quit1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_cut1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_copy1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_paste1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_delete1_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_about1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}

void
on_editAsc_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  GtkWidget* window  = create_ascEditControls ();
  gtk_widget_show (window);
}

void
on_3dmenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  mod_display = MOD_DISPLAY_3D;
 on_drawing3D_expose_event(drawing3D,NULL, NULL);
}


void
on_xymenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  mod_display = MOD_DISPLAY_XY; 
on_drawing3D_expose_event(drawing3D,NULL, NULL);
}


void
on_xzmenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  mod_display = MOD_DISPLAY_XZ;
 on_drawing3D_expose_event(drawing3D,NULL, NULL);
}


void
on_yzmenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  mod_display = MOD_DISPLAY_YZ;
 on_drawing3D_expose_event(drawing3D,NULL, NULL);
}


void
on_combomenu_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  mod_display = MOD_DISPLAY_COMBO;
 on_drawing3D_expose_event(drawing3D,NULL, NULL);
}


