#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gtk/gtk.h>
#include <fstream>

#include "callbacks.h"
#include "interface.h"
#include "support.h"
#include "globalsE.h"
//#include "utils.h"
#include "functions.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <fstream>

void
on_create_selection_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* list_selections=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"list_selections"));
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      currentSelectionSet = new DoubleSet<float>;
      lSelections.push_back(currentSelectionSet);
      gtk_combo_box_append_text(list_selections, currentSelectionSet->name.c_str());
    }
}

bool pressMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType pointType)
{
  bool bRes = false;
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(selectionEditor,"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet == 0)
	return false;
    }

  GtkSpinButton* layer_XY_spin=GTK_SPIN_BUTTON(lookup_widget(GTK_WIDGET(ascEditor),"layer_XY_spin"));
  int z = gtk_spin_button_get_value_as_int(layer_XY_spin);

  GLdouble wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y, z);

  printf("pressMouseSelectTool %d %d %d\n", mouse_last_x, mouse_last_y, layerSpanViewZ);
  printf("World  coordinates: [%f %f %f]\n", wx, wy, wz);

  switch(selectToolMode)
    {
    case SELTOOL_MODE_RECTANGLE:
      mouse_startSel_x = mouse_last_x;
      mouse_startSel_y = mouse_last_y;
      break;
    case SELTOOL_MODE_ADD_POINTS:
      {
	if(active_id == CT_SIMPLE_SELECTION)
	  {
            printf("Adding new point\n");
	    PointDs<float>* point=new PointDs<float>();
	    point->coords.push_back((float)wx);
	    point->coords.push_back((float)wy);
	    point->coords.push_back((float)wz);

            int x,y,z;
            cube->micrometersToIndexes3(wx,wy,wz,                                        
                                        x,y,z);
            point->indexes.push_back(x);
            point->indexes.push_back(y);
            point->indexes.push_back(z);

            printf("x %d y %d z %d\n",x,y,z);

	    currentSelectionSet->addPoint(point, pointType+1);
	  }
        on_drawing3D_expose_event(drawing3D,NULL,NULL);
	bRes = true;
	break;
      }
    default:
      break;
    }
  return bRes;
}

bool motionMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType pointType)
{
  bool bRes = false;
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(selectionEditor,"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet == 0)
	return false;
    }

  GtkSpinButton* layer_XY_spin=GTK_SPIN_BUTTON(lookup_widget(GTK_WIDGET(ascEditor),"layer_XY_spin"));
  int z = gtk_spin_button_get_value_as_int(layer_XY_spin);

  GLdouble wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y, z);

  switch(selectToolMode)
    {
    case SELTOOL_MODE_RECTANGLE:
      // nothing to do
      break;
    case SELTOOL_MODE_ADD_POINTS:
      {
	if(active_id == CT_SIMPLE_SELECTION)
	  {
            printf("Add new point\n");
	    PointDs<float>* point=new PointDs<float>;
	    point->coords.push_back((float)wx);
	    point->coords.push_back((float)wy);
	    point->coords.push_back((float)wz);
            /*
            if(pointType == CPT_SOURCE)
              point->coords.push_back(0.0f);
            else
              point->coords.push_back(1.0f);
            */

            int x,y,z;
            cube->micrometersToIndexes3(wx,wy,wz,                                        
                                        x,y,z);
            point->indexes.push_back(x);
            point->indexes.push_back(y);
            point->indexes.push_back(z);

            printf("x %d y %d z %d\n",x,y,z);

	    currentSelectionSet->addPoint(point, pointType+1);
	  }

        on_drawing3D_expose_event(drawing3D,NULL,NULL);
	bRes = true;
	break;
      }
    default:
      break;
    }
  return bRes;
}

bool releaseMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType pointType)
{
  bool bRes = false;
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(selectionEditor,"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet == 0)
	return false;
    }

  GtkSpinButton* layer_XY_spin=GTK_SPIN_BUTTON(lookup_widget(GTK_WIDGET(ascEditor),"layer_XY_spin"));
  int z = gtk_spin_button_get_value_as_int(layer_XY_spin);

  GLdouble wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y, z);

  printf("releaseMouseSelectTool %d %d %d\n", mouse_last_x, mouse_last_y, layerSpanViewZ);
  printf("World  coordinates: [%f %f %f]\n", wx, wy, wz);

  printf("selectToolMode %d\n",selectToolMode);

  switch(selectToolMode)
    {
    case SELTOOL_MODE_RECTANGLE:
      {
	if(active_id == CT_SIMPLE_SELECTION)
	  {
            int min_x, min_y;
            int max_x, max_y;
            if(mouse_startSel_x<mouse_last_x)
              {
                min_x = mouse_startSel_x;
                max_x = mouse_last_x;
              }
            else
              {
                min_x = mouse_last_x;
                max_x = mouse_startSel_x;
              }
            if(mouse_startSel_y<mouse_last_y)
              {
                min_y = mouse_startSel_y;
                max_y = mouse_last_y;
              }
            else
              {
                min_y = mouse_last_y;
                max_y = mouse_startSel_y;
              }

            GtkSpinButton* layer_XY_spin=GTK_SPIN_BUTTON(lookup_widget(GTK_WIDGET(ascEditor),"layer_XY_spin"));
            int z = gtk_spin_button_get_value_as_int(layer_XY_spin);

            for(int x=min_x;x<max_x;x+=rect_sel_step_x)
              for(int y=min_y;y<max_y;y+=rect_sel_step_y)
                {
                  PointDs<float>* point = new PointDs<float>();
                  get_world_coordinates(wx, wy, wz, x, y, z);
                  point->coords.push_back((float)wx);
                  point->coords.push_back((float)wy);
                  point->coords.push_back((float)wz);
                  /*
                  if(pointType == CPT_SOURCE)
                    point->coords.push_back(0.0f);
                  else
                    point->coords.push_back(1.0f);
                  */

                  int s,t,u;
                  cube->micrometersToIndexes3(wx,wy,wz,                                        
                                              s,t,u);
                  point->indexes.push_back(s);
                  point->indexes.push_back(t);
                  point->indexes.push_back(u);

                  currentSelectionSet->addPoint(point, pointType+1);
                }
	  }

        on_drawing3D_expose_event(drawing3D,NULL,NULL);
	bRes = true;
	break;
      }
    default:
      break;
    }
  return bRes;
}

void
on_save_selection_clicked                (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet)
	{
	  //If the neuron is modified, the previous will be saved in the following name
	  const string selection_name_save = currentSelectionSet->name + ".save";
	  printf("Saving file %s\n", selection_name_save.c_str());
	  currentSelectionSet->save(selection_name_save);
	}
    }
}

void
on_clear_selection_clicked               (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet)
	{
	  currentSelectionSet->clear();
	}
    }
}


void
on_remove_selection_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkComboBox* list_selections=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"list_selections"));
    gchar* active_text = gtk_combo_box_get_active_text(list_selections);
    if(active_text != 0)
    {
        gtk_combo_box_remove_text(list_selections, gtk_combo_box_get_active(list_selections));
        for(vector< DoubleSet<float>* >::iterator itSelections = lSelections.begin();
            itSelections != lSelections.end();)
        {
            if(strcmp((*itSelections)->name.c_str(), active_text)==0)
            {
                printf("Erase %s\n", active_text);
                itSelections = lSelections.erase(itSelections);
                break;
            }
            else
                itSelections++;
        }
    }
}

void plugin_activate(const char* label)
{
  string dir("lib/plugins/");

  char * pPath = getenv ("NESEG_PATH");
  if (pPath!=0)
    dir = string(pPath) + dir;

  string sfile = dir+label;
  const char* plugin_name=(char*)sfile.c_str();

  plugin_run p_run;
  GModule *module = g_module_open (plugin_name, G_MODULE_BIND_LAZY);
  if (!module)
    {
      printf("Error while linking module %s\n", plugin_name);
    }
  else
    {
      if (!g_module_symbol (module, "plugin_run", (gpointer *)&p_run))
        {
          printf("Error while searching for symbol\n");
        }
      if (p_run == NULL)
        {
          printf("Symbol plugin_init is NULL\n");
        }
      else
        {
          vector<Object*> lObjects;
          lObjects.push_back(cube);
          lObjects.push_back(currentSelectionSet);

          for(int i = 0; i < toDraw.size(); i++)
            {
              lObjects.push_back(toDraw[i]);
              /* if((*itObj)->className()=="Image") */
                /* { */
                  /* Image< float >* img = (Image<float>*)*itObj; */
                  /* if(img!=0) */
                    /* { */
                      /* lObjects.push_back(img); */
                    /* } */
                  /* else */
                    /* printf("Null img\n"); */
                /* } */
            }

          p_run(lObjects); // execute init function
        }

      //Loads the key_pressed_symbol
      if (!g_module_symbol (module, "plugin_key_press_event",
                            (gpointer *)&p_key_press_event))
        {
          printf("Error while searching for symbol plugin_key_press_event\n");
        }
      if (p_key_press_event == NULL)
        {
          printf("Symbol p_key_press_event is NULL\n");
        }
      if (!g_module_symbol (module, "plugin_unproject_mouse",
                            (gpointer *)&p_unproject_mouse))
        {
          printf("Error while searching for symbol plugin_unproject_mouse\n");
        }
      if (p_unproject_mouse == NULL)
        {
          printf("Symbol p_unproject_mouse is NULL\n");
        }
      if (!g_module_symbol (module, "plugin_expose",
                            (gpointer *)&p_expose))
        {
          printf("Error while searching for symbol plugin_expose\n");
        }
      if (p_expose == NULL)
        {
          printf("Symbol p_expose is NULL\n");
        }

      /* if (!g_module_close (module)) */
        /* g_warning ("%s: %s", plugin_name, g_module_error ()); */

    }

}

void
on_run_graph_cuts_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
  printf("run_graph_cut %s\n", cube->type.c_str());
  const char* nm1 = "GraphCuts";
  plugin_activate(nm1);

  /*
  if(currentGraphCut)
    {
      gint layer_xy = -1;
      if(flag_draw_XY)
	{
	  GtkSpinButton* layer_XY_spin=GTK_SPIN_BUTTON(lookup_widget(GTK_WIDGET(ascEditor),"layer_XY_spin"));
	  layer_xy = gtk_spin_button_get_value_as_int(layer_XY_spin);
	  printf("layer_xy %d\n", layer_xy);
	}
      if(cube->type == "uchar"){
	currentGraphCut->run_maxflow((Cube<uchar,ulong>*)cube, layer_xy);
      }
      else if(cube->type == "float"){
	currentGraphCut->run_maxflow((Cube<float,double>*)cube, layer_xy);
      }
    }
  */
}

void
on_load_seeds_pressed                  (GtkButton       *button,
                                        gpointer         user_data)
{
  const char* nm1 = "LoadSeeds";
  plugin_activate(nm1);
}

void
on_load_selection_clicked                (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  GtkComboBox* list_selections=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"list_selections"));
  char* active_text = gtk_combo_box_get_active_text(list_selections);
  string selName(active_text);
  selName += ".save";
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet)
	{
	  printf("Loading %s in %s\n", selName.c_str(), currentSelectionSet->name.c_str());

	  currentSelectionSet->load(selName,cube);
	}
    }
  /*
  else
    {
      if(currentGraphCut)
	{
	  printf("Loading %s\n", currentGraphCut->graphcut_name.c_str());

	  if(cube->type == "uchar"){
	    currentGraphCut->load((Cube<uchar,ulong>*)cube, active_text);
	  }
	  else if(cube->type == "float"){
	    currentGraphCut->load((Cube<float,double>*)cube, active_text);
	  }
	  currentGraphCut->list();
	}
    }
  */
}

gboolean
on_selection_mode_button_press_event   (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
  GtkToolbar* toolbar = GTK_TOOLBAR(widget);
  GtkToolItem* item;
  int nbItems = gtk_toolbar_get_n_items(toolbar);
  for(int i=0;i<nbItems;i++)
    {
      item  = gtk_toolbar_get_nth_item(toolbar, i);
      gboolean active = gtk_toggle_tool_button_get_active(GTK_TOGGLE_TOOL_BUTTON(item));
      if(active)
        {
          if(i==0)
            selectToolMode = SELTOOL_MODE_ADD_POINTS;
          else if (i==1)
            selectToolMode = SELTOOL_MODE_RECTANGLE;

          break;
        }
    }

  return TRUE;
}

void on_mode_toggled(GtkToggleToolButton *toggletoolbutton, int itemId)
{
  gboolean active=gtk_toggle_tool_button_get_active(toggletoolbutton);
  if(active)
    {
      GtkToolbar *toolbar = GTK_TOOLBAR (gtk_widget_get_parent (GTK_WIDGET(toggletoolbutton)));
      GtkToolItem* item;
      int nbItems = gtk_toolbar_get_n_items(toolbar);
      for(int i=0;i<nbItems;i++)
        {
          if(i!=itemId)
            {
              item  = gtk_toolbar_get_nth_item(toolbar, i);
              gtk_toggle_tool_button_set_active(GTK_TOGGLE_TOOL_BUTTON(item), false);
            }
        }
    }
}

void
on_mode_point_toggled                  (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
  on_mode_toggled(toggletoolbutton,1);
  selectToolMode = SELTOOL_MODE_ADD_POINTS;
}

void
on_mode_select_toggled                 (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
  on_mode_toggled(toggletoolbutton,0);
  selectToolMode = SELTOOL_MODE_SELECT;
}

void
on_mode_rect_toggled                   (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
  on_mode_toggled(toggletoolbutton,2);
  selectToolMode = SELTOOL_MODE_RECTANGLE;
}

void
on_display_drawings_toggled            (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
  gboolean active=gtk_toggle_tool_button_get_active(toggletoolbutton);
  if(active)
    display_selection = true;
  else
    display_selection = false;
}
