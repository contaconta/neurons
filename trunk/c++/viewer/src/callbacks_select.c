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
      currentSelectionSet = new DoubleSet<Point>;
      lSelections.push_back(currentSelectionSet);
      gtk_combo_box_append_text(list_selections, currentSelectionSet->name.c_str());
    }
  else
    {
      currentGraphCut = new GraphCut<Point>(cube);
      lGraphCuts.push_back(currentGraphCut);
      gtk_combo_box_append_text(list_selections, currentGraphCut->graphcut_name.c_str());
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
  else
    {
      if(currentGraphCut == 0)
	return false;
    }

  GLdouble wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y);

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
            printf("Add new point\n");
	    Point3Dc* point=new Point3Dc();
	    point->coords.push_back((float)wx);
	    point->coords.push_back((float)wy);
	    point->coords.push_back((float)wz);
	    currentSelectionSet->addPoint(point, pointType+1);
	  }
	else
	  {
	    Point3Di* point=new Point3Di();
	    point->w_coords.push_back((float)wx);
	    point->w_coords.push_back((float)wy);
	    point->w_coords.push_back((float)wz);
	    cube->micrometersToIndexes(point->w_coords, point->coords);
	    if(pointType == CPT_SOURCE)
	      {
		printf("Add source point : %d %d %d\n", point->coords[0], point->coords[1], point->coords[2]);
		currentGraphCut->addSourcePoint(point);
	      }
	    else
	      {
		printf("Add sink point : %d %d %d\n", point->coords[0], point->coords[1], point->coords[2]);
		currentGraphCut->addSinkPoint(point);
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
  else
    {
      if(currentGraphCut == 0)
	return false;
    }

  GLdouble wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y);

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
	    Point3Dc* point=new Point3Dc();
	    point->coords.push_back((float)wx);
	    point->coords.push_back((float)wy);
	    point->coords.push_back((float)wz);
	    currentSelectionSet->addPoint(point, pointType+1);
	  }
	else
	  {
	    Point3Di* point=new Point3Di();
	    point->w_coords.push_back((float)wx);
	    point->w_coords.push_back((float)wy);
	    point->w_coords.push_back((float)wz);
	    cube->micrometersToIndexes(point->w_coords, point->coords);
	    if(pointType == CPT_SOURCE)
	      {
		printf("Add source point : %d %d %d\n", point->coords[0], point->coords[1], point->coords[2]);
		currentGraphCut->addSourcePoint(point);
	      }
	    else
	      {
		printf("Add sink point : %d %d %d\n", point->coords[0], point->coords[1], point->coords[2]);
		currentGraphCut->addSinkPoint(point);
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
  else
    {
      if(currentGraphCut == 0)
	return false;
    }

  GLdouble wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y);

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

            for(int x=min_x;x<max_x;x+=rect_sel_step_x)
              for(int y=min_y;y<max_y;y+=rect_sel_step_y)
                {
                  Point3Dc* point=new Point3Dc();
                  get_world_coordinates(wx, wy, wz, x, y);
                  point->coords.push_back((float)wx);
                  point->coords.push_back((float)wy);
                  point->coords.push_back((float)wz);
                  //printf("World  coordinates: [%f %f %f]\n", wx, wy, wz);
                  currentSelectionSet->addPoint(point, pointType+1);
                }
	  }
	else
	  {
            /*
	    Point3Di* point=new Point3Di();
	    point->w_coords.push_back((float)wx);
	    point->w_coords.push_back((float)wy);
	    point->w_coords.push_back((float)wz);
	    cube->micrometersToIndexes(point->w_coords, point->coords);
	    if(pointType == CPT_SOURCE)
	      {
		printf("Add source point : %d %d %d\n", point->coords[0], point->coords[1], point->coords[2]);
		currentGraphCut->addSourcePoint(point);
	      }
	    else
	      {
		printf("Add sink point : %d %d %d\n", point->coords[0], point->coords[1], point->coords[2]);
		currentGraphCut->addSinkPoint(point);
	      }
            */
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
  else
    {
      if(currentGraphCut)
	{
	  //If the neuron is modified, the previous will be saved in the following name
	  string graphcut_name_save = currentGraphCut->graphcut_name + ".save";
	  printf("Saving file %s\n", graphcut_name_save.c_str());
	  currentGraphCut->save(graphcut_name_save);
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
  else
    {
      if(currentGraphCut)
	{
	  printf("Clearing %s\n", currentGraphCut->graphcut_name.c_str());
	  currentGraphCut->clear();
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
        for(vector< DoubleSet<Point>* >::iterator itSelections = lSelections.begin();
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

void
on_run_graph_cuts_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
  printf("run_graph_cut %s\n", cube->type.c_str());
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
}


void
on_load_selection_clicked                (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* selection_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"selection_type"));
  int active_id = gtk_combo_box_get_active(selection_type);
  GtkComboBox* list_selections=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"list_selections"));
  char* active_text = gtk_combo_box_get_active_text(list_selections);
  if(active_id == CT_SIMPLE_SELECTION)
    {
      if(currentSelectionSet)
	{
	  printf("Loading %s in %s\n", active_text, currentSelectionSet->name.c_str());

	  currentSelectionSet->load(active_text);
	}
    }
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
