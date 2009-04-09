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
  printf("activeId: %d\n", active_id);
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

void
on_add_selection_point_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
  if(togglebutton->active)
    selectToolMode = CPA_ADD_POINTS;
  else
     selectToolMode = CPA_SELECT;
}

bool unProjectMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType pointType)
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

  bool need_redraw = false;
  GLint viewport[4];
  GLdouble mvmatrix[16], projmatrix[16];
  GLdouble wx, wy, wz;
  GLdouble nx,ny,nz;
  GLint realy; // OpenGL y coordinate position, not the Mouse one of Gdk

  realy = (GLint)widgetHeight - 1 - mouse_last_y;
  int window_x = mouse_last_x;
  int window_y = realy;
  if(flag_draw_3D){
    setUpVolumeMatrices();
  }
  else if(flag_draw_XY)
    setUpMatricesXY(layerSpanViewZ);
  else if(flag_draw_XZ)
    setUpMatricesXZ(layerSpanViewZ);
  else if(flag_draw_YZ)
    setUpMatricesYZ(layerSpanViewZ);
  else if(flag_draw_combo){
    //If the click is on the XY corner
    if( (window_x > widgetWidth/2) && (window_y > widgetHeight/2) ){
      setUpMatricesXY(layerSpanViewZ);
      glViewport ((GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2,
		  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    }
    // In the YZ corner
    else if( (window_x > widgetWidth/2) && (window_y < widgetHeight/2) ){
      setUpMatricesYZ(layerSpanViewZ);
      glViewport ((GLsizei)widgetWidth/2, (GLsizei)0,
		  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    }
    //In the XZ corner
    else if( (window_x < widgetWidth/2) && (window_y > widgetHeight/2) ){
      setUpMatricesXZ(layerSpanViewZ);
      glViewport ((GLsizei)0,(GLsizei)widgetHeight/2,
		  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    }
    //In the 3D view, it makes no sense
    else if( (window_x < widgetWidth/2) && (window_y < widgetHeight/2) ){
      return false;
    }
  }

  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y);

  glPopMatrix();

  printf("unProjectMouseSelectTool %d %d\n", mouse_last_x, mouse_last_y);
  printf("World  coordinates: [%f %f %f]\n", wx, wy, wz);

  switch(selectToolMode)
    {
    case CPA_ADD_POINTS:
      {
	if(active_id == CT_SIMPLE_SELECTION)
	  {
	    Point3D* point=new Point3D();
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
	  string selection_name_save = currentSelectionSet->name + ".save";
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
	  //currentSelection->load(active_text);
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
