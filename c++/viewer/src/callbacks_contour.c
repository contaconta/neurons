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
on_create_contour_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* list_contours=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"list_contours"));
  GtkComboBox* contour_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"contour_selection_type"));
  int active_id = gtk_combo_box_get_active(contour_type);
  if(active_id == CT_SIMPLE_CONTOUR)
    {
      currentContour = new Contour<Point>;
      lContours.push_back(currentContour);
      gtk_combo_box_append_text(list_contours, currentContour->contour_name.c_str());
    }
  else
    {
      currentGraphCut = new GraphCut<Point>;
      lGraphCuts.push_back(currentGraphCut);
      gtk_combo_box_append_text(list_contours, currentGraphCut->graphcut_name.c_str());
    }
}

void
on_add_contour_point_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    if(togglebutton->active)
        contourEditor_action = CPA_ADD_POINTS;
    else
        contourEditor_action = CPA_SELECT;
}

void unProjectMouseContour(int mouse_last_x, int mouse_last_y, ContourPointType pointType)
{
  GtkComboBox* contour_type=GTK_COMBO_BOX(lookup_widget(drawing3D->parent,"contour_selection_type"));
  int active_id = gtk_combo_box_get_active(contour_type);
  if(active_id == CT_SIMPLE_CONTOUR)
    {
      if(currentContour == 0)
	return;
    }
  else
    {
      if(currentGraphCut == 0)
	return;
    }

    bool need_redraw = false;
    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    GLdouble wx, wy, wz;
    GLdouble nx,ny,nz;
    GLint realy; /*  OpenGL y coordinate position, not the Mouse one of Gdk */

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
            return;
        }
    }

    get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y);

    glPopMatrix();

    printf("unProjectMouseContour\n");
    printf("World  coordinates: [%f %f %f]\n", wx, wy, wz);

    switch(contourEditor_action)
    {
        case CPA_ADD_POINTS:
            {
	      Point3D* point=new Point3D();
	      vector<float> coords;
	      coords.push_back((float)wx);
	      coords.push_back((float)wy);
	      coords.push_back((float)wz);
	      /* cube->micrometersToIndexes(coords, point->coordsV); */
	      if(active_id == CT_SIMPLE_CONTOUR)
		{
		  currentContour->addPoint(point);
		}
	      else
		{
		  if(pointType == CPT_SOURCE)
		    currentGraphCut->addSourcePoint(point);
		  else
		    currentGraphCut->addSinkPoint(point);
		}
                break;
            }
        default:
            break;
    }
}

void
on_save_contour_clicked                (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkComboBox* contour_type=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"contour_selection_type"));
  int active_id = gtk_combo_box_get_active(contour_type);
  if(active_id == CT_SIMPLE_CONTOUR)
    {
      if(currentContour)
	{
	  //If the neuron is modified, the previous will be saved in the following name
	  string contour_name_save = currentContour->contour_name + ".save";
	  currentContour->save(contour_name_save);
	}
    }
  else
    {
      if(currentGraphCut)
	{
	  //If the neuron is modified, the previous will be saved in the following name
	  string graphcut_name_save = currentGraphCut->graphcut_name + ".save";
	  currentGraphCut->save(graphcut_name_save);
	}
    }
}

void
on_clear_contour_clicked               (GtkButton       *button,
                                        gpointer         user_data)
{
    if(currentContour)
    {
        currentContour->clear();
    }
}


void
on_remove_contour_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
    GtkComboBox* list_contours=GTK_COMBO_BOX(lookup_widget(GTK_WIDGET(button),"list_contours"));
    gchar* active_text = gtk_combo_box_get_active_text(list_contours);
    if(active_text != 0)
    {
        gtk_combo_box_remove_text(list_contours, gtk_combo_box_get_active(list_contours));
        for(vector< Contour<Point>* >::iterator itContours = lContours.begin();
            itContours != lContours.end();)
        {
            if(strcmp((*itContours)->contour_name.c_str(), active_text)==0)
            {
                printf("Erase %s\n", active_text);
                itContours = lContours.erase(itContours);
                break;
            }
            else
                itContours++;
        }
    }
}

void
on_run_graph_cuts_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
  printf("run_graph_cut\n");
  if(currentGraphCut)
    {
      currentGraphCut->run_maxflow((Cube<float,float>*)cube);
    }
}
