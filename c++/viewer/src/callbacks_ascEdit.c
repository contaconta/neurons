#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gtk/gtk.h>
#include <fstream>

#include "callbacks.h"
#include "interface.h"
#include "support.h"
#include "Neuron.h"
#include "globalsE.h"
#include "CubeFactory.h"
#include "CloudFactory.h"
#include "GraphFactory.h"
#include "utils.h"
#include "functions.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <fstream>


//Functions specific to the ascParser
void undo()
{
  string neuron_name_save = neuron_name + ".save";

  // Resets the neuron
  neuronita->axon.resize(0);
  neuronita->dendrites.resize(0);
  neuronita->allPointsVector.resize(0);
  neuronita->asc = new ascParser2(neuron_name_save);
  neuronita->asc->parseFile(neuronita);
  glDeleteLists(neuronita->v_glList,1);
  glNewList (neuronita->v_glList, GL_COMPILE);
  neuronita->drawInOpenGlAsLines();
  glEndList ();
  neuronita->save(neuron_name);
  on_drawing3D_expose_event(drawing3D, NULL, NULL);
}


//KEYBOARD
void keyPressedAsc
(GtkWidget       *widget,
 GdkEventKey     *event,
 gpointer         user_data)
{
  if(event->keyval == 'z')
    {
      ascEditor_action = NPA_SELECT;
      printf("The next neuron point will be selected\n");
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'x')
    {
      ascEditor_action = NPA_APPLY_RECURSIVE_OFFSET;
      printf("The last neuron point and following be moved to the next click\n");
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'c')
    {
      ascEditor_action = NPA_APPLY_RECURSIVE_OFFSET_CLOSEST_POINT_TO_CLICK;
      printf("The closest neuron point and following will be moved to the next click\n");
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'v')
    {
/*       flag_save_cube_coordinate = !flag_save_cube_coordinate; */
/*       printf("The next point is saved as a cube coordinate\n"); */
      ascEditor_action = NPA_CHANGE_POINT_CLOSEST_TO_CLICK;
      printf("The closest point is moved here\n");
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'f')
    {
      ascEditor_action = NPA_CHANGE_POINT;
      printf("The selected neuron point will be moved to the next click\n");
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'b')
    {
      ascEditor_action = NPA_NONE;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'q')
    {
      ascEditor_action = NSA_ADD_POINTS;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'w')
    {
      ascEditor_action = NSA_ADD_BRANCH;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'e')
    {
      ascEditor_action = NSA_ADD_DENDRITE;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'r')
    {
      ascEditor_action = NSA_ERASE_POINT;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'g')
    {
      ascEditor_action = NSA_CONTINUE_SEGMENT;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 't')
    {
      ascEditor_action = NSA_ERASE_WHOLE_SEGMENT;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'h')
    {
      ascEditor_action = NSA_ERASE_SEGMENT_FROM_HERE;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
 if(event->keyval == 'u')
    {
      undo();
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }


}


// BUTTONS CALLBACKS
void
on_branch_button_clicked               (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NSA_ADD_BRANCH;
}

void
on_release_asc_action_clicked          (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NPA_NONE;
}

void
on_continue_segment_button_clicked     (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NSA_CONTINUE_SEGMENT;
}

void
on_Erase_Point_clicked                 (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NSA_ERASE_POINT;
}

void
on_new_dendrite_button_clicked         (GtkButton       *button,
                                        gpointer         user_data)
{
  printf("In the next click a branck will be included\n");
  ascEditor_action = NSA_ADD_DENDRITE;
}

void
on_delete_segment_clicked              (GtkButton       *button,
                                        gpointer         user_data)
{
  printf("In the next click the segment will be deleted\n");
  ascEditor_action = NSA_ERASE_WHOLE_SEGMENT;
}


void
on_ascEditor_width_value_changed       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  asc_point_width = gtk_spin_button_get_value(spinbutton);
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}


void
on_select_point_clicked                (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NPA_SELECT;
}


void
on_recursive_offset_closer_point_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NPA_APPLY_RECURSIVE_OFFSET_CLOSEST_POINT_TO_CLICK;
}


void
on_recursive_offset_selected_point_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NPA_APPLY_RECURSIVE_OFFSET;
}


void
on_move_selected_point_clicked         (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NPA_CHANGE_POINT;
}


void
on_move_closer_point_clicked           (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NPA_CHANGE_POINT_CLOSEST_TO_CLICK;
}


void
on_delete_segment_from_point_clicked   (GtkButton       *button,
                                        gpointer         user_data)
{
  ascEditor_action = NSA_ERASE_SEGMENT_FROM_HERE;
}



void scrollAsc
(GtkWidget       *widget,
 GdkEvent        *event,
 gpointer         user_data)
{
  GdkEventScroll* e = (GdkEventScroll*)event;
   if (e->direction == GDK_SCROLL_DOWN)
   {
     asc_point_width -= 0.25;
     on_drawing3D_expose_event(widget, NULL, user_data);
   }
   if (e->direction == GDK_SCROLL_UP){
     asc_point_width += 0.25;
     on_drawing3D_expose_event(widget, NULL, user_data);
   }
}


/* GET THE WORLD POSITION AND ... ACTION! */
void unProjectMouseAsc(int mouse_lsat_x, int mouse_last_y)
{

  if(neuronita == NULL)
    return;

  bool need_redraw = false;
  GLint viewport[4];
  GLdouble mvmatrix[16], projmatrix[16];
  GLdouble wx, wy, wz;
  GLdouble nx,ny,nz;
  GLint realy; /*  OpenGL y coordinate position, not the Mouse one of Gdk */
  NeuronPoint*  pp = NULL;

  realy = (GLint)widgetHeight - 1 - mouse_last_y;
  int window_x = mouse_last_x;
  int window_y = realy;
  if(mod_display == MOD_DISPLAY_3D){
    setUpVolumeMatrices();
  }
  if(mod_display == MOD_DISPLAY_XY)
    setUpMatricesXY(layerSpanViewZ);
  if(mod_display == MOD_DISPLAY_XZ)
    setUpMatricesXZ(layerSpanViewZ);
  if(mod_display == MOD_DISPLAY_YZ)
    setUpMatricesYZ(layerSpanViewZ);
  if(mod_display == MOD_DISPLAY_COMBO){
    //If the click is on the XY corner
    if( (window_x > widgetWidth/2) && (window_y > widgetHeight/2) ){
      setUpMatricesXY(layerSpanViewZ);
      glViewport ((GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2,
                  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    }
    // In the YZ corner
    if( (window_x > widgetWidth/2) && (window_y < widgetHeight/2) ){
      setUpMatricesYZ(layerSpanViewZ);
      glViewport ((GLsizei)widgetWidth/2, (GLsizei)0,
                  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    }
    //In the XZ corner
    if( (window_x < widgetWidth/2) && (window_y > widgetHeight/2) ){
      setUpMatricesXZ(layerSpanViewZ);
      glViewport ((GLsizei)0,(GLsizei)widgetHeight/2,
                  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    }
    //In the 3D view, it makes no sense
    if( (window_x < widgetWidth/2) && (window_y < widgetHeight/2) ){
      return;
    }
  }

  get_world_coordinates(wx, wy, wz, mouse_last_x, mouse_last_y);

  neuronita->setUpGlMatrices();
  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
  glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
  GLfloat depth;
  glReadPixels( mouse_last_x,
                realy,
                1,
                1,
                GL_DEPTH_COMPONENT,
                GL_FLOAT,
                &depth );
  gluUnProject ((GLdouble) mouse_last_x, (GLdouble) realy, depth,
                mvmatrix, projmatrix, viewport, &nx, &ny, &nz);

  glPopMatrix();

  pp = neuronita->findClosestPoint(nx,ny,nz);
  if(pp!= NULL)
    printf("PP = %s\n", pp->name.c_str());
  else
    printf("The neuron has no points\n");

  /* printf("World  coordinates: [%f %f %f]\n", wx, wy, wz); */
  /* printf("Neuron coordinates: [%f %f %f]\n", nx, ny, nz); */
  /* if(pp!=NULL) */
    /* printf("Closep coordinates: [%f %f %f]\n", pp->coords[0],  pp->coords[1],  pp->coords[2]); */

  vector< float > new_coords(4);
  new_coords[0] = wx;
  new_coords[1] = wy;
  new_coords[2] = wz;
  new_coords[3] = 1;
  vector< float > neuron_coords(4);
  neuronita->micrometersToNeuron(new_coords,neuron_coords);

  printf("unProjectMouseAsc\n");

  // ACTION!
  //If the neuron is modified, the previous will be saved in the following name
  string neuron_name_save = neuron_name + ".save";

  if(ascEditor_action == NPA_SELECT){
    if(pp!=NULL){
      last_point = pp;
      /* printf("Point selected: %s %i\n", last_point->name.c_str(), last_point->pointNumber); */
      /* printf("Point parent  : %s\n", last_point->parent->name.c_str()); */
    }
    ascEditor_action = NPA_NONE;
  }

  if(ascEditor_action == NPA_APPLY_RECURSIVE_OFFSET){
    neuronita->save(neuron_name_save);
    if(last_point != NULL){
      printf("P_name: %s P_parent: %s P_parent_kids %i\n",
             last_point->name.c_str(), last_point->parent->name.c_str(),
             last_point->parent->childs.size());
        neuronita->applyRecursiveOffset
          (last_point->parent,
           last_point->pointNumber,
           nx - last_point->coords[0],
           ny - last_point->coords[1],
           nz - last_point->coords[2]);

      need_redraw = true;
      ascEditor_action = NPA_NONE;
    }
  }

  //Move the closest point to the mouse position
  if(ascEditor_action == NPA_APPLY_RECURSIVE_OFFSET_CLOSEST_POINT_TO_CLICK){
    neuronita->save(neuron_name_save);
    last_point = pp;
    neuronita->applyRecursiveOffset
      (last_point->parent,
       last_point->pointNumber,
       nx - last_point->coords[0],
       ny - last_point->coords[1],
       nz - last_point->coords[2]);
    need_redraw = true;
    ascEditor_action = NPA_NONE;
  }

  //Moves last_point to the mouse position.
  if(ascEditor_action == NPA_CHANGE_POINT_CLOSEST_TO_CLICK){
    neuronita->save(neuron_name_save);
    last_point = pp;
    neuronita->changePointPosition(last_point->parent,
                                   last_point->pointNumber,
                                   nx, ny, nz);
    need_redraw = true;
    ascEditor_action = NPA_NONE;
  }

  //Changes the selected point to the position
  if(ascEditor_action == NPA_CHANGE_POINT){
    neuronita->save(neuron_name_save);
    if(last_point!=NULL){
      neuronita->changePointPosition(last_point->parent,
                                     last_point->pointNumber,
                                     nx, ny, nz);
      need_redraw = true;
      ascEditor_action = NPA_NONE;
    }
  }


  if(ascEditor_action == NSA_ERASE_WHOLE_SEGMENT){
    neuronita->save(neuron_name_save);
    last_point = pp;
    if(current_segment!=NULL)
      printf("Erasing the segment %s\n", current_segment->name.c_str());
    current_segment = last_point->parent;
    //We will not erase dendrites or axons
    if(current_segment->parent != NULL){
      int idx = 0;
      for(idx; idx < current_segment->parent->childs.size(); idx++)
        if (current_segment == current_segment->parent->childs[idx])
          break;
      printf("The idx is %i\n", idx);
      vector<NeuronSegment*>::iterator itr = current_segment->parent->childs.begin() + idx;
      current_segment->parent->childs.erase(itr);
    }
    else {
      int idx = 0;
      bool isDendrite = false;
      for(idx; idx < neuronita->dendrites.size(); idx++)
        if (current_segment == neuronita->dendrites[idx]){
          isDendrite = true;
          break;
        }
      if(isDendrite){
        printf("The idx is %i\n", idx);
        vector<NeuronSegment*>::iterator itr = neuronita->dendrites.begin() + idx;
        neuronita->dendrites.erase(itr);
      }
      idx = 0;
      bool isAxon = false;
      for(idx; idx < neuronita->axon.size(); idx++)
        if (current_segment == neuronita->axon[idx]){
          isAxon = true;
          break;
        }
      if(isAxon){
        printf("The idx is %i\n", idx);
        vector<NeuronSegment*>::iterator itr = neuronita->axon.begin() + idx;
        neuronita->axon.erase(itr);
      }
    }
    ascEditor_action = NPA_NONE;
    need_redraw = true;
    current_segment = neuronita->findClosestSegment(nx, ny, nz);
  }

  if(ascEditor_action == NSA_ERASE_SEGMENT_FROM_HERE){
    if(  (pp == NULL) || (pp->parent == NULL)) {
      printf("I do not know which segment to erase or from which point\n");
      ascEditor_action = NPA_NONE;
    }
    else{
      neuronita->save(neuron_name_save);
      last_point = pp;
      current_segment = pp->parent;
      int last_point_idx = 0;
      for(int i = 0; i < current_segment->points.size(); i++){
        if( &current_segment->points[i] == last_point)
          last_point_idx = i;
      }
      current_segment->points.resize(last_point_idx);
      current_segment->childs.resize(0);
      ascEditor_action = NPA_NONE;
      need_redraw = true;
    }
  }


  if(flag_save_cube_coordinate){
    flag_save_cube_coordinate = !flag_save_cube_coordinate;
    std::ofstream out("cube.txt", ios_base::app);
    out << wx << " " << wy << " " << wz << std::endl;
    out.close();
  }

  if(flag_save_neuron_coordinate){
    flag_save_neuron_coordinate = !flag_save_neuron_coordinate;
    std::ofstream out("neuron.txt", ios_base::app);
    out << nx << " " << ny << " " << nz << std::endl;
    out.close();
  }

  if(ascEditor_action == NSA_ADD_POINTS)
    {
      neuronita->save(neuron_name_save);
      current_segment->points.push_back(NeuronPoint(nx,ny,nz,asc_point_width));
      last_point = new NeuronPoint(nx,ny,nz,asc_point_width);
      need_redraw = true;
    }


  if(ascEditor_action == NSA_ADD_DENDRITE){
    neuronita->save(neuron_name_save);
    printf("Adding dendrite\n");
    NeuronColor color(1.0,0.0,0.0);
    neuronita->dendrites.push_back(new NeuronSegment());
    neuronita->dendrites[neuronita->dendrites.size()-1]->root =
      NeuronPoint(nx,ny,nz,asc_point_width);
    neuronita->dendrites[neuronita->dendrites.size()-1]->points.push_back(
                           NeuronPoint(nx,ny,nz,asc_point_width));
    char buff[512];
    sprintf(buff, "d-%02i",neuronita->dendrites.size()-1);
    neuronita->dendrites[neuronita->dendrites.size()-1]->name = buff;
    current_segment = neuronita->dendrites[neuronita->dendrites.size()-1];
    current_segment->color = color;
    printf("Now we will add points to the dendrite\n");
    ascEditor_action = NSA_ADD_POINTS;
    need_redraw = true;
  }

  if(ascEditor_action == NSA_ADD_BRANCH){
    neuronita->save(neuron_name_save);
    printf("add_branch\n");
    NeuronSegment* cl_sg = neuronita->findClosestSegment(nx, ny, nz);
    int pt_idx = neuronita->findIndexOfClosestPointInSegment(nx, ny, nz, cl_sg);
    current_segment = neuronita->splitSegment(cl_sg, pt_idx);
    last_point = &current_segment->points[0];
    ascEditor_action = NSA_ADD_POINTS;
    need_redraw = true;
  }

  if(ascEditor_action == NSA_ERASE_POINT){
    neuronita->save(neuron_name_save);
    NeuronSegment* cl_sg = neuronita->findClosestSegment(nx, ny, nz);
    int idx = neuronita->findIndexOfClosestPointInSegment(nx, ny, nz, cl_sg);
    vector< NeuronPoint>::iterator itr = cl_sg->points.begin();
    itr = itr + idx;
    cl_sg->points.erase(itr);
    ascEditor_action = NPA_NONE;
    need_redraw = true;
  }

  if(ascEditor_action == NSA_CONTINUE_SEGMENT){
    current_segment = neuronita->findClosestSegment(nx, ny, nz);
    last_point      = &current_segment->points[current_segment->points.size()-1];
    ascEditor_action = NSA_ADD_POINTS;
    need_redraw = true;
  }

  if(need_redraw){
    glDeleteLists(neuronita->v_glList,1);
    glNewList (neuronita->v_glList, GL_COMPILE);
    neuronita->drawInOpenGlAsLines();
    glEndList ();
    neuronita->save(neuron_name);
    on_drawing3D_expose_event(drawing3D, NULL, NULL);
    need_redraw = false;
    printf("need_redraw stuff\n");
  }
}

void exposeAsc
(GtkWidget       *widget,
 GdkEventExpose        *event,
 gpointer         user_data)
{

  double wx, wy, wz;
  get_world_coordinates(wx, wy, wz, mouse_current_x, mouse_current_y);
  glDisable(GL_DEPTH_TEST);
  glColor3f(0.0,0.0,1.0);
  glPushMatrix();
  glTranslatef(wx, wy, wz);
  glutWireSphere(asc_point_width,10,10);
  glPopMatrix();
  glEnable(GL_DEPTH_TEST);
}





