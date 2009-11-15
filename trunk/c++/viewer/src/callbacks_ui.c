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
#include "Cube_P.h"
#include "Cube_T.h"
#include "GraphFactory.h"
#include "utils.h"
#include "functions.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <fstream>

void get_world_coordinates(double &wx, double &wy, double &wz, int x, int y, int z)
{
  GLint realy; /*  OpenGL y coordinate position, not the Mouse one of Gdk */
               /*   realy = widgetHeight - mouse_last_y; */
  realy =(GLint) widgetHeight - 1 - y;
  int window_x = x;
  int window_y = realy;
  GLint viewport[4];
  GLdouble mvmatrix[16], projmatrix[16];
  GLdouble nx,ny,nz;

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

  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
  glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
  GLfloat depth;
  //if(z!=-1)
  //  depth = z;
  //else
    glReadPixels( x,
                  realy,
                  1,
                  1,
                  GL_DEPTH_COMPONENT,
                  GL_FLOAT,
                  &depth );


  /*
  gluUnProject ((GLdouble) mouse_last_x, (GLdouble) realy, depth,
                mvmatrix, projmatrix, viewport, &wx, &wy, &wz);
  */
  gluUnProject ((GLdouble) x, (GLdouble) realy, depth,
                mvmatrix, projmatrix, viewport, &wx, &wy, &wz);
}


void get_world_coordinates(double &wx, double &wy, double &wz, bool change_layers, int z )
{
  GLint realy; /*  OpenGL y coordinate position, not the Mouse one of Gdk */
               /*   realy = widgetHeight - mouse_last_y; */
  realy =(GLint) widgetHeight - 1 - mouse_last_y;
  int window_x = mouse_last_x;
  int window_y = realy;
/*   printf("WidgetSize =          [%f, %f]\n", widgetWidth, widgetHeight); */
/*   printf("Window coordinates:   [%i, %i]\n", mouse_last_x ,mouse_last_y); */
/*   printf("ViewPort coordinates: [%i, %i]\n", mouse_last_x ,realy); */

  //There is no information on the depth (as it is only the first layer)
  /* if(flag_draw_3D) */
    /* return; */

  GLint viewport[4];
  GLdouble mvmatrix[16], projmatrix[16];
  GLdouble nx,ny,nz;

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

  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
  glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
  GLfloat depth;
  //if(z!=-1)
  //  depth = z;
  //else
    glReadPixels( mouse_last_x,
                  realy,
                  1,
                  1,
                  GL_DEPTH_COMPONENT,
                  GL_FLOAT,
                  &depth );


  gluUnProject ((GLdouble) mouse_last_x, (GLdouble) realy, depth,
                mvmatrix, projmatrix, viewport, &wx, &wy, &wz);

  vector< int > indexes(3);
  vector< float > world(3);
  world[0] = wx;
  world[1] = wy;
  world[2] = wz;
  cube->micrometersToIndexes(world, indexes);

  if( (mod_display == MOD_DISPLAY_COMBO) && change_layers){
    //If the click is on the XY corner
    if( (window_x > widgetWidth/2) && (window_y > widgetHeight/2) ){
      layerToDrawXZ = indexes[1]%D_MAX_TEXTURE_SIZE;
      layerToDrawYZ = indexes[0]%D_MAX_TEXTURE_SIZE;
    }
    // In the YZ corner
    if( (window_x > widgetWidth/2) && (window_y < widgetHeight/2) ){
      layerToDrawXY = indexes[2]%D_MAX_TEXTURE_SIZE;
      layerToDrawXZ = indexes[1]%D_MAX_TEXTURE_SIZE;
    }
    //In the XZ corner
    if( (window_x < widgetWidth/2) && (window_y > widgetHeight/2) ){
      layerToDrawXY = indexes[2]%D_MAX_TEXTURE_SIZE;
      layerToDrawYZ = indexes[0]%D_MAX_TEXTURE_SIZE;
    }
    //In the 3D view, it makes no sense
    if( (window_x < widgetWidth/2) && (window_y < widgetHeight/2) ){
      return;
    }
  }
}


void unProjectMouse()
{
  GLdouble wx, wy, wz;
  vector< int > indexes(3);

  get_world_coordinates(wx, wy, wz, true);

  printf("World Coordinates: %f %f %f\n", wx, wy, wz);
  vector< float > world(3);
  world[0] = wx;
  world[1] = wy;
  world[2] = wz;
  if(cube!=NULL && cube->dummy == false){
    //vector< int > indexes(3);
    cube->micrometersToIndexes(world, indexes);

/*         layerToDrawXY = indexes[2]%D_MAX_TEXTURE_SIZE; */
/*         layerToDrawXZ = indexes[1]%D_MAX_TEXTURE_SIZE; */
/*         layerToDrawYZ = indexes[0]%D_MAX_TEXTURE_SIZE; */

    if( (indexes[0] >= 0) && (indexes[0] < cube->cubeWidth) &&
        (indexes[1] >= 0) && (indexes[1] < cube->cubeHeight) &&
        (indexes[2] >= 0) && (indexes[2] < cube->cubeDepth) ){
      printf("Indexes: %i %i %i and value %f\n", indexes[0], indexes[1], indexes[2],
             cube->getValueAsFloat(indexes[0], indexes[1], indexes[2]));
    }
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }

  if(img!=NULL){
    img->micrometersToIndexes(world, indexes);
    indexes[0]+=img->width/2;
    indexes[1]-=img->height/2;
    printf("Indexes: %i %i %i\n", indexes[0], indexes[1], indexes[2]);
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }


  /** If the mode is MOD_ASCEDITOR, change the asc.*/
  if(majorMode == MOD_ASCEDITOR){
    unProjectMouseAsc(mouse_last_x, mouse_last_y);
  }
  if(majorMode == MOD_SELECT_EDITOR){
    pressMouseSelectTool(mouse_last_x, mouse_last_y, CPT_SOURCE);
  }

  if(p_unproject_mouse != NULL){
    p_unproject_mouse(mouse_last_x, mouse_last_y);
    /* p_unproject_mouse(indexes[0], indexes[1]); */
  }


}

bool select_tool_handle_mouse(int x, int y)
{
  bool bRes = false;
  if(majorMode == MOD_SELECT_EDITOR)
    {
    if( mouse_buttons[2] )
      {
	bRes = motionMouseSelectTool(x, y, CPT_SINK);
      }
    else
      if( mouse_buttons[0] )
	{
	  bRes = motionMouseSelectTool(x, y, CPT_SOURCE);
	}
    }
  return bRes;
}

gboolean
on_drawing3D_motion_notify_event       (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data)
{
  int x, y;
  GdkModifierType state;
  gdk_window_get_pointer (event->window, &x, &y, &state);

  if(select_tool_handle_mouse(x,y))
    return true;

  mouse_current_x = x;
  mouse_current_y = y;
  int diffx=x-mouse_last_x;
  int diffy=y-mouse_last_y;
  if((abs(diffx) + abs(diffy)) < 2)
    return true;
  mouse_last_x=x;
  mouse_last_y=y;
  if( mouse_buttons[2] )
  {
    if( mouse_buttons[1] )
        disp3DZ -= (float) 1.0f * diffx;
    else
        disp3DZ -= (float) 0.20f * diffx;
    on_drawing3D_expose_event(widget, NULL, user_data);
  }
  else
    if( mouse_buttons[0] )
      {
        rot3DX -= (float) 0.5f * diffy;
        rot3DY -= (float) 0.5f * diffx;

        on_drawing3D_expose_event(widget, NULL, user_data);
      }
    else
      if( mouse_buttons[1] )
        {
          disp3DX += (float) 0.5f * diffx;
          disp3DY -= (float) 0.5f * diffy;
          on_drawing3D_expose_event(widget, NULL, user_data);
        }

  if( (mod_display == MOD_DISPLAY_COMBO) &&
      !((mouse_last_x < widgetWidth/2) && (mouse_last_y > widgetHeight/2))
      ){
    get_world_coordinates(wx, wy, wz, false); //Nasty global variables
/*     printf("%f %f %f\n", wx, wy, wz); */
  }

  if(MOD_ASCEDITOR)
    on_drawing3D_expose_event(drawing3D,NULL, NULL);

  if(p_motion_notify!= NULL){
    p_motion_notify(widget, event, user_data);
  }

  return FALSE;
}


gboolean
on_drawing3D_button_press_event        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
  printf("on_drawing3D_button_press_event\n");
  mouse_last_x= (int)event->x;
  mouse_last_y= (int)event->y;
  switch(event->button)
    {
    case 1:
      mouse_buttons[0] = (mouse_buttons[0]==0)?1:0;
      unProjectMouse();
      break;
    case 2:
      mouse_buttons[1] = (mouse_buttons[1]==0)?1:0;
      break;
    case 3:
      mouse_buttons[2] = (mouse_buttons[2]==0)?1:0;
      if(majorMode == MOD_SELECT_EDITOR){
	pressMouseSelectTool(mouse_last_x, mouse_last_y,CPT_SINK);
      }
      break;
    default:
      break;
     }
  return FALSE;
}


gboolean
on_drawing3D_button_release_event      (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
  mouse_last_x=(int)event->x;
  mouse_last_y=(int)event->y;
  switch(event->button)
  {
  case 1:
    mouse_buttons[0] = 0;
    if(majorMode == MOD_SELECT_EDITOR)
      releaseMouseSelectTool(mouse_last_x, mouse_last_y,CPT_SOURCE);
    break;
  case 2:
    mouse_buttons[1] = 0;
    break;
  case 3:
    mouse_buttons[2] = 0;
    if(majorMode == MOD_SELECT_EDITOR)
      releaseMouseSelectTool(mouse_last_x, mouse_last_y,CPT_SINK);
    break;
  default:
    break;
  }
  return FALSE;
}


gboolean
on_drawing3D_key_press_event           (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data)
{
  if(event->keyval == 'a')
    {
      if(mod_display == MOD_DISPLAY_3D)
        disp3DZ -= 10;
      else if(mod_display == MOD_DISPLAY_XY)
        layerToDrawXY++;
      else if(mod_display == MOD_DISPLAY_XZ)
        layerToDrawXZ++;
      else if(mod_display == MOD_DISPLAY_YZ)
        layerToDrawYZ++;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 's')
    {
      if(mod_display == MOD_DISPLAY_3D)
        disp3DZ += 10;
      else if(mod_display == MOD_DISPLAY_XY)
        layerToDrawXY--;
      else if(mod_display == MOD_DISPLAY_XZ)
        layerToDrawXZ--;
      else if(mod_display == MOD_DISPLAY_YZ)
        layerToDrawYZ--;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'j'){
    if(mod_display == MOD_DISPLAY_3D)
      rot3DY -= 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'l'){
    if(mod_display == MOD_DISPLAY_3D)
      rot3DY += 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'i'){
    if( mod_display == MOD_DISPLAY_3D)
      rot3DX -= 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'k'){
    if(mod_display == MOD_DISPLAY_3D)
      rot3DX += 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'J'){
    if(mod_display == MOD_DISPLAY_3D)
      disp3DX -= 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'L'){
    if(mod_display == MOD_DISPLAY_3D)
      disp3DX += 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'I'){
    if(mod_display == MOD_DISPLAY_3D)
      disp3DY += 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == 'K'){
    if(mod_display == MOD_DISPLAY_3D)
      disp3DY -= 5;
    on_drawing3D_expose_event(drawing3D,NULL, NULL);
  }
  if(event->keyval == '?'){
    printf("Drawing parameters=\n  position = [%f,%f,%f]\n rotation=[%f,%f]\n",
           disp3DX,disp3DY,disp3DZ,rot3DX,rot3DY);
  }

  if(event->keyval == 'n')
    {
      flag_draw_neuron = !flag_draw_neuron;
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }

  if(event->keyval == 'e')
    {
      timeStep--;
      for(vector< VisibleE* >::iterator itObj = toDraw.begin();
          itObj != toDraw.end(); itObj++)
        {
          if((*itObj)->className()=="Cube_T"){
            Cube_T* cb = dynamic_cast<Cube_T*>(*itObj);
            if(timeStep < 0){
              timeStep = cb->cubes.size()-1;
            }
            cb->timeStep = timeStep;
          }
        }
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }

  if(event->keyval == 'r')
    {
      timeStep ++;
      for(vector< VisibleE* >::iterator itObj = toDraw.begin();
          itObj != toDraw.end(); itObj++)
        {
          if((*itObj)->className()=="Cube_T"){
            Cube_T* cb = dynamic_cast<Cube_T*>(*itObj);
            if(timeStep >= cb->cubes.size()){
              timeStep = 0;
            }
            cb->timeStep = timeStep;
          }
        }
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'w')
    {
      for(vector< VisibleE* >::iterator itObj = toDraw.begin();
          itObj != toDraw.end(); itObj++)
        {
          if((*itObj)->className()=="Cube_T"){
            Cube_T* cb = dynamic_cast<Cube_T*>(*itObj);
            cb->d_halo = !cb->d_halo;
          }
        }
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }
  if(event->keyval == 'q')
    {
      for(vector< VisibleE* >::iterator itObj = toDraw.begin();
          itObj != toDraw.end(); itObj++)
        {
          if((*itObj)->className()=="Cube_T"){
            Cube_T* cb = dynamic_cast<Cube_T*>(*itObj);
            cb->d_gt = !cb->d_gt;
          }
        }
      on_drawing3D_expose_event(drawing3D,NULL, NULL);
    }



  if(majorMode == MOD_ASCEDITOR)
    keyPressedAsc(widget,event,user_data);

  if(p_key_press_event != NULL){
    p_key_press_event(widget, event, user_data);
  }

  return FALSE;
}

gboolean
on_drawing3D_scroll_event              (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data)
{
  if(majorMode == MOD_ASCEDITOR){
    scrollAsc(widget, event, user_data);
  }

  return FALSE;
}

//-------------------------------------------------
// SHADERS

void
on_shaders_clicked                     (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkWidget* ascSelectShaders = create_ascSelectShaders();
  gtk_widget_show (ascSelectShaders);
}

char *shaders_textFileRead(const char *fn)
{
  FILE *fp;
  char *content = NULL;

  int count=0;

  if (fn != NULL) {
    fp = fopen(fn,"rt");

    if (fp != NULL) {

      fseek(fp, 0, SEEK_END);
      count = ftell(fp);
      rewind(fp);

      if (count > 0) {
        content = (char *)malloc(sizeof(char) * (count+1));
        count = fread(content,sizeof(char),count,fp);
        content[count] = '\0';
      }
      fclose(fp);
    }
  }
  return content;
}

void shaders_activation(gint active)
{
    if(active!=0)
    {
        char *vs = NULL,*fs = NULL,*fs2 = 0;

        shader_v = glCreateShader(GL_VERTEX_SHADER);
        shader_f = glCreateShader(GL_FRAGMENT_SHADER);

        const char* nm1 = "../shaders/edge.vert";
        vs = shaders_textFileRead(nm1);
        const char* nm2 = "../shaders/edge.frag";
        fs = shaders_textFileRead(nm2);

        const char * ff = fs;
        const char * vv = vs;

        glShaderSource(shader_v, 1, &vv,0);
        glShaderSource(shader_f, 1, &ff,0);

        free(vs);
        free(fs);

        glCompileShader(shader_v);
        glCompileShader(shader_f);

        shader_p = glCreateProgram();
        glAttachShader(shader_p, shader_f);
        glAttachShader(shader_p, shader_v);

        glLinkProgram(shader_p);
        glUseProgram(shader_p);
	}
	else
	{
		glLinkProgram(0);
		glUseProgram(0);

		glDeleteShader(shader_v);
		glDeleteShader(shader_f);
		glDeleteProgram(shader_p);
	}
}

void
on_select_shaders_changed              (GtkComboBox     *combobox,
                                        gpointer         user_data)
{
    gint active = gtk_combo_box_get_active(combobox);
    shaders_activation(active);
    // Set filterId used by the fragment shader
    GLint filterId = glGetUniformLocation(shader_p,"filterId");
    glUniform1i(filterId,active);
}

void
on_min_alpha_changed                   (GtkEditable     *editable,
                                        gpointer         user_data)
{
  min_alpha = gtk_spin_button_get_value(GTK_SPIN_BUTTON(editable));
  cube->min_alpha = min_alpha;
}


void
on_max_alpha_changed                   (GtkEditable     *editable,
                                        gpointer         user_data)
{
  max_alpha = gtk_spin_button_get_value(GTK_SPIN_BUTTON(editable));
  cube->max_alpha = max_alpha;
}


void
on_cbBlendFunction_changed             (GtkComboBox     *combobox,
                                        gpointer         user_data)
{
  blendFunction = (eBlendFunction)gtk_combo_box_get_active(combobox);
  cube->blendFunction = (Cube_P::eBlendFunction)blendFunction;
}


