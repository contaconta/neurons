
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

void draw_last_point()
{
  if(last_point != NULL){
    vector< float > microm(3);
    neuronita->neuronToMicrometers(last_point->coords,microm);
    glPushMatrix();
    glTranslatef(microm[0],
                 microm[1],
                 microm[2]);
    glColor4f(0.0,1.0,0.0,0.5);
    glutWireSphere(last_point->coords[3], 10, 10);
    glPopMatrix();
  }
}




void setUpVolumeMatrices()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fovy3D, aspect3D, zNear3D, zFar3D);
  glScalef(1.0,1.0,1.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScalef(1, 1, 1);
  glTranslatef(disp3DX,disp3DY,-disp3DZ);
  glRotatef(rot3DX,1,0,0);
  glRotatef(rot3DY,0,1,0);

  if(!flag_draw_combo)
    glViewport ((GLsizei)0,(GLsizei)0,
                (GLsizei)widgetWidth, (GLsizei)widgetHeight);
}


void setUpMatricesXY(int layerSpan)
{
  //Gets the cube coordinates
  GLfloat widthStep = float(cube->cubeWidth)*cube->voxelWidth/2;
  GLfloat heightStep = float(cube->cubeHeight)*cube->voxelHeight/2;
  GLfloat depthStep = float(cube->cubeDepth)*cube->voxelDepth/2;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = 512;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  float step_x = 1.0;
  float step_y = 1.0;

  if(cube->cubeWidth < max_texture_size){
    step_x = float(cube->cubeWidth)/max_texture_size;
  }
  if(cube->cubeHeight < max_texture_size){
    step_y = float(cube->cubeHeight)/max_texture_size;
  }


  glOrtho(-widthStep + cubeColToDraw*max_texture_size*cube->voxelWidth,
          -widthStep + (cubeColToDraw+step_x)*max_texture_size*cube->voxelWidth,
          heightStep - (cubeRowToDraw+step_y)*max_texture_size*cube->voxelHeight,
          heightStep - cubeRowToDraw*max_texture_size*cube->voxelHeight,
          (layerToDrawXY-layerSpan)*cube->voxelDepth,
          (layerToDrawXY+layerSpan)*cube->voxelDepth);

  glScalef(1.0,1.0,-1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0,0,depthStep);

  float tileWidth = min(float(cube->cubeWidth - max_texture_size*cubeColToDraw), float(max_texture_size));
  float tileHeight = min(float(cube->cubeHeight - max_texture_size*cubeRowToDraw), float(max_texture_size));

  if(!flag_draw_combo)
    glViewport ((GLsizei)0,(GLsizei)0,
                (GLsizei)widgetWidth, (GLsizei)widgetHeight);
}


void setUpMatricesYZ(int layerSpan)
{
  //Gets the cube coordinates
  GLfloat widthStep = float(cube->cubeWidth)*cube->voxelWidth/2;
  GLfloat heightStep = float(cube->cubeHeight)*cube->voxelHeight/2;
  GLfloat depthStep = float(cube->cubeDepth)*cube->voxelDepth/2;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = 512;
   int size_texture = 512;
  int max_texture  = 512;

  GLfloat increment_height =
    min(float(cube->cubeHeight - cube->nRowToDraw*size_texture), float(max_texture));
  float x_max = min(size_texture, (int)cube->cubeWidth - cube->nColToDraw*size_texture);


  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //FIXMEEEEEE
  glOrtho(-depthStep,
          +depthStep,
          heightStep - cubeRowToDraw*max_texture_size*cube->voxelHeight -
          increment_height*cube->voxelHeight,
          heightStep - cubeRowToDraw*max_texture_size*cube->voxelHeight,
          -widthStep + cubeColToDraw*max_texture_size*cube->voxelWidth+
          (layerToDrawYZ - layerSpan)*cube->voxelWidth,
          -widthStep + cubeColToDraw*max_texture_size*cube->voxelWidth +
          (layerToDrawYZ + layerSpan)*cube->voxelWidth
          );
  glScalef(1.0,1.0,-1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(-90,0,1,0);
/*   glTranslatef(widthStep,0,0); */
  glColor3f(1.0,1.0,1.0);

  if(!flag_draw_combo)
    glViewport ((GLsizei)0,(GLsizei)0,
                (GLsizei)widgetWidth, (GLsizei)widgetHeight);

}

void setUpMatricesXZ(int layerSpan)
{
  //Gets the cube coordinates
  GLfloat widthStep = float(cube->cubeWidth)*cube->voxelWidth/2;
  GLfloat heightStep = float(cube->cubeHeight)*cube->voxelHeight/2;
  GLfloat depthStep = float(cube->cubeDepth)*cube->voxelDepth/2;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = 512;
  int size_texture = 512;
  int max_texture  = 512;
  glMatrixMode(GL_PROJECTION);
  GLfloat increment_width =
    min(float(cube->cubeWidth - cube->nColToDraw*size_texture), float(max_texture));
  float y_max = min(size_texture, (int)cube->cubeHeight - cube->nRowToDraw*size_texture);

  glLoadIdentity();
  glOrtho(-widthStep + cubeColToDraw*max_texture_size*cube->voxelWidth,
          -widthStep + cubeColToDraw*max_texture_size*cube->voxelWidth+
          increment_width*cube->voxelWidth,
          +depthStep,
          -depthStep,
          heightStep
          - cubeRowToDraw*max_texture_size*cube->voxelHeight
          - (layerToDrawXZ-layerSpan)*cube->voxelHeight,
          heightStep
          - cubeRowToDraw*max_texture_size*cube->voxelHeight
          -(layerToDrawXZ+layerSpan)*cube->voxelHeight
          );
  glScalef(1.0,1.0,1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(-90,1,0,0);
/*   glTranslatef(0,-heightStep,0); */
  glColor3f(1.0,1.0,1.0);

  if(!flag_draw_combo)
    glViewport ((GLsizei)0,(GLsizei)0,
                (GLsizei)widgetWidth, (GLsizei)widgetHeight);

}

void draw_contours()
{
    for(vector< Contour<Point>* >::iterator itContours = lContours.begin();
        itContours != lContours.end(); itContours++)
    {
        (*itContours)->draw();
    }
}

void draw_graphcuts()
{
  int x = -1;
  int y = -1;
  int z = -1;
  if(flag_draw_XY)
    {
      GtkSpinButton* layer_XY_spin=GTK_SPIN_BUTTON(lookup_widget(GTK_WIDGET(ascEditor),"layer_XY_spin"));
      z = gtk_spin_button_get_value_as_int(layer_XY_spin);
    }
  else
  if(flag_draw_XZ)
    {
    }
  else
  if(flag_draw_YZ)
    {
    }


    for(vector< GraphCut<Point>* >::iterator itGraphCut = lGraphCuts.begin();
        itGraphCut != lGraphCuts.end(); itGraphCut++)
    {
      (*itGraphCut)->draw(x,y,z);
    }
}

gboolean
on_drawing3D_expose_event              (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data)
{
  //Gets the 3D drawing context
  GdkGLContext  *glcontext  = gtk_widget_get_gl_context  (widget);
  GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable (widget);

  //If we can not draw, return
  if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext))
    return FALSE;

  if(flag_minMax==0)
    glClearColor(1.0,1.0,1.0,1.0);
  else
    glClearColor(0.0,0.0,0.0,0.0);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  if(flag_draw_3D)
    {
      setUpVolumeMatrices();
      if(drawCube_flag){
        /* cube->draw(rot3DX,rot3DY,200,3*flag_minMax,0); */
        for(int i = 0; i < toDraw.size(); i++){
          toDraw[i]->draw();
          setUpVolumeMatrices();
        }
      }
      setUpVolumeMatrices();
      if(flag_cube_transparency)
        glDisable(GL_DEPTH_TEST);
      else
        glEnable(GL_DEPTH_TEST);
      if(flag_draw_neuron)
        /* neuronita->drawInOpenGl(); */
        glCallList(1);
      draw_last_point();
      draw_contours();
      draw_graphcuts();
    }

  //Draws the XY view
  if(flag_draw_XY)
    {
      glEnable(GL_DEPTH_TEST);
      setUpMatricesXY(layerSpanViewZ);
      glColor3f(1.0,1.0,1.0);
      if(drawCube_flag)
        cube->draw_layer_tile_XY(layerToDrawXY);
      if(flag_cube_transparency)
        glDisable(GL_DEPTH_TEST);
      else
        glEnable(GL_DEPTH_TEST);
      if(flag_draw_neuron)
        glCallList(1);
      /* for(int i = 0; i < toDraw.size(); i++) */
        /* toDraw[i]->draw(); */
      draw_last_point();
      draw_contours();
      draw_graphcuts();
      glDisable(GL_DEPTH_TEST);
    }

  if(flag_draw_XZ)
    {
      glEnable(GL_DEPTH_TEST);
      setUpMatricesXZ(layerSpanViewZ);
      if(drawCube_flag)
        cube->draw_layer_tile_XZ(layerToDrawXZ);
      if(flag_cube_transparency)
        glDisable(GL_DEPTH_TEST);
      else
        glEnable(GL_DEPTH_TEST);
      if(flag_draw_neuron)
        glCallList(1);
      draw_last_point();
      draw_contours();
      draw_graphcuts();
      glDisable(GL_DEPTH_TEST);
    }

  if(flag_draw_YZ)
    {
      glEnable(GL_DEPTH_TEST);
      setUpMatricesYZ(layerSpanViewZ);
      if(drawCube_flag)
        cube->draw_layer_tile_YZ(layerToDrawYZ);
      if(flag_cube_transparency)
        glDisable(GL_DEPTH_TEST);
      else
        glEnable(GL_DEPTH_TEST);
      if(flag_draw_neuron)
        glCallList(1);
      draw_last_point();
      draw_contours();
      draw_graphcuts();
      glDisable(GL_DEPTH_TEST);
    }

  if(flag_draw_combo){
    setUpVolumeMatrices();
    glViewport ((GLsizei)0,(GLsizei)0,
                (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);

    if(drawCube_flag){
      for(int i = 0; i < toDraw.size(); i++){
        setUpVolumeMatrices();
        glViewport ((GLsizei)0,(GLsizei)0,
                    (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
        toDraw[i]->draw();
      }
/*       cube->draw(rot3DX,rot3DY,200,flag_minMax,0); */
      setUpVolumeMatrices();
      glViewport ((GLsizei)0,(GLsizei)0,
                  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
      glEnable(GL_BLEND);
      setUpVolumeMatrices();
      cube->draw_layer_tile_XY(layerToDrawXY,1);
      cube->draw_layer_tile_XZ(layerToDrawXZ,1);
      cube->draw_layer_tile_YZ(layerToDrawYZ,1);
      glPushMatrix();
      glTranslatef(wx,wy,wz);
      glColor3f(0.0,1.0,1.0);
      glutSolidSphere(3, 10,10);
      glPopMatrix();
      glDisable(GL_BLEND);
    }
    if(flag_cube_transparency)
      glDisable(GL_DEPTH_TEST);
    else
      glEnable(GL_DEPTH_TEST);
    if(flag_draw_neuron)
      glCallList(1);
    draw_last_point();

    glEnable(GL_DEPTH_TEST);
    setUpMatricesXZ(100000);
    glViewport ((GLsizei)0,(GLsizei)widgetHeight/2,
                (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    if(drawCube_flag){
      cube->draw_layer_tile_XY(layerToDrawXY,1);
      cube->draw_layer_tile_XZ(layerToDrawXZ,0);
      cube->draw_layer_tile_YZ(layerToDrawYZ,1);
      glEnable(GL_BLEND);
      glDisable(GL_DEPTH_TEST);
      glPushMatrix();
      glTranslatef(wx,wy,wz);
      glColor3f(0.0,1.0,1.0);
      glutSolidSphere(1, 10,10);
      glPopMatrix();
      glDisable(GL_BLEND);
    }
    if(flag_cube_transparency)
      glDisable(GL_DEPTH_TEST);
    else
      glEnable(GL_DEPTH_TEST);
    if(flag_draw_neuron){
      setUpMatricesXZ(layerSpanViewZ);
      glViewport ((GLsizei)0,(GLsizei)widgetHeight/2,
                  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
      glCallList(1);
      draw_last_point();
    }

    glEnable(GL_DEPTH_TEST);
    setUpMatricesYZ(1000000);
    glViewport ((GLsizei)widgetWidth/2, (GLsizei)0,
                (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    if(drawCube_flag){
      cube->draw_layer_tile_XY(layerToDrawXY,1);
      cube->draw_layer_tile_XZ(layerToDrawXZ,1);
      cube->draw_layer_tile_YZ(layerToDrawYZ,0);
      glEnable(GL_BLEND);
      glDisable(GL_DEPTH_TEST);
      glPushMatrix();
      glTranslatef(wx,wy,wz);
      glColor3f(0.0,1.0,1.0);
      glutSolidSphere(1, 10,10);
      glPopMatrix();
      glDisable(GL_BLEND);
    }
    if(flag_cube_transparency)
      glDisable(GL_DEPTH_TEST);
    else
      glEnable(GL_DEPTH_TEST);
    if(flag_draw_neuron){
      setUpMatricesYZ(layerSpanViewZ);
      glViewport ((GLsizei)widgetWidth/2, (GLsizei)0,
                  (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
      glCallList(1);
      draw_last_point();
    }

    glEnable(GL_DEPTH_TEST);
    setUpMatricesXY(1000000);
    glViewport ((GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2,
                (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
    if(drawCube_flag){
      cube->draw_layer_tile_XY(layerToDrawXY,0);
      cube->draw_layer_tile_XZ(layerToDrawXZ,1);
      cube->draw_layer_tile_YZ(layerToDrawYZ,1);
      glEnable(GL_BLEND);
      glDisable(GL_DEPTH_TEST);
      glPushMatrix();
      glTranslatef(wx,wy,wz);
      glColor3f(0.0,1.0,1.0);
      glutSolidSphere(1, 10,10);
      glPopMatrix();
      glDisable(GL_BLEND);
    }
      if(flag_cube_transparency)
        glDisable(GL_DEPTH_TEST);
      else
        glEnable(GL_DEPTH_TEST);
      if(flag_draw_neuron){
        setUpMatricesXY(layerSpanViewZ);
        glViewport ((GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2,
                    (GLsizei)widgetWidth/2, (GLsizei)widgetHeight/2);
        glCallList(1);
        draw_last_point();
      }
    glDisable(GL_DEPTH_TEST);
  }


  if(majorMode == MOD_ASCEDITOR){
    exposeAsc(widget, event, user_data);
  }


  //Show what has been drawn
  if (gdk_gl_drawable_is_double_buffered (gldrawable))
    gdk_gl_drawable_swap_buffers (gldrawable);
  else
    glFlush ();
  gdk_gl_drawable_gl_end (gldrawable);
  return TRUE;
}
