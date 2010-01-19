#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gtk/gtk.h>
#include <fstream>

#include "callbacks.h"
#include "interface.h"
#include "support.h"
#include "Neuron.h"
#include "Contour.h"
#include "globals.h"
#include "CubeFactory.h"
#include "CloudFactory.h"
#include "GraphFactory.h"
#include "TiffFactory.h"
#include "ListOfObjects.h"
#include "utils.h"
#include "functions.h"
#include "Axis.h"
#ifdef WITH_BBP
#include "BBP_Morphology.h"
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <fstream>
#include "SWC.h"
#include <pthread.h>

using namespace std;
int tick = 0;

static void* saveScreenShotThreadF(void* vptr_args)
{

  sleep(0.5);
  screenShot_waitedEnough = true;
}

void
saveScreenShot (char* filename)
{
    IplImage* toSave = cvCreateImage(cvSize((int)widgetWidth,(int)widgetHeight),
                                     IPL_DEPTH_32F, 3);
    printf("Smileeeeee\n");
    glReadPixels( 0, 0,
                  (int)widgetWidth,
                  (int)widgetHeight,
                  GL_BGR,
                  GL_FLOAT, toSave->imageData );
    printf("... cheeseee\n");
    toSave->origin = 1;
    cvSaveImage(filename, toSave);
    printf(" .. you are beautiful! -> %s\n", filename);
    cvReleaseImage(&toSave);
}

void addObjectFromString(string name)
{

  string extension;
  extension = getExtension(name);
  printf("The extension is %s\n", extension.c_str());
  /* std::cout << name << "a" << std::endl; */

  if(extension == "nfo"){
    cube = CubeFactory::load(name);
    float* tf = new float[256];
    for(int i=0;i<255;i++)
      tf[i] = i;
    tf[255] = 0;
    //((Cube<float,double>*)cube)->set_tf(tf);
    cube->v_draw_projection = flag_minMax;
    cube->load_texture_brick(cubeRowToDraw, cubeColToDraw);
    if(nCubes == 0){
      cube->v_r = 1.0;
      cube->v_g = 1.0;
      cube->v_b = 1.0;
      nCubes++;
    } else if(nCubes == 1){
      cube->v_r = 1.0;
      cube->v_g = 0.0;
      cube->v_b = 0.0;
      nCubes++;
    } else if(nCubes == 2){
      cube->v_r = 0.0;
      cube->v_g = 1.0;
      cube->v_b = 0.0;
      nCubes++;
    } else if(nCubes == 3){
      cube->v_r = 0.0;
      cube->v_g = 0.0;
      cube->v_b = 1.0;
      nCubes++;
    }
    toDraw.push_back(cube);
  }
  else if (extension == "nfc")  {
    Cube_C* cubec = new Cube_C(name);
    cube = cubec;
    cubec->v_draw_projection = flag_minMax;
    toDraw.push_back(cubec);
    cubec->load_texture_brick(cubeRowToDraw, cubeColToDraw);
  }
  else if ( (extension == "tiff") ||
            (extension == "TIFF") ||
            (extension == "tif")  ||
            (extension == "TIF")
           )  {
    /* Cube_P* cubec = CubeFactory::load(name); */
    /* cube = cubec; */
    VisibleE* tiff = TiffFactory::load(name);
    tiff->v_draw_projection = flag_minMax;
    if((tiff->className() == "Cube") ||
       (tiff->className() == "Cube_C") ||
       (tiff->className() == "Cube_DTC") ) {
      cube = (Cube_P*)tiff;
      cube->load_texture_brick(cubeRowToDraw, cubeColToDraw);
    }
    toDraw.push_back(tiff);

  }
  else if ((extension == "swc") || (extension == "SWC"))  {
    toDraw.push_back(new SWC(name));
  }
  else if( extension == "cbt"){
    cube = new Cube_T(name);
    cube->v_r = 1.0;
    cube->v_g = 1.0;
    cube->v_b = 1.0;
    cube->v_draw_projection = flag_minMax;
    cube->load_texture_brick(cubeRowToDraw, cubeColToDraw);
    toDraw.push_back(cube);
  }
  else if( (extension == "asc") || (extension == "ASC") ){
    neuron_name = name;
    neuronita = new Neuron(name);
    string classN = neuronita->className();
    printf("There is a neuron and its class is: %s\n", classN.c_str());
    toDraw.push_back(neuronita);
  }
  else if (extension == "gr") {
    printf("Hey, the extension is really gr\n");
    Graph_P* gr = GraphFactory::load(name);
    toDraw.push_back(gr);
  } else if (extension == "cl"){
    Cloud_P* cd = CloudFactory::load(name);
    toDraw.push_back(cd);
  }
  else if ((extension == "jpg") || (extension == "png"))  {
    img = new Image<float>(name);
    toDraw.push_back(img);
  }
#ifdef WITH_BBP
  else if ((extension == "h5")){
    BBP_Morphology* bbpmorph = new BBP_Morphology(name);
    toDraw.push_back(bbpmorph);
  }
#endif
  //Text file with a lot of objects to draw
  else if( extension == "lst"){
    std::ifstream in(name.c_str());
    if(!in.good())
      {
        printf("vivaView::addObjectFromString does not recognize %s\n",name.c_str());
        exit(0);
      }
    string s;
    while(getline(in,s))
    {
      printf("%s\n", s.c_str());
      fflush(stdout);
      addObjectFromString(s);
    }
    printf("\n");
  }
  else if (extension == "lsto" ){
    ListOfObjects* lsto = new ListOfObjects(name);
    toDraw.push_back(lsto);
  }
  else if (extension == "stc") {
    Configuration* conf = new Configuration(name);
    Cube<uchar, ulong>* cube =
      loadImageStackFromSFC
      (conf->retrieve("directory"),
       conf->retrieve("format"),
       conf->retrieveInt("layerInit"), conf->retrieveInt("layerEnd"),
       conf->retrieveFloat("voxelWidth"), conf->retrieveFloat("voxelHeight"),
       conf->retrieveFloat("voxelDepth"));
    toDraw.push_back(cube);
  }
  else{
    printf("neseg::on_drawing3D_realize:: unknown file type %s, exiting... \n",
           name.c_str());
    exit(0);
  }

 on_drawing3D_expose_event(drawing3D,NULL, NULL);

  if(majorMode == MOD_SCREENSHOT){
    pthread_create(&screenShotThread, NULL, saveScreenShotThreadF, NULL);
  }

}


void
on_drawing3D_realize                   (GtkWidget       *widget,
                                        gpointer         user_data)
{
  /* for(int i = 0; i < objectNames.size(); i++) */
    /* std::cout << objectNames[i] << std::endl; */
}



void init_GUI_late()
{
  std::cout << "on_drawing3D_realize" << std::endl;

  /* GdkGLContext *glcontext   = gtk_widget_get_gl_context  (drawing3D); */
  /* GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable (drawing3D); */

  /* Axis* axis = new Axis(); */
  /* toDraw.push_back(axis); */
  /* objectNames.push_back("Axis"); */

  //Create the objects
  nCubes = 0;
  for(int i = 0; i < objectNames.size(); i++){
    if(objectNames[i] == "Axis")
      continue; //Nothing to be done
    addObjectFromString(objectNames[i]);
  }
  flag_initializing_GUI = false;

  /*** OpenGL BEGIN ***/
  /* if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext)) */
  /* { */
    /* printf("Realize event: there is no gdk_gl_drawable\n"); */
    /* return; */
  /* } */

  /** Loads the drawing parameters to the cube.*/
  /* for(int i = 0; i < toDraw.size(); i++){ */
    /* toDraw[i]->v_draw_projection = flag_minMax; */
  /* } */
  /* gdk_gl_drawable_gl_end (gldrawable); */
  /*** OpenGL END*/
 /* on_drawing3D_expose_event(drawing3D,NULL, NULL); */

}



gboolean
on_drawing3D_configure_event           (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data)
{
  drawing3D = widget;
  widgetWidth = widget->allocation.width;
  widgetHeight = widget->allocation.height;
  std::cout << "Widget width and height = "
            << widgetWidth << " " << widgetHeight << std::endl;
  return FALSE;
}





void
on_view_entry_changed                  (GtkComboBox     *combobox,
                                        gpointer         user_data)
{
  gint active = gtk_combo_box_get_active(combobox);
  printf("The entry has been changed!\n");
  if(!flag_initializing_GUI)
    {
      switch(active)
        {
        case 0:
          mod_display = MOD_DISPLAY_3D;
          break;
        case 1:
          mod_display = MOD_DISPLAY_XY;
          break;
        case 2:
          mod_display = MOD_DISPLAY_XZ;
          break;
        case 3:
          mod_display = MOD_DISPLAY_YZ;
          break;
        case 4:
          mod_display = MOD_DISPLAY_COMBO;
          break;
        case 5:
          mod_display = MOD_DISPLAY_DUAL;
          break;
        }
    }
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}

void
on_layer_XY_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  layerToDrawXY = gtk_spin_button_get_value(spinbutton);
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
  //gtk_widget_queue_draw(drawing3D);  
}


void
on_draw_cube_toggle_toggled            (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
  drawCube_flag = !drawCube_flag;
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}

void
on_buttonViewOnlyCube_toggled          (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
  drawOnlyCube_flag = !drawOnlyCube_flag;
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}



void
on_layer_view_value_changed            (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  layerSpanViewZ = (int)gtk_spin_button_get_value(spinbutton);
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}


void
on_cube_col_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  cubeColToDraw = (int)gtk_spin_button_get_value(spinbutton);
  for(vector< VisibleE* >::iterator itObj = toDraw.begin();
      itObj != toDraw.end(); itObj++)
    {
      if( (*itObj)->className()=="Cube" ||
          (*itObj)->className()=="Cube_C"||
          (*itObj)->className()=="Cube_T"){
        Cube_P* cubeDraw = dynamic_cast<Cube_P*>(*itObj);
        cubeDraw->load_texture_brick(cubeRowToDraw, cubeColToDraw);
      }
    }
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}


void
on_cube_row_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  cubeRowToDraw = (int)gtk_spin_button_get_value(spinbutton);
  for(vector< VisibleE* >::iterator itObj = toDraw.begin();
      itObj != toDraw.end(); itObj++)
    {
      if((*itObj)->className()=="Cube" ||
          (*itObj)->className()=="Cube_C"||
          (*itObj)->className()=="Cube_T"){
        Cube_P* cubeDraw = dynamic_cast<Cube_P*>(*itObj);
        cubeDraw->load_texture_brick(cubeRowToDraw, cubeColToDraw);
      }
    }
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}



void
on_ascEditor_destroy                   (GtkObject       *object,
                                        gpointer         user_data)
{
  exit(0);
}


void
on_draw_neuron_toggled                 (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
  flag_draw_neuron = !flag_draw_neuron;
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}


void
on_cube_transparency_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
  flag_cube_transparency = !flag_cube_transparency;
  on_drawing3D_expose_event(drawing3D,NULL, user_data);

}


void
on_layerXZ_spin_value_changed          (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  layerToDrawXZ = gtk_spin_button_get_value(spinbutton);
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}


void
on_layer_YZ_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  layerToDrawYZ = gtk_spin_button_get_value(spinbutton);
  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}


void
on_get_matrix_button_clicked           (GtkButton       *button,
                                        gpointer         user_data)
{
  int error = system("matlab -nosplash -nojvm -r getMatrix");
  delete neuronita;
  neuronita = new Neuron(neuron_name);
  glDeleteLists(1,1);
  glNewList (1, GL_COMPILE);
  /*     glEnable(GL_BLEND); */
  neuronita->drawInOpenGlAsLines();
  /*     glBlendFunc(GL_SRC_ALPHA,GL_SRC_COLOR); */
  /*     neuronita->drawInOpenGl(); */
  /*     glDisable(GL_BLEND); */
  glEndList();

  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}




void
on_screenshot_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

  GtkWidget *dialog;

  dialog = gtk_file_chooser_dialog_new ("Save Screenshot",
                                        NULL,
                                        GTK_FILE_CHOOSER_ACTION_SAVE,
                                        GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                                        GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
                                        NULL);

  if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
    {
      char *filename;

      filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));

      saveScreenShot(filename);
    }
  gtk_widget_destroy (dialog);
}

void
on_menu_plugins_activate               (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}

void
on_videolayers_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  if(cube!=NULL){
     /* flag_draw_3D = false; */
     /* flag_draw_XY = true; */
     /* flag_draw_XZ = false; */
     /* flag_draw_YZ = false; */
     /* flag_draw_combo = false; */
     /* flag_draw_dual = false; */
      mod_display = MOD_DISPLAY_3D;
     on_drawing3D_expose_event(drawing3D,NULL, user_data);
     char imageName[1024];
     int error = system("rm -rf /tmp/img*.jpg");
     for(int i = 0; i < cube->cubeDepth; i++){
       layerToDrawXY = i;
       on_drawing3D_expose_event(drawing3D,NULL, user_data);
       sprintf(imageName,"/tmp/img%03i.jpg", i);
       saveScreenShot(imageName);
     }
     char command[1024];
     sprintf(command, "mencode_movie.sh /tmp %i %i output.avi 3", (int)widgetWidth, (int)widgetHeight);
     printf("%s\n", command);
     error = system(command);
  }
}


void
on_videorotation_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  if(cube!=NULL){
     /* flag_draw_3D = true; */
     /* flag_draw_XY = false; */
     /* flag_draw_XZ = false; */
     /* flag_draw_YZ = false; */
     /* flag_draw_combo = false; */
     /* flag_draw_dual = false; */
      mod_display = MOD_DISPLAY_3D;
      on_drawing3D_expose_event(drawing3D,NULL, user_data);
     int error = system("rm -rf /tmp/img*.jpg");
     char imageName[1024];
     for(int i = 0; i < 360; i+=5){
       rot3DY = i;
       on_drawing3D_expose_event(drawing3D,NULL, user_data);
       sprintf(imageName,"/tmp/img%03i.jpg", i);
       saveScreenShot(imageName);
     }
     char command[1024];
     sprintf(command, "mencode_movie.sh /tmp %i %i output.avi 10", (int)widgetWidth, (int)widgetHeight);
     printf("%s\n", command);
     error = system(command);
  }

}


//-------------------------------------------------


void
on_videorotationtime_activate          (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  //Gets the pointer to the Cube_T
  Cube_T* cb = NULL;
  for(vector< VisibleE* >::iterator itObj = toDraw.begin();
      itObj != toDraw.end(); itObj++)
    {
      if((*itObj)->className()=="Cube_T"){
        cb = dynamic_cast<Cube_T*>(*itObj);
        if(timeStep < 0){
          timeStep = cb->cubes.size()-1;
        }
        cb->timeStep = timeStep;
      }
    }

  if(cb!=NULL){
     /* flag_draw_3D = true; */
     /* flag_draw_XY = false; */
     /* flag_draw_XZ = false; */
     /* flag_draw_YZ = false; */
     /* flag_draw_combo = false; */
     /* flag_draw_dual = false; */
    mod_display = MOD_DISPLAY_3D;
     on_drawing3D_expose_event(drawing3D,NULL, user_data);
     int error = system("rm -rf /tmp/img*.jpg");
     char imageName[1024];
     int maxTime = cb->cubes.size();
     for(int ts = 0; ts < maxTime; ts++){
     /* for(int i = 0; i < 360; i+=5){ */
       /* int i = 360*ts/maxTime; //angle */
       int i = 0;
       rot3DY = i;
       cb->timeStep = ts;
       on_drawing3D_expose_event(drawing3D,NULL, user_data);
       sprintf(imageName,"/tmp/img%03i.jpg", ts);
       saveScreenShot(imageName);
     }
     char command[1024];
     sprintf(command, "mencode_movie.sh /tmp %i %i output.avi 10", (int)widgetWidth, (int)widgetHeight);
     printf("%s\n", command);
     error = system(command);
  }


}


void
on_open_3d_stack1_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  _3DLIS = create_loadImageStack();
  gtk_widget_show (_3DLIS);
}


void
on_open_4d_stack1_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}

void
on_projectionComboBox_changed          (GtkComboBox     *combobox,
                                        gpointer         user_data)
{
  gint active = gtk_combo_box_get_active(combobox);
  if(!flag_initializing_GUI)
    {
      switch(active)
        {
        case 0:
          flag_minMax = 0;
          break;
        case 1:
          flag_minMax = 1;
          break;
        default:
          break;
        }
    }
  for(vector< VisibleE* >::iterator itObj = toDraw.begin();
      itObj != toDraw.end(); itObj++)
    {
      if( ((*itObj)->className()=="Cube") ||
               ((*itObj)->className()=="Cube_T") ||
               ((*itObj)->className()=="Cube_C") ||
               ((*itObj)->className()=="Cube_P")
               ){
        Cube_P* cubeDraw = dynamic_cast<Cube_P*>(*itObj);
        cubeDraw->v_draw_projection = flag_minMax;
      }
    }



  on_drawing3D_expose_event(drawing3D,NULL, user_data);
}




gboolean
on_drawing3D_drag_drop                 (GtkWidget       *widget,
                                        GdkDragContext  *context,
                                        gint             x,
                                        gint             y,
                                        guint            time,
                                        gpointer         user_data)
{
  const gchar *name = gtk_widget_get_name (widget);
  g_print ("%s: drag_drop_handl\n", name);
  GdkAtom         target_type;
  if (context-> targets)
    {

      /* Choose the best target type */
      target_type = GDK_POINTER_TO_ATOM
        (g_list_nth_data (context-> targets, 0));

      /* Request the data from the source. */
      gtk_drag_get_data
        (
         widget,         /* will receive 'drag-data-received' signal */
         context,        /* represents the current state of the DnD */
         target_type,    /* the target type we want */
         time            /* time stamp */
         );
    }

  printf("Something dropped: [%i,%i,%i], %i\n", x, y, time,
         g_list_length(context->targets));
  return TRUE;
}


gboolean
on_main_window_drag_drop               (GtkWidget       *widget,
                                        GdkDragContext  *drag_context,
                                        gint             x,
                                        gint             y,
                                        guint            time,
                                        gpointer         user_data)
{

  return TRUE;
}


void
on_drawing3D_drag_data_received        (GtkWidget       *widget,
                                        GdkDragContext  *drag_context,
                                        gint             x,
                                        gint             y,
                                        GtkSelectionData *selection_data,
                                        guint            target_type,
                                        guint            time,
                                        gpointer         user_data)
{
  const gchar *name = gtk_widget_get_name (widget);
  g_print ("%s: drag_data_received_handl\n", name);
  glong   *_idata;
  char   *_sdata;
  /* Deal with what we are given from source */

  if((selection_data != NULL) && (selection_data-> length >= 0)){
    _sdata = (char*)selection_data-> data;
    g_print ("string: %s", _sdata);
  }
  g_print (".\n");
  string nameo(_sdata);
  // Elliminates the file:// and the \n at the end
  string name2 = nameo.substr(7, nameo.size()-7-2);
  printf("The name passed is %sa\n", name2.c_str());

  objectNames.push_back(name2);

  addObjectFromString(name2);

  /* on_drawing3D_expose_event(drawing3D,NULL, user_data); */
  /* return; */
}

