#include "globalsE.h"
#include "support.h"
#include "Configuration.h"
#include "functions.h"

Cube<uchar, ulong>* loadImageStackFromSFC
(string directory, string imageFormat, int layerInit, int layerEnd,
 float voxelWidth, float voxelHeight, float voxelDepth)
{
  char fileFormat[1024];
  sprintf(fileFormat, "%s/%s", directory.c_str(), imageFormat.c_str());

  char fileName[1024];
  sprintf(fileName, fileFormat, layerInit);
  IplImage* pepe = cvLoadImage(fileName,0);
  Cube<uchar, ulong>* cubeN = new Cube<uchar, ulong>
    (pepe->width, pepe->height, layerEnd-layerInit+1,
     voxelWidth, voxelHeight, voxelDepth);

  for(int i = layerInit; i <= layerEnd; i++){
    sprintf(fileName, fileFormat, i);
    printf("%s\n", fileName);
    IplImage* pepe = cvLoadImage(fileName,0);
    for(int y = 0; y < pepe->height; y++)
      for(int x = 0; x < pepe->width; x++)
        cubeN->put(x,y,i-layerInit,
                  ((uchar *)(pepe->imageData + y*pepe->widthStep))[x]);

  }
  cubeN->v_r = 1.0;
  cubeN->v_g = 1.0;
  cubeN->v_b = 1.0;
  cubeN->load_texture_brick(cubeRowToDraw, cubeColToDraw);
  nCubes++;


  printf("Directory %s\nFormat %s\nLayerInit %i\n"
         "LayerEnd %i\n voxelWidth %f\n voxelHeight %f\n"
         "voxelDepth %f\n",
         directory.c_str(), imageFormat.c_str(), layerInit,
         layerEnd, voxelWidth, voxelHeight,
         voxelDepth);

  cube = cubeN;

  return cubeN;
}


void
on_3DLIS_D_changed                     (GtkEditable     *editable,
                                        gpointer         user_data)
{
  gchar* caracteres = gtk_editable_get_chars(editable,0,-1);
  _3DLIS_directory = caracteres;
}


void
on_3DLIS_FF_changed                    (GtkEditable     *editable,
                                        gpointer         user_data)
{
  gchar* caracteres = gtk_editable_get_chars(editable,0,-1);
  _3DLIS_format     = caracteres;
}


void
on_3DLIS_SBI_value_changed             (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  _3DLIS_layerInit = gtk_spin_button_get_value(GTK_SPIN_BUTTON(spinbutton));
}



void
on_3DLIS_SE_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  _3DLIS_layerEnd = gtk_spin_button_get_value(GTK_SPIN_BUTTON(spinbutton));
}


void
on_3DISL_VW_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  _3DLIS_voxelWidth = gtk_spin_button_get_value(GTK_SPIN_BUTTON(spinbutton));
}


void
on_3DISL_VH_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  _3DLIS_voxelHeight = gtk_spin_button_get_value(GTK_SPIN_BUTTON(spinbutton));
}


void
on_3DISL_VD_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
  _3DLIS_voxelDepth = gtk_spin_button_get_value(GTK_SPIN_BUTTON(spinbutton));
}

void
on_3DLIS_C_clicked                     (GtkButton       *button,
                                        gpointer         user_data)
{
  gtk_widget_destroy(_3DLIS);
}


void
on_3DLIS_CDir_clicked                  (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkWidget *dialog;
  dialog = gtk_file_chooser_dialog_new ("Choose the directory where the files are stored",
                                        (GtkWindow*)_3DLIS,
                                        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
                                        GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                                        GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
                                        NULL);
  if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
    {
      char *filename;
      filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
      _3DLIS_directory = filename;
      GtkWidget* entry = lookup_widget(_3DLIS, "entry_3DLIS_D");
      gtk_entry_set_text((GtkEntry*)entry , filename);
      g_free (filename);
    }
  gtk_widget_destroy (dialog);

}


void
on__3DLIS_saveStackB_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
  _3DLIS_saveStack = !_3DLIS_saveStack;
  printf("_3DLIS_saveStack = %i\n", _3DLIS_saveStack);
}


void
on__3DLIS_saveStackText_changed        (GtkEditable     *editable,
                                        gpointer         user_data)
{
  gchar* caracteres = gtk_editable_get_chars(editable,0,-1);
  _3DLIS_saveName     = caracteres;
}


void
on__3DLIS_saveStack_BB_clicked         (GtkButton       *button,
                                        gpointer         user_data)
{
  GtkWidget *dialog;
  dialog = gtk_file_chooser_dialog_new ("Choose the file where to store the configuration",
                                        (GtkWindow*)_3DLIS,
                                        GTK_FILE_CHOOSER_ACTION_SAVE,
                                        GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                                        GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
                                        NULL);
  if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
    {
      char *filename;
      filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
      _3DLIS_saveName = filename;
      GtkWidget* entry = lookup_widget(_3DLIS, "entry_3DLIS_saveStackText");
      gtk_entry_set_text((GtkEntry*)entry , filename);
      g_free (filename);
    }
  gtk_widget_destroy (dialog);
}




// All the rest was just preparation for the big hit which is in here.
void
on_3DIS_OK_clicked                     (GtkButton       *button,
                                        gpointer         user_data)
{
  printf("Directory %s\nFormat %s\nLayerInit %i\n"
         "LayerEnd %i\n voxelWidth %f\n voxelHeight %f\n"
         "voxelDepth %f\n",
         _3DLIS_directory.c_str(), _3DLIS_format.c_str(), _3DLIS_layerInit,
         _3DLIS_layerEnd, _3DLIS_voxelWidth, _3DLIS_voxelHeight,
         _3DLIS_voxelDepth);

  if(!directoryExists(_3DLIS_directory)){
    printf("3DLIS::the directory %s does not exist, the operation is canceled\n",
           _3DLIS_directory.c_str());
    return;
  }

  char fileName[1024];
  char fileFormat[1024];
  sprintf(fileFormat, "%s/%s", _3DLIS_directory.c_str(), _3DLIS_format.c_str());
  for(int i = _3DLIS_layerInit; i <= _3DLIS_layerEnd; i++){
    sprintf(fileName, fileFormat, i);
    if(!fileExists(fileName)){
      printf("3DLIS::the file %s does not exist, the operation is canceled\n",
             fileName);
      return;
    }
  }

  //And now we save the file
  if(_3DLIS_saveStack){
    string extension = getExtension(_3DLIS_saveName);
    if(extension != "stc"){
      printf("3DLIS::the file %s does not have the extension stc, cancelling\n",
             _3DLIS_saveName.c_str());
      return;
    }
    Configuration* conf = new Configuration();
    conf->add("directory", _3DLIS_directory);
    conf->add("format",_3DLIS_format);
    conf->add("layerInit",_3DLIS_layerInit);
    conf->add("layerEnd",_3DLIS_layerEnd);
    conf->add("voxelWidth",_3DLIS_voxelWidth);
    conf->add("voxelHeight",_3DLIS_voxelHeight);
    conf->add("voxelDepth",_3DLIS_voxelDepth);
    conf->add("saveStack",_3DLIS_saveStack);
    string saveFile = _3DLIS_directory + "/" + getNameFromPath(_3DLIS_saveName);
    printf("Saving the file in %s\n", saveFile.c_str());
    conf->saveToFile(saveFile);
  }

  //call the function
  Cube<uchar, ulong>* cube = loadImageStackFromSFC
    (_3DLIS_directory, _3DLIS_format, _3DLIS_layerInit, _3DLIS_layerEnd,
     _3DLIS_voxelWidth, _3DLIS_voxelHeight, _3DLIS_voxelDepth);
  toDraw.push_back(cube);

  gtk_widget_destroy(_3DLIS);
}


void
on_open_stc_file_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
  printf("Here0\n");
  GtkWidget *dialog;
  dialog = gtk_file_chooser_dialog_new ("Choose file",
                                        NULL,
                                        GTK_FILE_CHOOSER_ACTION_SAVE,
                                        /* NULL); */
                                        GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                                        GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
                                        NULL);
  gtk_widget_show (dialog);
  char *filename;
  if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
    {
      filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));

    }
  gtk_widget_destroy (dialog);

  /* Configuration* conf = new Configuration(filename); */

  /* Cube<uchar, ulong>* cube = loadImageStackFromSFC */
    /* (conf->retrieve("directory"), */
     /* conf->retrieve("format"), */
     /* conf->retrieveInt("layerInit"), conf->retrieveInt("layerEnd"), */
     /* conf->retrieveFloat("voxelWidth"), conf->retrieveFloat("voxelHeight"), */
     /* conf->retrieveFloat("voxelDepth")); */
  /* toDraw.push_back(cube); */

  addObjectFromString(filename);
  on_drawing3D_expose_event(drawing3D,NULL, NULL);
  g_free (filename);
}


