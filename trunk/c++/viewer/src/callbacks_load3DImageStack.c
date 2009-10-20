#include "globalsE.h"
#include "support.h"

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


  sprintf(fileName, fileFormat, _3DLIS_layerInit);
  IplImage* pepe = cvLoadImage(fileName,0);
  Cube<uchar, ulong>* cubeN = new Cube<uchar, ulong>
    (pepe->width, pepe->height, _3DLIS_layerEnd-_3DLIS_layerInit+1,
     _3DLIS_voxelWidth, _3DLIS_voxelHeight, _3DLIS_voxelDepth);

  for(int i = _3DLIS_layerInit; i <= _3DLIS_layerEnd; i++){
    sprintf(fileName, fileFormat, i);
    printf("%s\n", fileName);
    IplImage* pepe = cvLoadImage(fileName,0);
    for(int y = 0; y < pepe->height; y++)
      for(int x = 0; x < pepe->width; x++)
        cubeN->put(x,y,i-_3DLIS_layerInit,
                  ((uchar *)(pepe->imageData + y*pepe->widthStep))[x]);

  }
  cube = cubeN;
  cubeN->v_r = 1.0;
  cubeN->v_g = 1.0;
  cubeN->v_b = 1.0;
  cubeN->load_texture_brick(cubeRowToDraw, cubeColToDraw);
  objectNames.push_back("newCube.nfo");
  toDraw.push_back(cubeN);
  nCubes++;
  gtk_widget_destroy(_3DLIS);
}


