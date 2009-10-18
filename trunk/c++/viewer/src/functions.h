
void unProjectMouse();

void setUpVolumeMatrices();

void setUpMatricesXY(int layerSpan);

void setUpMatricesYZ(int layerSpan);

void setUpMatricesXZ(int layerSpan);

void saveScreenShot(char* filename);

void init_GUI_late();

void get_world_coordinates(double &wx, double &wy, double &wz, bool change_layers = false, int z = -1);

extern "C"{
  void get_world_coordinates(double &wx, double &wy, double &wz, int x, int y, int z = -1);
}

void draw_last_point();


/** Functions related to the ascEditting.*/

void exposeAsc(GtkWidget       *widget,
               GdkEventExpose        *event,
               gpointer         user_data);

void unProjectMouseAsc(int mouse_last_x, int mouse_last_y);

bool pressMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType objectType);

bool motionMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType objectType);

bool releaseMouseSelectTool(int mouse_last_x, int mouse_last_y, SelectToolPointType pointType);

void keyPressedAsc(GtkWidget       *widget,
                   GdkEventKey        *event,
                   gpointer         user_data);

void scrollAsc(GtkWidget       *widget,
               GdkEvent        *event,
               gpointer         user_data);


