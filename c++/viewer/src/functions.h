void unProjectMouse();



void setUpVolumeMatrices();

void setUpMatricesXY(int layerSpan);

void setUpMatricesYZ(int layerSpan);

void setUpMatricesXZ(int layerSpan);

void get_world_coordinates(double &wx, double &wy, double &wz, bool change_layers = false);

void get_world_coordinates(double &wx, double &wy, double &wz, int x, int y);

void draw_last_point();


/** Functions related to the ascEditting.*/

void exposeAsc(GtkWidget       *widget,
               GdkEventExpose        *event,
               gpointer         user_data);

void unProjectMouseAsc(int mouse_last_x, int mouse_last_y);

bool unProjectMouseContour(int mouse_last_x, int mouse_last_y, ContourPointType pointType);

void keyPressedAsc(GtkWidget       *widget,
                   GdkEventKey        *event,
                   gpointer         user_data);

void scrollAsc(GtkWidget       *widget,
               GdkEvent        *event,
               gpointer         user_data);
