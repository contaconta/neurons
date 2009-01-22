#ifndef GLOBAL_VARIABLES_FILEE_H_
#define GLOBAL_VARIABLES_FILEE_H_

#include "Neuron.h"
#include "CubeFactory.h"
#include "CloudFactory.h"
#include "GraphFactory.h"
#include "utils.h"
#include "Axis.h"
#include "Contour.h"

//Camera parameters
extern double fovy3D;
extern double aspect3D;
extern double zNear3D;
extern double zFar3D;
extern double disp3DX;
extern double disp3DY;
extern double disp3DZ;
extern double rot3DX;
extern double rot3DY;

// Drawing controls
extern bool flag_draw_3D;
extern bool flag_draw_XY;
extern bool flag_draw_XZ;
extern bool flag_draw_YZ;
extern bool flag_draw_combo;
extern bool flag_draw_neuron;
extern int layerSpanViewZ;
extern bool drawCube_flag;
extern bool flag_minMax;
extern bool flag_cube_transparency;

extern bool flag_save_neuron_coordinate;
extern bool flag_save_cube_coordinate;


//Canvas parameters
extern GtkWidget* drawing3D;
extern double widgetWidth;
extern double widgetHeight;

//Cube variables
extern Cube_P* cube;
extern int cubeColToDraw;
extern int cubeRowToDraw;
extern float layerToDrawXY;
extern float layerToDrawXZ;
extern float layerToDrawYZ;

extern double wx, wy, wz;

//Neuron Variables
extern Neuron* neuronita;
extern NeuronPoint* last_point;
extern NeuronSegment* current_segment;

//Control of the editor
/** Actions done on the points of the neuron|: (with last_point).
NPA_SELECT = select the last point
NPA_APPLY_RECURSIVE_OFFSET = from the point, move all the following points and the sons
             towards wherever
NPA_APPLY_RECURSIVE_OFFSET_CLOSEST_POINT_TO_CLICK = idem but from the closest point
             to the click
NPA_CHANGE_POINT_CLOSEST_TO_CLICK = moves the closest point to the mouse position
NPA_CHANGE_POINT = changes the selected point to the click
 */

enum NeuronPointActions{
  NPA_SELECT,
  NPA_APPLY_RECURSIVE_OFFSET,
  NPA_APPLY_RECURSIVE_OFFSET_CLOSEST_POINT_TO_CLICK,
  NPA_CHANGE_POINT_CLOSEST_TO_CLICK,
  NPA_CHANGE_POINT,
  NSA_ADD_POINTS,
  NSA_ADD_DENDRITE,
  NSA_ADD_BRANCH,
  NSA_ERASE_POINT,
  NSA_ERASE_WHOLE_SEGMENT,
  NSA_ERASE_SEGMENT_FROM_HERE,
  NSA_CONTINUE_SEGMENT,
  NPA_NONE
};

extern int  ascEditor_action;
extern bool flag_verbose;
extern float asc_point_width;
extern bool ascEditor2D;


/** According to the mode the key-bindings and the actions to take with the unprojected mouse position change.
Modes:
MOD_VIEWER ---- the mode per default.
MOD_ASCEDITOR - to edit asc files
*/
enum MayorMode { MOD_VIEWER,
                 MOD_ASCEDITOR,
                 MOD_CONTOUREDITOR};

extern int majorMode;

//For the dynamic camera control (with the mouse)
extern unsigned char mouse_buttons[3];
extern int mouse_last_x;
extern int mouse_last_y;
extern int mouse_current_x;
extern int mouse_current_y;

//Names of the stuff
extern string neuron_name;
extern string volume_name;

extern vector< string >    objectNames;
extern vector< VisibleE* > toDraw;
extern vector< Contour<Point>* > lContours;


// Parameters
extern int argcp;
extern char **argvp;

// Shaders
extern GLuint shader_v; // vertex shader id
extern GLuint shader_f; // fragment shader id
extern GLuint shader_p; // program shader id

// Contours
enum ContourPointActions{
  CPA_SELECT,
  CPA_ADD_POINTS,
  CPA_NONE
};

extern ContourPointActions contourEditor_action;

extern Contour<Point>* currentContour;


#endif
