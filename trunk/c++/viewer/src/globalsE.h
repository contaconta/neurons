#ifndef GLOBAL_VARIABLES_FILEE_H_
#define GLOBAL_VARIABLES_FILEE_H_

#include <gtk/gtk.h>
#include "Neuron.h"
#include "CubeFactory.h"
#include "CloudFactory.h"
#include "GraphFactory.h"
#include "utils.h"
#include "Axis.h"
#include "DoubleSet.h"
#include "GraphCut.h"
#include "BBP_Morphology.h"

// the plugin function signatures
#include "plugin_info.h"

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
extern bool flag_draw_dual;
extern bool flag_draw_neuron;
extern int layerSpanViewZ;
extern bool drawCube_flag;
extern bool flag_minMax;
extern bool flag_cube_transparency;

extern bool flag_save_neuron_coordinate;
extern bool flag_save_cube_coordinate;


//Canvas parameters
extern GtkWidget* ascEditor;
extern GtkWidget* drawing3D;
extern GtkWidget* selectionEditor;
extern double widgetWidth;
extern double widgetHeight;

//Cube variables
extern Cube_P* cube;
extern Image<float>* img;
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
MOD_SELECT_EDITOR - select tool
*/
enum MayorMode { MOD_VIEWER,
                 MOD_ASCEDITOR,
                 MOD_SELECT_EDITOR};

extern int majorMode;

//For the dynamic camera control (with the mouse)
extern unsigned char mouse_buttons[3];
extern int mouse_last_x;
extern int mouse_last_y;
extern int mouse_current_x;
extern int mouse_current_y;
extern int mouse_startSel_x;
extern int mouse_startSel_y;

//Names of the stuff
extern string neuron_name;
extern string volume_name;

extern vector< string >    objectNames;
extern vector< VisibleE* > toDraw;
extern vector< DoubleSet<Point3D>* > lSelections;
extern vector< GraphCut<Point3D>* > lGraphCuts;

// Parameters
extern int argcp;
extern char **argvp;

// Shaders
extern GLuint shader_v; // vertex shader id
extern GLuint shader_f; // fragment shader id
extern GLuint shader_p; // program shader id

/** Plugins */

// Plugin names
extern vector<string> plugins;
extern plugin_key_press_event p_key_press_event;
extern plugin_unproject_mouse p_unproject_mouse;
extern plugin_expose          p_expose         ;
extern plugin_motion_notify   p_motion_notify  ;


/** Select tool parameters */

// const
extern int rect_sel_step_x;
extern int rect_sel_step_y;

// Select tool mode
enum SelectToolMode{
  SELTOOL_MODE_SELECT,
  SELTOOL_MODE_ADD_POINTS,
  SELTOOL_MODE_RECTANGLE,
  SELTOOL_MODE_NONE
};

enum SelectToolPointType{
  CPT_SOURCE,
  CPT_SINK
};

enum SelectToolObjectType{
  CT_SIMPLE_SELECTION=0,
  CT_GRAPHCUT
};

extern SelectToolMode selectToolMode;

extern DoubleSet<Point3D>* currentSelectionSet;

extern GraphCut<Point3D>* currentGraphCut;

#endif
