#include <gtk/gtk.h>


void
on_new1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_open1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save_as1_activate                   (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_quit1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_cut1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_copy1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_paste1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_delete1_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_about1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

gboolean
on_drawing3D_configure_event           (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data);

void
on_drawing3D_realize                   (GtkWidget       *widget,
                                        gpointer         user_data);

gboolean
on_drawing3D_expose_event              (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data);

gboolean
on_drawing3D_motion_notify_event       (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data);

gboolean
on_drawing3D_button_press_event        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_drawing3D_button_release_event      (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_drawing3D_key_press_event           (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data);

void
on_view_entry_changed                  (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_layer_XY_spin_changed               (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_cube_col_spin_change_value          (GtkSpinButton   *spinbutton,
                                        GtkScrollType    scroll,
                                        gpointer         user_data);

void
on_cube_row_spin_change_value          (GtkSpinButton   *spinbutton,
                                        GtkScrollType    scroll,
                                        gpointer         user_data);

void
on_draw_cube_toggle_toggled            (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_bew_dendrite_button_activate        (GtkButton       *button,
                                        gpointer         user_data);

void
on_branch_button_clicked               (GtkButton       *button,
                                        gpointer         user_data);

void
on_new_neuron_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_open_neuron_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save_neuron_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_layer_XY_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_layer_view_value_changed            (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_cube_col_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_cube_row_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_continue_segment_button_clicked     (GtkButton       *button,
                                        gpointer         user_data);

void
on_Erase_Point_clicked                 (GtkButton       *button,
                                        gpointer         user_data);

void
on_new_dendrite_button_clicked         (GtkButton       *button,
                                        gpointer         user_data);

void
on_delete_segment_clicked              (GtkButton       *button,
                                        gpointer         user_data);

void
on_ascEditor_destroy                   (GtkObject       *object,
                                        gpointer         user_data);

void
on_draw_neuron_toggled                 (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_cube_transparency_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_layerXZ_spin_value_changed          (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_layer_YZ_spin_value_changed         (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_get_matrix_button_clicked           (GtkButton       *button,
                                        gpointer         user_data);

void
on_editAsc_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_ascEditor_width_value_changed       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

gboolean
on_drawing3D_scroll_event              (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data);

void
on_release_asc_action_pressed          (GtkButton       *button,
                                        gpointer         user_data);

void
on_release_asc_action_clicked          (GtkButton       *button,
                                        gpointer         user_data);

void
on_screenshot_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_select_point_clicked                (GtkButton       *button,
                                        gpointer         user_data);

void
on_recursive_offset_closer_point_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_recursive_offset_selected_point_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_move_selected_point_clicked         (GtkButton       *button,
                                        gpointer         user_data);

void
on_move_closer_point_clicked           (GtkButton       *button,
                                        gpointer         user_data);

void
on_delete_segment_from_point_clicked   (GtkButton       *button,
                                        gpointer         user_data);

void
on_shaders_toggled                     (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_select_shaders_changed              (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_shaders_clicked                     (GtkButton       *button,
                                        gpointer         user_data);

void
on_select_shaders_changed              (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_run_graph_cuts_clicked              (GtkButton       *button,
                                        gpointer         user_data);

void
on_create_selection_clicked            (GtkButton       *button,
                                        gpointer         user_data);

void
on_save_selection_clicked              (GtkButton       *button,
                                        gpointer         user_data);

void
on_clear_selection_clicked             (GtkButton       *button,
                                        gpointer         user_data);

void
on_remove_selection_clicked            (GtkButton       *button,
                                        gpointer         user_data);

void
on_load_selection_clicked              (GtkButton       *button,
                                        gpointer         user_data);

gboolean
on_selection_mode_button_press_event   (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);
void
on_mode_point_toggled                  (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_mode_select_toggled                 (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_mode_rect_toggled                   (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_menu_plugins_activate               (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

/*
void
on_menu_plugins_submenu_activate       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);
*/

void
on_min_alpha_changed                   (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_max_alpha_changed                   (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_display_drawings_toggled            (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_cbBlendFunction_changed             (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_3dmenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_xymenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_xzmenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_yzmenu_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_combomenu_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_combomenu_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_videolayers_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_videorotation_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_videorotationtime_activate          (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_load_seeds_pressed                  (GtkButton       *button,
                                        gpointer         user_data);

void
on_open_3d_stack1_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_open_4d_stack1_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_3DLIS_D_changed                     (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_3DLIS_D_editing_done                (GtkCellEditable *celleditable,
                                        gpointer         user_data);

gboolean
on_image2_button_press_event           (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

void
on_3DLIS_FF_editing_done               (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_3DLIS_FF_changed                    (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_3DLIS_SBI_changed                   (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_3DLIS_SBI_value_changed             (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_3DLIS_SE_changed                    (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_3DLIS_SE_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_3DISL_VW_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_3DISL_VH_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_3DISL_VD_value_changed              (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_3DIS_OK_clicked                     (GtkButton       *button,
                                        gpointer         user_data);

void
on_3DLIS_C_clicked                     (GtkButton       *button,
                                        gpointer         user_data);

void
on_3DLIS_ChooseDirectory_clicked       (GtkButton       *button,
                                        gpointer         user_data);

void
on_3DLIS_ChooseDirectory_pressed       (GtkButton       *button,
                                        gpointer         user_data);

void
on_3DLIS_CD_clicked                    (GtkButton       *button,
                                        gpointer         user_data);

void
on_3DLIS_CDir_clicked                  (GtkButton       *button,
                                        gpointer         user_data);

void
on__3DLIS_saveStackB_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on__3DLIS_saveStackText_changed        (GtkEditable     *editable,
                                        gpointer         user_data);

void
on__3DLIS_saveStack_BB_clicked         (GtkButton       *button,
                                        gpointer         user_data);

void
on_buttonViewOnlyCube_toggled          (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_maxProjection_group_changed         (GtkRadioButton  *radiobutton,
                                        gpointer         user_data);

void
on_minProjection_group_changed         (GtkRadioButton  *radiobutton,
                                        gpointer         user_data);

void
on_projectionComboBox_changed          (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_open_stc_file_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

gboolean
on_drawing3D_drag_drop                 (GtkWidget       *widget,
                                        GdkDragContext  *drag_context,
                                        gint             x,
                                        gint             y,
                                        guint            time,
                                        gpointer         user_data);

gboolean
on_main_window_drag_drop               (GtkWidget       *widget,
                                        GdkDragContext  *drag_context,
                                        gint             x,
                                        gint             y,
                                        guint            time,
                                        gpointer         user_data);

void
on_drawing3D_drag_data_received        (GtkWidget       *widget,
                                        GdkDragContext  *drag_context,
                                        gint             x,
                                        gint             y,
                                        GtkSelectionData *data,
                                        guint            info,
                                        guint            time,
                                        gpointer         user_data);
