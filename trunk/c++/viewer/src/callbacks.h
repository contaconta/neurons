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
on_create_contour_clicked              (GtkButton       *button,
                                        gpointer         user_data);

void
on_save_contour_clicked                (GtkButton       *button,
                                        gpointer         user_data);

void
on_clear_contour_clicked               (GtkButton       *button,
                                        gpointer         user_data);

void
on_remove_contour_clicked              (GtkButton       *button,
                                        gpointer         user_data);

void
on_run_graph_cuts_clicked              (GtkButton       *button,
                                        gpointer         user_data);

void
on_add_contour_point_toggled           (GtkToggleButton *togglebutton,
                                        gpointer         user_data);
