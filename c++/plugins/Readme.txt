Plugins should be placed in the plugins/bin directory. They have to be compiled as dynamic libraries and should defined the 3 following functions :
* const bool plugin_init()
* const bool plugin_exec()
* const bool plugin_quit()

Look at the example directory that contains a simple example of a plugin.
