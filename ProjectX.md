## Channels ##
  * convert features so that they work as Channels
  * make sure haar features work for channels of different data types
  * strings -> 'C1\_W1x32434blah'

## Mem Daemon ##
  * make a matlab replacement for memdaemon for computers we cannot compile on
  * add a switch in settings to change between memdaemon and matlab equivelent

## C++ implementation of classification function ##
  * we wrote feature\_response, needs to be debugged/tested
  * we made changes to memdaemon:  getRow (changed name), getElement (added)
  * we need to write functions for: decision stumps, decision trees, strong classifier applied to a single example, strong classifier applied to many examples
  * eventually modify c++ to use data from matlab matrix when necessary

## Bootstrapping ##
  * bootstrapping for soft cascades can be complicated, need to worry about how and when to recollect data for training & validation
  * Data recollection -> use Aurelien's resize polygon so we can adjust size of regions selected, and can properly select boundary pixels.

## Compatibility across computers ##
  * change buildmex.sh and paths in settings.m so that compiled code is sent to /temp directory. this way we can compile from ICFILER and executables will be stored on the local machine.