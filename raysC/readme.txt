1. First, you need to install openCv and copy FindOpenCV.cmake to the directory containing the cmake modules. You can download the latest file at http://opencv.willowgarage.com/wiki/Getting_started
Copy this file to /usr/share/cmake-2.x/Modules/. For example :
cp FindOpenCV.cmake /usr/share/cmake-2.6/Modules/
2. Create the makefile. To do so, type "ccmake .". 
3. You can then use the script buildmex.sh to build the library and compile the mex file. Edit buildmex.sh and if MEX_PATH and MEX_EXE are not set properly, you have to edit them manually.
4. Run buildmex.sh and voila, you are done ! ;-)
