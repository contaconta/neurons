## Comments included in the makefile to understand this criptic language
SHELL=/bin/bash

## Compilation options (at some points this should be done automatically)

# Compiler to use - if you want to use openmp you neefd g++ > 4.2
# if the flags are 1, you enable the use of the libraries
# libraries required:
#    viva:  opencv, gsl, opengl
#    viewer: gtk, gtkglext, glew
#    tools:  the same as viva
#CPP = /opt/intel/cc/10.0.023/bin/icc 
CPP = g++-4.3
RELEASE=1
USE_OPENMP=1
USE_GLEW=1
USE_BBP=0

## Compilation flags
ifeq ($(USE_OPENMP),1)
  OPENMP = -openmp
  OPENMPL =  -lpthread -lgomp
  D_OPENMP = -D WITH_OPENMP
else
  OPENMP =
  OPENMPL =
  D_OPENMP =
endif

ifeq ($(USE_GLEW),1)
  D_GLEW = -D WITH_GLEW
  GLEWL = -lGLEW
else
  D_GLEW =
  GLEWL =
endif

## Interface with the BBP_SDK for visualization of their morphologies
ifeq ($(USE_BBP),1)
  D_BBP = -D WITH_BBP
  CFLAGSBBP  = -I/usr/local/include/boost-1_38 -I/usr/local/include/boost-1_38
  LDFLAGSBBP = -L/usr/local/lib -lboost_system-gcc43-mt -lboost_serialization-gcc43-mt -lBBP-SDK
else
  D_BBP =
  CFLAGSBBP  =
  LDFLAGSBBP =
endif


CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
CFLAGSGTK    = $(shell pkg-config --cflags gtk+-2.0)
CFLAGSGTKEXT    = $(shell pkg-config --cflags gtkglext-1.0 pangoft2)
CFLAGSGL = -I/usr/include/ -I/usr/include/GL
CFLAGSPLUGINS = $(shell pkg-config --cflags gmodule-2.0)

LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = $(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = $(shell pkg-config --libs gtkglext-1.0 pangoft2)
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib $(GLEWL) -lglut -lGLU -lGL
LDFLAGSPLUGINS = $(shell pkg-config --libs gmodule-2.0)

INCLUDEFLAGSEXT = -I../../../external/graphCuts
INCLUDEFLAGS = -I../../../core $(D_OPENMP) $(D_GLEW) $(D_BBP) $(CFLAGSOPENCV) $(CFLAGSGL) $(INCLUDEFLAGSEXT) $(CFLAGSPLUGINS) $(CFLAGSBBP)
LINKFLAGS    = -L../../../lib $(LDFLAGSOPENCV)  $(LDFLAGSOPENGL) $(OPENMPL) $(LDFLAGSBBP)

ifeq ($(RELEASE), 1)
 	CFLAGS = -fPIC -w -c -O3  $(OPENMP)  $(INCLUDEFLAGS)
#	CFLAGS = -w -c -O3  $(OPENMP)  $(INCLUDEFLAGS)
	LDFLAGS =  -O3 $(LINKFLAGS)
else
 	CFLAGS = -fPIC -w -c -g -pg  $(OPENMP)  $(INCLUDEFLAGS)
#	CFLAGS = -w -c -g -pg  $(OPENMP)  $(INCLUDEFLAGS)
	LDFLAGS =  -g -pg $(LINKFLAGS)
endif

## Relevant directories (I am dislexic)

OBJ_DIR = ../../../objects/
SRC_DIR =
BIN_DIR = ../../../plugins/bin/

#Files for the plugins
SRC_N_P  = $(shell ls $(SRC_DIR)*.cpp)
SRC_N    = $(subst $(SRC_DIR),,$(SRC_N_P))
EXEC_P   = $(SRC_N:.cpp=)
EXEC     = $(addprefix $(BIN_DIR),$(EXEC_P))
SRC      = $(addprefix $(SRC_DIR),$(SRC_N))
OBJ_T    = $(SRC:.cpp=.o)
OBJ      = $(addprefix $(OBJ_DIR),$(OBJ_T))



# For the dependencies
%.d: %.cpp
	$(CPP) -M $(INCLUDEFLAGS) $(CFLAGSGTK) $< > $@.tmp
	sed 's;$(@F:.d=.o);$(@:.d=.o) $@;' $@.tmp > $@
	rm -f $@.tmp


############ THE RULES START HERE #########################

# This will compile all the library
all: $(OBJ) $(EXEC)

include  $(ALL_SOURCES:.cpp=.d)

# Compilation of the plugins
$(OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $(CFLAGSPLUGINS) $(CFLAGSGTK) $(CFLAGSGTKEXT)  $< -o $@

$(EXEC): bin/%:$(OBJ_DIR)$(SRC_DIR)%.o lib/libviva.a
	$(CPP) -shared -Wl $(LDFLAGS) $(LDFLAGSPLUGINS) $(OBJ_DIR)$(SRC_DIR)$(@F).o -o $@ -lviva
ifndef VIVA_NO_LINKS
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)
endif

cleanDepend:
	rm -f *.d


clean :
	#rm -f bin/*

