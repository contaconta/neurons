INTERFACES = block.h graph.h
REALISATIONS = block.h graph.h example.cpp
OBJECTS       = $(REALISATIONS:.cpp=.o)
CC = g++
CFLAGS = -c -Wall -ansi -O3
LD = $(CC) -s
LDFLAGS = -L/usr/X11R6/lib
CLEAR= @clear
ECHO = @echo
EXECUTABLE = maxflow
INCLUDE = -I .
CFLAG        = -g
#CFLAG        = -O2

#
# objects
#

%.o : %.cpp %.h
	$(ECHO) "Compiling $< -> $(<:.cpp=.o)"
	$(CC) $(INCLUDE) $(CFLAG) -c $< -o $(<:.cpp=.o)

#
# binaries
#

$(EXECUTABLE) : $(OBJECTS)
	$(ECHO) "Linking"
	$(CC) $(LDFLAGS) $(OBJECTS) -o $(EXECUTABLE)

all : $(EXECUTABLE)

clr :
	$(ECHO) "Cleaning..."
	$(RM) core
	$(RM) %.o
	$(RM) $(EXECUTABLE)
	$(ECHO) "Cleaning over"

clean : clr
