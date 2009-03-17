
CPP = g++
CC = g++
CFLAGS = $(shell pkg-config --cflags gmodule-2.0)
LDFLAGS = $(shell pkg-config --libs gmodule-2.0)
LD = $(CPP) -s

all : example.so

example.so : example.cpp
	$(CPP) $(CFLAGS) -fPIC -g -c -Wall example.cpp
	$(CPP) -shared -Wl,-soname,libmy_module.so -o libexample.so example.o -lc

clean :
	rm -f *.o
	rm -f *.so
