# CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
# CFLAGSGL = -I/usr/include/

# LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
# LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib -L../ascParser -lGLEW -lglut -lGLU -lGL

# LDFLAGS =  -pg -g -fopenmp $(LDFLAGSOPENCV) $(LDFLAGSOPENGL) 

# CFLAGS = -c -pg -g -fopenmp -I../../neuronStackDb -I../../ascParser -I../../adaBoost -I../../volumeBoosting -I../steerableFilters2D $(CFLAGSOPENCV) $(CFLAGSGL)

# PREFIX=/home/ggonzale/

# SOURCES=


# OBJECTS=$(SOURCES:.cpp=.o)

# all: neuronToCloud  neuronStatistics

# install:
# 	ln -sf  $(PREFIX)workspace/neuronTools/neuronToCloud $(PREFIX)bin/neuronToCloud
# 	ln -sf  $(PREFIX)workspace/neuronTools/neuronStatistics $(PREFIX)bin/neuronStatistics

# depend:
# 	makedepend -Y *.{c,cpp} 2>/dev/null

# neuronStatistics: $(OBJECTS) ascParser neuronStatisticsMain
# 	g++ $(LDFLAGS) $(OBJECTS) ../ascParser/Neuron.o ../ascParser/ascParser2.o neuronStatistics.o -o neuronStatistics

# neuronStatisticsMain:
# 	g++ $(CFLAGS) neuronStatistics.cpp -o neuronStatistics.o

# neuronToCloud: $(OBJECTS) ascParser neuronToCloudMain
# 	g++ $(LDFLAGS) $(OBJECTS) ../ascParser/Neuron.o ../ascParser/ascParser2.o neuronToCloud.o -o neuronToCloud

# neuronToCloudMain:
# 	g++ $(CFLAGS) neuronToCloud.cpp -o neuronToCloud.o


# ascParser:
# 	make -C ../ascParser/

# .cpp.o:
# 	g++ $(CFLAGS) $< -o $@

# clean :
# 	rm -rf *.o neuronToCloud neuronStatistics

all:
	make -C ../../lib
	make -C ../../neuronTools
