CC=gcc
CPP=g++
CFLAGS=-c -Wall -fPIC -m32 -O3
# CFLAGS=-c -Wall -fPIC -m32 -g
CXXFLAGS=
LDFLAGS=-shared -m32
CSOURCES=
CXXSOURCES=											\
	mwu_dynamic.cpp
OBJECTS=$(CXXSOURCES:%.cpp=%.o) $(CSOURCES:%.c=%.o)
LIBRARYNAME=libmwu.so

cleanlib: clean library

library: $(LIBRARYNAME) 

$(LIBRARYNAME): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

.cpp.o: 
	$(CPP) $(CFLAGS) $(CXXFLAGS) $< -o $@

.PHONY: clean cleanobj cleanso

clean: cleanobj cleanso

cleanso: 
	rm -f $(LIBRARYNAME)

cleanobj: 
	rm -f *.o
