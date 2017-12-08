CC=clang++

CPPFLAGS=-std=c++17 -pedantic -Wall -O3
CPPLIBS=-lgsl -lcblas -lm -lgmpxx -lmpfr -lgmp
CPPINCL=-Iinclude

BINNAME=ctemp
BINPATH=bin
SRCPATH=src
TEMPPATH=tmp

SRC=$(wildcard $(SRCPATH)/*.cpp)
OBJ=$(SRC:.cpp=.o)

all: $(OBJ)
	$(CC) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) $^ -o $(BINPATH)/$(BINNAME)

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CPPINCL) -c $^ -o $@

.PHONY: clean
clean:
	\rm $(BINPATH)/$(BINNAME)
	\rm $(SRCPATH)/*.o
