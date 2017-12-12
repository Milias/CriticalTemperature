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
OBJ_NO_MAIN=$(filter-out $(SRCPATH)/main.o, $(OBJ))
OBJ_LIB=$(addsuffix .so, $(basename $(filter-out $(SRCPATH)/main.o, $(OBJ))))

all: $(OBJ)
	$(CC) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) $^ -o $(BINPATH)/$(BINNAME)

lib: $(OBJ_NO_MAIN)
#	ar cr libintegrals.a $(addprefix $(BINPATH)/lib, $(notdir $^))
	$(CC) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) -shared -fPIC -o $(BINPATH)/libintegrals.so $^

main: $(OBJ)
	$(CC) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) $^ -o $(BINPATH)/$(BINNAME)

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CPPINCL) -fPIC -c $^ -o $@

%.so: $(OBJ_NO_MAIN)
	$(CC) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) -shared -fPIC -o $(BINPATH)/lib$(notdir $@) $^

.PHONY: clean
clean:
	\rm -f $(BINPATH)/$(BINNAME)
	\rm -f $(BINPATH)/*.so
	\rm -f $(SRCPATH)/*.o

