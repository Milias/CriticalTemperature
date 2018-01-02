CC=clang
CCPP=clang++

CFLAGS=-O3 -Wall

CPPFLAGS=-std=c++17 -pedantic -Wall -O3
CPPLIBS=-lgsl -lcblas -lm -lgmpxx -lmpfr -lgmp -ldb -larb
CPPINCL=-Iinclude

BINNAME=ctemp
BINPATH=bin
SRCPATH=src
TEMPPATH=tmp

C_SRC=$(wildcard $(SRCPATH)/*.c)
C_OBJ=$(C_SRC:.c=.c.o)

SRC=$(wildcard $(SRCPATH)/*.cpp)
OBJ=$(SRC:.cpp=.cpp.o)
OBJ_NO_MAIN=$(filter-out $(SRCPATH)/main.cpp.o, $(OBJ))

all: $(C_OBJ) $(OBJ)
	$(CCPP) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) $^ -o $(BINPATH)/$(BINNAME)

lib: $(C_OBJ) $(OBJ_NO_MAIN)
	$(CCPP) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) -shared -fPIC -o $(BINPATH)/libintegrals.so $^

main: $(C_OBJ) $(OBJ)
	$(CCPP) $(CPPFLAGS) $(CPPLIBS) $(CPPINCL) $^ -o $(BINPATH)/$(BINNAME)

%.cpp.o: %.cpp
	$(CCPP) $(CPPFLAGS) $(CPPINCL) -fPIC -c $^ -o $@

%.c.o: %.c
	$(CC) $(CFLAGS) $(CPPINCL) -fPIC -c $^ -o $@

.PHONY: clean
clean:
	\rm -f $(BINPATH)/$(BINNAME)
	\rm -f $(BINPATH)/*.so
	\rm -f $(SRCPATH)/*.o

