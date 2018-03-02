CC = g++ -std=c++11
CFLAGS = -ansi -pedantic -Wall 
INCL = -I../include
OPT = -O3 
DEF = -DRAN2 -DNDEBUG

all:
	$(CC) $(OPT) $(CFLAGS) $(INCL) $(DEF) IsingMonteCarlo.C -o IsingMonteCarlo.x
	mv IsingMonteCarlo.x ../bin/

