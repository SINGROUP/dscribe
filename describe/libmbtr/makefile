CC		= g++
CPPFLAGS = -g -O3 -Wall
LDFLAGS =
OBJS = main.o cmbtr.o

cmbtr : $(OBJS)
	$(CC) $(LDFLAGS) -o cmbtr $(OBJS)

main.o : main.cpp
	$(CC) $(CPPFLAGS) -c main.cpp

cmbtr.o : cmbtr.cpp cmbtr.h
	$(CC) $(CPPFLAGS) -c cmbtr.cpp
