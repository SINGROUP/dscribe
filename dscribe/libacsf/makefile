
CC		= gcc -std=gnu99 -O3 -Wall
CCFLAGS	= -fPIC


all:
	$(CC) $(CCFLAGS) -shared -o libacsf.so acsf-utils.c acsf-compute.c -lm
