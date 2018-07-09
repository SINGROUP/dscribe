#include <stdio.h>
#include <stdlib.h>
#include "acsf.h"
#include <string.h>
#include <math.h>


inline int symm_index(int i, int j) {
	if(i>=j) return (i*(i+1))/2 + j;
	else return (j*(j+1))/2 + i;
}


void matrix_print_symm(double *m, int n, const char *name) {
	
	printf("matrix %s:\n",name);
	for (int i = 0; i < n; i++) {
		for(int j=0; j < n; j++) {
			printf("%lf ",m[symm_index(i,j)]);
		}
		printf("\n");
	}printf("\n");
}

void matrix_print(double *m, int n, const char *name) {
	
  printf("matrix %s:\n",name);
  for (int i = 0; i < n; i++) {
    for(int j=0; j < n; j++) {
      printf("%.3lf ",m[i*n+j]);
    }
    printf("\n");
  }printf("\n");
}

void matrix_rect_print(double *m, int row, int col, const char *name) {
	
  printf("rect matrix %s:\n",name);
  for (int i = 0; i < row; i++) {
    for(int j=0; j < col; j++) {
      printf("%.5lf ",m[i*col+j]);
    }
    printf("\n");
  }printf("\n");
}

void matrix_printerror(double *a, double *b, int n, const char *name) {
	
	printf("matrix %s:\n",name);
	for (int i = 0; i < n; i++) {
		for(int j=0; j < n; j++) {
			printf("%lf ",fabs(a[i*n+j]-b[i*n+j]));
		}
		printf("\n");
	}printf("\n");
}


void array_print(double *v, int n, const char *name) {
	
	printf("array %s:\n",name);
	for (int i = 0; i < n; i++) {
		printf("%lf ",v[i]);
	}printf("\n");
}
void farray_print(float *v, int n, const char *name) {
	
	printf("array %s:\n",name);
	for (int i = 0; i < n; i++) {
		printf("%f ",v[i]);
	}printf("\n");
}

void iarray_print(int *v, int n, const char *name) {
	
	printf("array %s:\n",name);
	for (int i = 0; i < n; i++) {
		printf("%i ",v[i]);
	}printf("\n");
}

