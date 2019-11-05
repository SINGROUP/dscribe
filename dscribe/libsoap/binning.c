#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "binning.h"

void free_binning(struct binning *self){
  free(self->counts); free(self->sizes);
  for(int i = 0; i < self->ntot; i++){
    free(self->atoms[i]);
  }
  free(self->atoms);
}

void init_binning(struct binning *self, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, double rcut){
  self->xmin = xmin; self->xmax = xmax;
  self->ymin = ymin; self->ymax = ymax;
  self->zmin = zmin; self->zmax = zmax;

  int nx = self->nx = (int) floor((xmax - xmin)/rcut);
  int ny = self->ny = (int) floor((ymax - ymin)/rcut);
  int nz = self->nz = (int) floor((zmax - zmin)/rcut);
  int ntot = self->ntot = nx*ny*nz;

  self->dx = (xmax - xmin)/nx;
  self->dy = (ymax - ymin)/ny;
  self->dz = (zmax - zmin)/nz;

  self->counts = (int*) malloc(ntot*sizeof(int));
  self->sizes = (int*) malloc(ntot*sizeof(int));
  self->atoms = (struct pos**) malloc(ntot*sizeof(struct pos*));

  int initial_size = 10;

  for(int i = 0; i < ntot; i++){
    self->counts[i] = 0;
    self->sizes[i] = initial_size;
    self->atoms[i] = (struct pos*) malloc(initial_size*sizeof(struct pos));
  }
}

void insert_atom(struct binning *self, double x, double y, double z){
  int i = (x - self->xmin)/self->dx;
  int j = (y - self->ymin)/self->dy;
  int k = (z - self->zmin)/self->dz;

  int idx = get_index(self, i, j, k);

  // Grow array if necessary
  if (self->counts[idx] == self->sizes[idx]){
    self->sizes[idx] *= 2.0;
    self->atoms[idx] = realloc(self->atoms[idx], self->sizes[idx]*sizeof(struct pos));
  }

  self->atoms[idx][self->counts[idx]].x = x;
  self->atoms[idx][self->counts[idx]].y = y;
  self->atoms[idx][self->counts[idx]].z = z;
  self->counts[idx] += 1;
}

int get_index(struct binning *self, int i, int j, int k){
  return k + j*self->nz + i*self->ny*self->nz;
}

double min(double *x, int N, int step){
  double res = x[0];

  for(int i = 0; i < N; i+=step){
    if (x[i] < res) res = x[i];
  }
  return res;
}

double max(double *x, int N, int step){
  double res = x[0];

  for(int i = 0; i < N; i+=step){
    if (x[i] > res) res = x[i];
  }
  return res;
}
