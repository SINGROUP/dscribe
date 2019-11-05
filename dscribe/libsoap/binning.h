#ifndef BINNING_H
#define BINNING_H

struct pos {
  double x, y, z;
};

struct binning {
  int nx, ny, nz, ntot;
  double xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz;
  int *counts, *sizes;
  struct pos **atoms;
};

void free_binning(struct binning *self);

void init_binning(struct binning *self, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, double rcut);

void insert_atom(struct binning *self, double x, double y, double z);

int get_index(struct binning *self, int i, int j, int k);

double min(double *x, int N, int step);
double max(double *x, int N, int step);

#endif
