#include <stdio.h>

#ifndef ACSF_DEF
#define ACSF_DEF

#define FAU2FKCAL 1185.821
#define BOHR2ANG 0.529177249
#define ANG2BOHR 1.889725989
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define true 1
#define false 0

#define NELEMENTS 100



typedef struct ACSF ACSF;
struct ACSF {
	
  int natm;
  int *Z;
  
  
  double *positions;
  
	
  ///\brief Number of unique types found in the system.
  int nTypes;
  ///\brief Array of Z of each unique atom type.
  int *types;
  ///\brief Array of indexes of the atom types.
  /// Example: typeID[Z=2] is the type index of He atom.
  int typeID[NELEMENTS];

  ///\brief Number of symmetric type pairs.
  int nSymTypes;

	
  double cutoff;
	
  int n_bond_params;
  double *bond_params;

  int n_bond_cos_params;
  double *bond_cos_params;

  int n_ang4_params;
  double *ang4_params;

  int n_ang5_params;
  double *ang5_params;


	
  ///\brief Symmetric matrix with interatomic distances.
  double *distances;

  int nG2;
  int nG3;

  /*
 ///\brief Two-body ACSFs container.
 /// Structure: for each atom, one G set wrt each other atom type.
 double *G2;

 ///\brief Three-body ACSFs container.
 /// Structure: for each atom, foreach typeJ foreach typeK, one G3 set.
 double *G3;
  */	
  ///\brief Complete ACSFs container.
  /// Structure: for each atom 2-body functions are first (one G set wrt each other atom type), then 3-body for each atom (foreach typeJ foreach typeK, one G3 set).
  double *acsfs;

  /*
    int alloc_atoms;
    int alloc_work;
  */
};


// INIT FUNCTIONS
// ACSF* qmnet_new();
// void acsf_free(ACSF *qm);


// void acsf_init(ACSF *qm);
// void acsf_init_distances(ACSF *qm);
// void acsf_reset(ACSF *qm);



void acsf_compute_acsfs(ACSF *qm);
void acsf_compute_Gbond(ACSF *qm, int ai, int bi);
void acsf_compute_Gangle(ACSF *qm, int i, int j, int k);
double acsf_cutoff(ACSF *qm, double Rij);

int symm_index(int i, int j);

/*
  void qmnet_clean(QMNet *qm);
  void qmnet_free(QMNet *qm);

  void qmnet_init(QMNet *qm);
  void qmnet_init_distances(QMNet *qm);

  void qmnet_deinit(QMNet *qm);
  void qmnet_reset(QMNet *qm);
  void qmnet_presetstates(QMNet* qm);

  //void qmnet_project_state(double *state, double *rotated, Vector3 *zmol, double *workspace);

  void qmnet_readmolecule_pbc_bin(QMNet *qm, FILE *bin);


  int symm_index(int i, int j, int n);


  double qmnet_compute_energy(QMNet *qm);
  void qmnet_compute_forces(QMNet *qm);
  void qmnet_compute_forces3(QMNet *qm);
  void qmnet_compute_forces3_ang(QMNet *qm);
*/

// *** ACSF FUNCTIONS *** ******************************************* //


/* *** UTILITY FUNCTIONS *** **************************************** */
void matrix_print(double *m, int n, const char *name);
void matrix_rect_print(double *m, int row, int col, const char *name);


#endif
