#include "acsf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


// UNUSED! remove this!
ACSF* acsf_new() {
  
	ACSF *qm = calloc(1, sizeof(ACSF));
	qm->alloc_atoms = false;
	qm->alloc_work  = false;
	
	// number of 2- and 3-body ACSFs per atom!
	//qm->nG2 = 1 + NBONDETA*NBONDRS + NBONDCOS;
	//qm->nG3 = 2*NBONDETA*NANGZETA;
	//qm->nG3 = 4*NBONDETA*NANGZETA; // for the Gang45 version
	
	/*
	qm->neta = 0;
	qm->nrs = 0;
	qm->ncos = 0;
	qm->nzeta = 0;
	*/
	
	/*
	qm->bond_eta[0] = 5.0;
	qm->bond_eta[1] = 1.0;
	qm->bond_eta[2] = 0.4;
	qm->bond_eta[3] = 0.2;
	qm->bond_eta[4] = 0.1;
	qm->bond_eta[5] = 0.06;
	qm->bond_eta[6] = 0.03;
	qm->bond_eta[7] = 0.01;
	*/
	
	return qm;
}


// UNUSED! remove this!
void acsf_free(ACSF *qm) {
  
  acsf_reset(qm);
    
  free(qm);
}



// UNUSED! remove this!
/*! \brief Precompute all interatomic distances and distance vectors.
 * First it converts positions from angstrom to bohr and stores them in
 * atom.posbohr vector.
 * Distances go in a symmetric matrix (upper triangle storage)
 * while distance vectors are stored in a full matrix of Vector3.
 * Units are supposed to be bohr at this point!
 */
void acsf_init_distances(ACSF *qm) {
	
	int natm = qm->natm;
	double dist;
	double tmp;
	
	for(int i=0;i<natm; i++) {
		for(int j=0; j<=i; j++) {
			
			int idx = symm_index(i,j);
			
			dist = 0;
			for(int c=0; c<3; c++) {
				tmp = (qm->positions[j*3+c] - qm->positions[i*3+c]);
				dist += tmp*tmp;
				//printf("%lf\n",tmp);
			}
			dist = sqrt(dist);
			qm->distances[idx] = dist;
			//printf("dist %i %i = %lf \n",i,j,dist);
		}
	}
	
	
}


// UNUSED! REMOVE THIS!
// allocates the interactors space - might change later
// fixed for multidimensional, directional states
void acsf_init(ACSF *qm) {
	//printf("initing %i...\n", qm->natm);
	int natm = qm->natm;

	// compute the total amount of atom types
	int ntyp = 0;
	int *typ = calloc(natm, sizeof(int));
	memset(qm->typeID, -1, sizeof(int)*NELEMENTS);
	
	qm->distances = malloc(sizeof(double)*(natm*(natm+1))/2);
	
	for(int i=0;i<natm; i++) {
		
		int found = 0;
		// check if Zi is already in the list
		for(int j=0;j<ntyp; j++) {
			if(qm->Z[i] == typ[j]) {
				found = 1;
				break;
			}
		}
		if(found == 0) {
			//Z was not found
			typ[ntyp] = qm->Z[i];
			qm->typeID[qm->Z[i]] = ntyp;
			ntyp++;
		}
		
	}
	
	// convert to bohr and compute distances
	acsf_init_distances(qm);
	
	//assign to types
	qm->types = typ;
	qm->nTypes = ntyp;
	qm->nSymTypes = (ntyp*(ntyp+1))/2;
	
	/*
	printf("QMNET: init - #types %i\n",qm->nTypes);
	iarray_print(qm->types, ntyp, "types");
	iarray_print(qm->typeID, 10, "typeIDs");*/
	
	/* Each pair gets its own set of G. Example:
	 * system has C, H, N, O
	 * for each C atom we get a set of
	 * Gs for C-C
	 * Gs for C-H 
	 * ...
	 * 
	 * 
	 * */
	
	// allocate 2-body Gs: for each atom, for each type, a vector of size nG2
	qm->G2  = calloc(natm*ntyp*qm->nG2,sizeof(double));
	
	// allocate 3-body Gs
	qm->G3  = calloc(natm * qm->nSymTypes * qm->nG3, sizeof(double));	

	qm->alloc_work = true;
	//printf("inited!\n");
}


// UNUSED! REMOVE THIS
/*! \brief Deallocates the atoms and everything allocated by init.
 */
void acsf_reset(ACSF *qm) {
  
  if(qm->alloc_work == true) {
		
		free(qm->types);
		free(qm->distances);
		
		free(qm->G2); free(qm->G3);
		
		qm->alloc_work = false;
	}
	
	
}
