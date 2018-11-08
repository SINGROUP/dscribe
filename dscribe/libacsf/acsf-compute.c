#include "acsf.h"
#include <math.h>
#include <string.h>
#include<stdio.h>


/*! \brief Compute ACSFs for all atoms in the system.
 *
 * */
void acsf_compute_acsfs(ACSF *qm) {

	int natm = qm->natm;

	for(int i=0; i<natm; i++) {
		for(int j=0; j<natm; j++) {
			if(i==j) continue;
			acsf_compute_Gbond(qm, i, j);
			for(int k=0; k<j; k++) {
				if(k==i) continue;
				acsf_compute_Gangle(qm, i,j,k);
			}
		}
	}
}

/*! \brief Compute ACSFs for a defined set of atoms.
 *
 * */
void acsf_compute_acsfs_some(ACSF *qm, int *indices, int nidx) {

	int natm = qm->natm;
	int i;

	for(int ii=0; ii<nidx; ii++) {
		i = indices[ii];
		for(int j=0; j<natm; j++) {
			if(i==j) continue;
			acsf_compute_Gbond(qm, i, j);

			for(int k=0; k<j; k++) {
				if(k==i) continue;
				acsf_compute_Gangle(qm, i, j, k);
			}
		}
	}
}

/*! \brief Compute the G1, G2 and G3 terms for atom ai with bi
 * */
void acsf_compute_Gbond(ACSF *qm, int ai, int bi) {

	int natm = qm->natm;

	// index of type of B
	int bti = qm->typeID[qm->Z[bi]];

	// fetch distance from the matrix
	double Rij = qm->distances[ai*natm + bi]; //vectors_norm(&r);
	//printf("computing 2body... %lf\n", Rij);
	if(Rij >= qm->cutoff) return; // not tested!

	// pointers to the G storage
	double *Ga;

	//       skip atoms   skip G of other types
	//Ga = qm->G2 + ai*SG + bti*qm->nG2;

	Ga = qm->acsfs;
	Ga += ai * qm->nTypes 		* qm->nG2; //skip other atoms - G2
	Ga += ai * qm->nSymTypes 	* qm->nG3; //skip other atoms - G3
	Ga += bti * qm->nG2; // skip nG2 types that are not the ones of atom bi

	//Gb = qm->G2 + bi*SG + ati*qm->nG2;
	//printf("bond %i %i: offsets %i %i\n",ai,bi,ai*SG + bti*qm->nG2,bi*SG + ati*qm->nG2);

	double fc, val;
	fc = acsf_cutoff(qm, Rij);
	int g = 1; double Rs, eta;

	// compute G1 - first function is just the cutoffs
	Ga[0] += fc; //printf("%i %i fc %lf -- %lf\n",ai,bi,fc, Rij);

	// compute G2 - gaussian types
	double *params = qm->bond_params;

	for (int i=0; i < qm->n_bond_params; ++i) {

		eta = params[0];
		Rs = params[1];

		Ga[g] += exp(-eta * (Rij - Rs)*(Rij - Rs)) * fc;

		g++;
		params += 2;
	}

	// compute G3 - cosine type
	for(int i=0; i<qm->n_bond_cos_params; i++) {

		val = qm->bond_cos_params[i]; // the oscillation k
		val = cos(Rij*val)*fc;
		Ga[g] += val;

		g++;
	}
}

/*! \brief Compute the G4 and G5 terms for triplet i, j, k
 * */
void acsf_compute_Gangle(ACSF *qm, int i, int j, int k) {

	// index of type of B
	int typj = qm->typeID[qm->Z[j]];
	int typk = qm->typeID[qm->Z[k]];

	// fetch distance from matrix
	double Rij = qm->distances[i*qm->natm + j]; //vectors_norm(&r);
	if(Rij >= qm->cutoff) return;

	double Rik = qm->distances[i*qm->natm + k]; //vectors_norm(&r);
	if(Rik >= qm->cutoff) return;

	double Rjk = qm->distances[k*qm->natm + j]; //vectors_norm(&r);
	if(Rjk >= qm->cutoff) return;

	// size of the total Gang allocation for one atom
	int its = symm_index(typj,typk);

	double *Ga = qm->acsfs; // pointers to the G storage
	// Ga = qm->G3 + (i*qm->nSymTypes + its)*qm->nG3;

	Ga += i * qm->nTypes 		* qm->nG2; // skip other atoms - G2
	Ga += i * qm->nSymTypes 	* qm->nG3; // skip other atoms - G3
	Ga += qm->nTypes 		* qm->nG2; // skip this atoms G2
	Ga += its			* qm->nG3; // skip nG3 types that are not the ones of atom bi

	double fc = acsf_cutoff(qm, Rij)*acsf_cutoff(qm, Rik)*acsf_cutoff(qm, Rjk);
	double fc5= acsf_cutoff(qm, Rij)*acsf_cutoff(qm, Rik);

	double costheta = 0.5/(Rij*Rik);
	Rij *= Rij; //square all distances!
	Rik *= Rik;
	Rjk *= Rjk;
	costheta = costheta * (Rij+Rik-Rjk);

	int g = 0;
	double eta, gauss, zeta, lambda;
	//double twominusZ, onepluslcosth, oplc0;

	double *params = qm->ang4_params;

	// cos( theta_ijk ) = ( r_ij^2 + r_ik^2 - r_jk^2 ) / ( 2*r_ij*r_ik ),

	// computes G4 at the moment
	for (int i = 0; i < qm->n_ang4_params; ++i)
	{
		eta = params[0];
		gauss  = exp(-eta*(Rij+Rik+Rjk)) * fc;

		zeta = params[1];
		lambda = params[2];

		Ga[g] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;

		g++;
		params += 3;
	}

	params = qm->ang5_params;

	// computes G5 at the moment
	for (int i = 0; i < qm->n_ang5_params; ++i)
	{
		eta = params[0];
		gauss  = exp(-eta*(Rij+Rik)) * fc5;

		zeta = params[1];
		lambda = params[2];

		Ga[g] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;

		g++;
		params += 3;
	}
}

/*! \brief Computes the value of the cutoff fuction at a specific distance.
 * */
inline double acsf_cutoff(ACSF *qm, double Rij) {
	return (Rij<qm->cutoff)? 0.5*(cos(Rij*PI/qm->cutoff)+1) : 0;
}


