#include "acsf.h"
#include <math.h>
#include <string.h>



/*! \brief Compute all ACSFs.
 * 
 * */
void acsf_compute_acsfs(ACSF *qm) {
	
	int natm = qm->natm;
	
	// reset - done in python
	// memset(qm->G2, 0, natm*qm->nTypes*qm->nG2*sizeof(double));
	// memset(qm->G3, 0, natm*qm->nSymTypes*qm->nG3*sizeof(double));
	
	for(int i=0; i<natm; i++) {
		
		for(int j=0; j<natm; j++) {
			if(i==j) continue;
			acsf_compute_Gbond(qm, i, j);
			
			for(int k=0; k<j; k++) {
				if(k==i) continue;
				//if(k==j) continue;
				//printf("computing Gang %i %i %i\n",i,j,k);
				acsf_compute_Gangle(qm, i,j,k);
			}
			
		}
	}
	
	//matrix_rect_print(qm->G2, natm, qm->nG2*qm->nTypes, "G2s");	
}


/*! \brief Compute the G_bond for atom ai with bi
 * */
void acsf_compute_Gbond(ACSF *qm, int ai, int bi) {
	
	int natm = qm->natm;
	
	// index of type of B
	int bti = qm->typeID[qm->Z[bi]];
	
	// fetch distance from the matrix
	double Rij = qm->distances[ai*natm + bi]; //vectors_norm(&r);
	printf("computing 2body... %lf\n", Rij);
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
		
		val = 0.5 * qm->bond_cos_params[i]; // the oscillation k
		val = cos(Rij*val)*fc;
		Ga[g] += val;
		
		g++;
	}
	
	
}


// atom i is the center of the 3-body term (type G4 only for now)
//
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
	
	double *Ga = qm->G3; // pointers to the G storage
	// Ga = qm->G3 + (i*qm->nSymTypes + its)*qm->nG3;

	Ga += i * qm->nTypes 		* qm->nG2; // skip other atoms - G2
	Ga += i * qm->nSymTypes 	* qm->nG3; // skip other atoms - G3
	Ga += qm->nTypes 			* qm->nG2; // skip this atoms G2
	Ga += its					* qm->nG3; // skip nG3 types that are not the ones of atom bi

	double fc = acsf_cutoff(qm, Rij)*acsf_cutoff(qm, Rik)*acsf_cutoff(qm, Rjk);
	
	double costheta = 0.5/(Rij*Rik);
	Rij *= Rij; //square all distances!
	Rik *= Rik;
	Rjk *= Rjk;
	costheta = costheta * (Rij+Rik-Rjk);
	
	int g = 0; 
	double eta, gauss, zeta, lambda;
	//double twominusZ, onepluslcosth, oplc0;
	
	double *params = qm->ang_params;

	// cos( theta_ijk ) = ( r_ij^2 + r_ik^2 - r_jk^2 ) / ( 2*r_ij*r_ik ),
	
	// computes G4 at the moment
	// TODO!!!!
	for (int i = 0; i < qm->n_ang_params; ++i)
	{
		eta = params[0];
		gauss  = exp(-eta*(Rij+Rik+Rjk)) * fc;

		zeta = params[1];
		lambda = params[2];

		Ga[g] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;


		g++;
		params += 3;
	}
	/*
	for(int e=0; e<qm->neta; e++) {
		
		eta = qm->bond_eta[e];
		gauss  = exp(-eta*(Rij+Rik+Rjk)) * fc;
		
		// loop over lambda = -1,1
		for(int lambda=-1; lambda<=1; lambda+=2) {
			
			twominusZ = 1.0;
			oplc0 = 1 + lambda*costheta;
			onepluslcosth = oplc0;
			
			for(int zi=0; zi<qm->nzeta; zi++) {
				
				Ga[g] += onepluslcosth * gauss * twominusZ;
				
				twominusZ *= 0.5;
				onepluslcosth *= oplc0;
				g++;
			}
		}
	}
	*/
	
}

/*
// atom i is the center of the 3-body term (type G4 only for now)
//
void acsf_compute_Gangle45(ACSF *qm, int i, int j, int k) {
	
	// index of type of B
	int typj = qm->typeID[qm->atoms[j].Z];
	int typk = qm->typeID[qm->atoms[k].Z];
	
	// compute distance
	double Rij = qm->distances[symm_index(i,j,qm->nAtoms)]; //vectors_norm(&r);
	if(Rij >= RCUT) return;
	
	double Rik = qm->distances[symm_index(i,k,qm->nAtoms)]; //vectors_norm(&r);
	if(Rik >= RCUT) return;
	
	double Rjk = qm->distances[symm_index(j,k,qm->nAtoms)]; //vectors_norm(&r);
	
	
	// size of the total Gang allocation for one atom
	int its = symm_index(typj,typk, qm->nTypes);
	
	double *Ga; // pointers to the G storage
	Ga = qm->G3 + (i*qm->nSymTypes + its)*qm->nG3;
	
	double fc5 = acsf_cutoff(Rij) * acsf_cutoff(Rik);
	double fc4 = acsf_cutoff(Rjk);
	
	double costheta = 0.5/(Rij*Rik);
	Rij *= Rij; //square all distances!
	Rik *= Rik;
	Rjk *= Rjk;
	costheta = costheta * (Rij+Rik-Rjk);
	
	int g = 0; double eta, gauss4, gauss5;
	double twominusZ, onepluslcosth, oplc0;
	
	// cos( theta_ijk ) = ( r_ij^2 + r_ik^2 - r_jk^2 ) / ( 2*r_ij*r_ik ),
	
	// computes G4 at the moment
	for(int e=0; e<NBONDETA; e++) {
		
		eta = qm->bond_eta[e];
		gauss5  = exp(-eta*(Rij+Rik)) * fc5;
		gauss4  = gauss5 * exp(-eta*(Rjk)) * fc4;
		
		// loop over lambda = -1,1
		for(int lambda=-1; lambda<=1; lambda+=2) {
			
			twominusZ = 1.0;
			oplc0 = 1 + lambda*costheta;
			onepluslcosth = oplc0;
			
			for(int zi=0; zi<NANGZETA; zi++) {
				
				Ga[g] += onepluslcosth * gauss4 * twominusZ;
				g++;
				Ga[g] += onepluslcosth * gauss5 * twominusZ;
				g++;
				
				twominusZ *= 0.5;
				onepluslcosth *= oplc0;
			}
		}
	}
	
	
	
}
*/


inline double acsf_cutoff(ACSF *qm, double Rij) {
	return (Rij<qm->cutoff)? 0.5*(cos(Rij*PI/qm->cutoff)+1) : 0;
}


