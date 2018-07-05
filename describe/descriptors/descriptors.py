#!/usr/bin/env python3
#
#
#

### DEFINE ###

import os, sys, time
import numpy as np
import scipy, scipy.stats, scipy.integrate
from scipy.spatial.distance import cdist, pdist
import ase, ase.io
import matplotlib.pyplot as plt
import itertools

class Atoms_descriptors:
    """class of an atoms object which additionally stores other information
    """
    def __init__(self, name, atomsobject=""):
        """Initialises with an atomsobject. If not given the atoms data can be read later.
        Positions, atom types, nuclear charges are set, as well as the single atom objects.
        """
        self.name = name
        self.var = 0.15
        self.accuracy = 0.01
        self.cutoff = 6 # cutoff distance
        self.positions = ""
        self.cell = ""
        self.atomtypes = ""
        self.nuclear_charges = ""
        self.atompairs = ""
        self.cmat = ""
        self.dmat = ""
        self.maximum = ""
        self.totalrdf = ""
        self.rdf_atom_type_lst = []
        self.rdf_type_lst = []
        self.intrdf = ""
        self.intrdf_type_lst = []
        self.sqfit = ""
        self.properties = {} # empty property dictionary
        self.atom_objects = []
        if atomsobject:
            self.atoms = atomsobject
            self.positions = atomsobject.get_positions()
            self.cell = atomsobject.get_cell()
            self.atomtypes = atomsobject.get_chemical_symbols()
            self.nuclear_charges = atomsobject.get_atomic_numbers()
            self.atompairs = self.get_pairs(atomsobject)
            for i in range(len(self.atomtypes)):
                self.atom_objects.append(Atom_descriptors(parent = self, name = self.name, idnumber = i, position = self.positions[i, :], atomtype = self.atomtypes[i], nuclcharge = self.nuclear_charges[i]))
        return
    def __str__(self):
        return str(vars(self))

    ### helper methods ###

    def jsonify():
        self.__dict__
        return

    def unjsonify():
        return

    def calc_dmat(self):
        """Returns distance matrix if not existing
        """
        if str(self.dmat)=="":
            self.dmat = cdist(self.positions, self.positions)
            self.maximum = self.dmat.max() + 5
        return self.dmat

    def gaussian_dist(self, mean, variance,):
        """norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        """
        convolving_term = scipy.stats.norm(mean,variance)
        xconv = np.linspace(-50*variance, 50*variance, 1000)
        yconv = convolving_term.pdf(xconv)
        a = np.trapz(yconv, dx = self.accuracy)
        return yconv / a

    def do_convolution(self, dmat, variance, accuracy, maximum):
        """ Helper function for Gaussian smearing. takes a distance matrix together with variance and maximum value. Returns a plot from 0 to maximum.
        """
        t0_conv = time.time()
        dmat = dmat[np.nonzero(dmat)]
        hist = np.histogram(dmat, int(maximum / self.accuracy), range=(0, maximum))
        hist_intens = hist[0]
        gaussian = self.gaussian_dist(0, variance,)
        rdf = np.convolve(hist_intens, gaussian, mode="same")
        t1_conv = time.time()
        print("Convolution time:" + str(t1_conv - t0_conv))
        return rdf

    def get_pairs(self, atomsobject):
        """Helper function to get unique pairs of atom types. Takes an atoms object and
        returns pairs in a list.
        """
        #atom_types = atomsobject.get_chemical_symbols()
        #atom_types = np.array(atom_types)
        set_atom_types = set(self.atomtypes)
        pairs_atom_types = itertools.combinations_with_replacement(set_atom_types, 2)
        lst_of_pairs = list(pairs_atom_types)
        return lst_of_pairs

    ### other methods ###

    def read_xyz(self, xyzfilename):
        """Takes a xyz filename. Returns None.
        """
        self.atoms = ase.io.read(xyzfilename)
        self.positions = self.atoms.get_positions()
        self.atomtypes = self.atoms.get_chemical_symbols()
        self.nuclear_charges = self.atoms.get_atomic_numbers()
        self.atompairs = self.get_pairs(self.atoms)
        for i in self.atom_types:
            self.atom_objects.append(Atom_descriptors(idnumber = i, position = self.positions[i, :], atomtype = self.atomtypes[i], nuclcharge = self.nuclear_charges[i]))
        return
    
    def fit2_ax2(self, function):
        """Takes a 1D function which is fitted to a * x ** 2.
        Returns a. """
        x = np.arange(0, self.maximum, step = self.accuracy)
        x = x[:len(function)]
        y = function
        self.sqfit = np.polynomial.polynomial.polyfit(x, y, deg=[2], rcond=None, full=False, w=None)
        print(self.sqfit)
        return self.sqfit

    ### rdf methods ###

    def write_rdf(self, function, typerdf=False, appendix = ""):
        """Takes an existing rdf or intrdf function (typepairs if necessary) and writes a 1D-array together with variables information into a file. Returns None.
        """
        x = np.arange(0, self.maximum, step = self.accuracy)
        if typerdf:
            rdf, typepairs = function
            for (single_rdf, pair) in zip(rdf, typepairs):
                np.savetxt(self.name + appendix + "_" + str(pair[0]) +  "_" + str(pair[1]) + ".dat", single_rdf, header = "Maximum = " + str(self.maximum) + " Accuracy = " + str(self.accuracy) + " Variance = " + str(self.var) + " type 1: " + str(pair[0]) +  " type 2: " + str(pair[1]))
                plt.plot(x[:len(single_rdf)], single_rdf, "-")
                plt.savefig(self.name + appendix + "_" + str(pair[0]) + "_" + str(pair[1]) + ".png")
                plt.close()
        else:
            rdf = function
            np.savetxt(self.name + appendix + ".dat", rdf, header = "total rdf " + "Maximum = " + str(self.maximum) + " Accuracy = " + str(self.accuracy) + " Variance = " + str(self.var))
            plt.plot(x[:len(rdf)], rdf, "-")
            plt.savefig(self.name + appendix + ".png")
            plt.close()
        return None

    def get_rdf(self, smearing=False,):
        """Returns the normalised radial distribution function (not volume-scaled).
        Can be either discrete or gaussian smeared.
        """
        self.calc_dmat()
        print(self.dmat)
        distances = np.triu(self.dmat, k=1)
        distances = distances[np.nonzero(distances)]
        if smearing:
            rdf = self.do_convolution(distances, self.var, self.accuracy, self.maximum)
            npairs = len(self.dmat)
            a = np.trapz(rdf / npairs, dx = self.accuracy)
            self.totalrdf = rdf / npairs
        else:
            hist = np.histogram(distances, int(self.maximum / self.accuracy), range=(0, self.maximum))
            self.totalrdf = hist[0]  # only take the y values
        return self.totalrdf

    def get_typerdf(self, smearing=False,):
        """Returns the normalised radial distribution function (not volume-scaled) of every combination of two atom types.
        Can be either discrete or gaussian smeared.
        !!!
        CURRENTLY NOT WORKING
        """
        self.calc_dmat()
        lst_of_types = self.get_pairs(self.atoms)
        for (a1,a2) in lst_of_types:
            dmatcopy = self.dmat.copy()
            x_list = self.atomtypes == a1
            #dmatcopy = dmatcopy[x_list, :]
            y_list = self.atomtypes == a2
            print(x_list, y_list)
            #dmatcopy = dmatcopy[:, y_list]
            dmatcopy = dmatcopy[x_list, y_list]
            if dmatcopy==0.0:

                self.rdf_type_lst.append(np.zeros(len(np.arange(0, self.maximum, step= self.accuracy))))
                self.rdf_atom_type_lst.append((a1,a2))
                break
            if a1 == a2:
                print(dmatcopy)
                dmatcopy = np.triu(dmatcopy, k=1)
            dmatcopy = dmatcopy[np.nonzero(dmatcopy)]
            submat = dmatcopy
            npairs = len(subdmat)
            if smearing:
                rdf = do_convolution(subdmat, self.var, self.accuracy, self.maximum)
                rdf = rdf / npairs
            else:
                hist = np.histogram(np.nonzero(subdmat), int(self.maximum / self.accuracy), range=(0, self.maximum))
                rdf = hist[0]
            self.rdf_type_lst.append(rdf,)
            self.rdf_atom_type_lst.append((a1,a2))
        return self.rdf_type_lst, self.rdf_atom_type_lst

    def get_intrdf(self, typerdf=False):
        """Returns the integral of the given radial distribution function.
        Can be either discrete or gaussian smeared.
        """
        if typerdf==False:
            rdf = self.totalrdf
        else:
            rdf = self.rdf_type_lst
        if list(rdf)==[] or str(rdf)=="":
            return "Calculate rdf first"
        if typerdf:
            for single_rdf in rdf[0]:
                x = np.arange(0, self.maximum, step = self.accuracy)
                x = x[:len(single_rdf)]
                intrdf = scipy.integrate.cumtrapz(single_rdf, x, axis=-1, initial=0)
                self.intrdf_type_lst.append(intrdf)
            return self.intrdf_type_lst, self.rdf_atom_type_lst
        else:
            x = np.arange(0, self.maximum, step = self.accuracy)
            x = x[:len(rdf)]
            self.intrdf = scipy.integrate.cumtrapz(rdf, x, dx = self.accuracy, axis=-1, initial=0)
            return self.intrdf


    ### cmat methods ###

    def get_cmat(self, dmat="", nuclcharges="", exponent = 2.4, size=100):
        """Building up on the distance matrix and the charges, the coulomb matrix is
        returned 
        Cij = 0.5 Zi**exponent      | i = j
            = (Zi*Zj)/(Ri-Rj)	| i != j
        The coulomb matrix is broadened by invisible atoms, means inserting 0-rows and columns
        until "size" is reached.
        """
        t0_get_cmat = time.time()

        self.calc_dmat()
        if str(dmat)=="":
            dmat = self.dmat
        if nuclcharges=="":
            nuclcharges = self.nuclear_charges

        zi = zj = nuclcharges
        repulsion = np.dot(zi[:,None],zj[None,:])

        np.seterr(divide="ignore")
        cmat = np.multiply(repulsion, 1 / dmat)
        di = np.diag_indices_from(cmat)
        cmat[di] = 0.5 * zi ** exponent # parameter
        zeros = np.zeros((size,size))
        zeros[:cmat.shape[0], :cmat.shape[1]] = cmat
        cmat = zeros # invisible atoms have been added
        self.cmat = cmat
        t1_get_cmat = time.time()
        dt_get_cmat = t1_get_cmat - t0_get_cmat
        #print("Time needed to get coulomb matrix:")
        #print(dt_get_cmat)
        return self.cmat

    def get_sinemat(self, nuclcharges="", exponent= 2.4, size=100):
        """Building up on the atomic positions and the charges, 
        the sine (coulomb) matrix is returned 
        Cij = 0.5 Zi**exponent      | i = j
            = (Zi*Zj)/phi(Ri, Rj)   | i != j
        with phi(r1, r2) = | B * sum(k = x,y,z)[ek * sin^2(pi * ek * B^-1 (r2-r1))] |
        (B is the matrix of basis cell vectors, ek are the unit vectors)
        The coulomb matrix is broadened by invisible atoms, 
        means inserting 0-rows and columns until "size" is reached.
        """
        t0_get_sinemat = time.time()
        if nuclcharges=="":
            nuclcharges = self.nuclear_charges
        zi = zj = nuclcharges
        repulsion = np.dot(zi[:,None],zj[None,:])
        B = np.transpose(self.cell)

        # difference vectors in tensor 3D-tensor-form
        pos_tensor = np.broadcast_to(self.positions, 
            (self.positions.shape[0], self.positions.shape[0], 3))
        diff_tensor = pos_tensor - np.transpose(pos_tensor, (1, 0, 2))
        # np.dot takes sum product over the last axis of a and the second-to-last of b
        arg_to_sin = np.pi * np.dot(diff_tensor, np.linalg.inv(B))
        # calculate phi
        phi = np.linalg.norm(np.dot(np.sin(arg_to_sin), B) ** 2, axis = 2)
        # calculate periodic repulsion between i and j, ignoring diagonal elements
        np.seterr(divide="ignore")
        sinemat = np.multiply(repulsion, 1 / phi)
        
        # set diagonal elements
        di = np.diag_indices_from(sinemat)
        sinemat[di] = 0.5 * zi ** exponent # parameter
        zeros = np.zeros((size,size))
        zeros[:sinemat.shape[0], :sinemat.shape[1]] = sinemat
        sinemat = zeros # invisible atoms have been added
        t1_get_sinemat = time.time()
        dt_get_sinemat = t1_get_sinemat - t0_get_sinemat
        #print("Time needed to get sine matrix:", dt_get_sinemat)
        return sinemat

    def get_codmat(self, dmat="", nuclcharges="", magic_number = 42, size = 100): 
        """Building up on the distance matrix and the charges, the 
        unsymmetric charge over distance matrix is returned 
        Cij = Zi / magic_number      | i = j
            = Zi/[(Ri-Rj) / magic_number] | i > j
            = Zj/[(Ri-Rj) / magic_number]  | i < j
        The coulomb matrix is broadened by invisible atoms, means inserting 
        0-rows and columns until "size" is reached.
        """
        t0_get_codmat = time.time()

        self.calc_dmat()
        if str(dmat)=="":
            dmat = self.dmat
        if nuclcharges=="":
            nuclcharges = self.nuclear_charges

        zi = zj = nuclcharges
        print("nuclcharges: ", zi, zi.shape[0])

        chargemat = np.triu(np.broadcast_to(zi, (zi.shape[0], 
            zi.shape[0]))) + np.tril(np.broadcast_to(zj, (zj.shape[0], zj.shape[0])))
        print(chargemat)

        np.seterr(divide="ignore")
        codmat = np.multiply(chargemat, 1 / dmat)
        
        di = np.diag_indices_from(codmat)
        codmat[di] = zi
        zeros = np.zeros((size,size))
        zeros[:codmat.shape[0], :codmat.shape[1]] = codmat
        codmat = zeros / magic_number # invisible atoms have been added
        t1_get_codmat = time.time()
        dt_get_codmat = t1_get_codmat - t0_get_codmat
        #print("Time needed to get charge over distance matrix:")
        #print(dt_get_cmat)

        return codmat 

    def get_eigvals_cmat(self, cmat="", size = 100):
        """Takes an arbitrary matrix and
        returns its eigenvalues in a vector of descending values
        """
        if str(cmat)=="":
            if self.cmat=="":
                self.get_cmat(dmat="", nuclcharges="", exponent=2.4, size = size)
            cmat = self.cmat

        eigvalues = np.linalg.eigvals(cmat)
        indexlist = np.argsort(eigvalues)
        indexlist = indexlist[::-1] # order highest to lowest
        sorted_eigvalues = eigvalues[indexlist]

        return sorted_eigvalues

    def sort_cmat(self, cmat="",):
        """It takes a coulomb matrix (by default the atoms coulomb matrix) and returns
        the sorted coulomb matrix (high to low) together with a list of indices how to sort
        and the cvector of row norms (cvector)
        """
        if str(cmat)=="":
            cmat = self.cmat
        cvector = np.linalg.norm(cmat, axis=1) # vector of row norms
        indexlist = np.argsort(cvector)
        indexlist = indexlist[::-1] # order highest to lowest
        sorted_cmat = cmat[indexlist][:,indexlist]

        return sorted_cmat, indexlist, cvector

    def permute_cmat(self, cmat, indexlist):
        """Permutes a given coulomb matrix by a given list of indices. Returns the permuted
        coulomb matrix.
        """
        permuted_cmat = cmat[indexlist][:,indexlist]
        return permuted_cmat

    def noise_cvector(self, cvector="", sigma=1000):
        """Takes a cvector and a standard deviation scalar (see Montavon 2012)
        and returns a cvector with added noise together with 
        a list of indices how to sort. """
        noise_cvector = np.random.normal(cvector, sigma)
  
        indexlist = np.argsort(noise_cvector)
        indexlist = indexlist[::-1] # order highest to lowest

        return noise_cvector, indexlist

    def par_noise_cvector(self, cvector="", sigma=77, nrep = 5):
        """As in noise_cvector, just creating parallel noise vectors and indexlists in an
        additional axis.
        CURRENTLY NOT IN USE, BUT WORKING
        """
        par_cvector = np.dstack([cvector]*nrep)
        par_cvector = par_cvector[0] #np.tile(cvector, (nrep, 1)) # copy cvector, add axis
        print(par_cvector)
        noise_cvector = np.random.normal(par_cvector, sigma)
        print("Parallel noise cvector")
        print(noise_cvector, noise_cvector.shape) 

        indexlist = np.argsort(noise_cvector, axis=-1)
        print("indexlist")
        print(indexlist)
        indexlist = indexlist[::-1] # order highest to lowest
        print("flipped indexlist")
        print(indexlist)
        return noise_cvector, indexlist

    def par_permute_cmat(self, cmat, indexlist, nrep = 5):
        """CURRENTLY NOT WORKING
        """
        par_cmat = np.dstack([cmat]*nrep) #np.tile(cmat, (nrep, 1)) # copy cvector, add axis
        print("shape of parallel cmat:", par_cmat.shape)
        permuted_cmat = par_cmat[:,indexlist]
        print("shape of permuted cmats")
        print(permuted_cmat.shape)
        permuted_cmat = permuted_cmat[indexlist[:],:,indexlist[:,:]]
        print(indexlist[:],indexlist[:,:])
        print("shape of permuted cmats")
        print(permuted_cmat.shape)
        #print(permuted_cmat)
        #print(permuted_cmat)
        return permuted_cmat


    def gen_permute_cmat(self, cmat="", sigma=77, nrep=10):
        """Takes coulomb matrix = dmat (by default the atoms coulomb matrix), 
        a standard deviation = sigma and
        number of representations to generate = nrep .
        Returns nrep amount of randomly sorted coulomb matrices.
        sigma = 77 as in
        Hansen, K.; Biegler, F.; Ramakrishnan, R.; Pronobis, W.; von Lilienfeld, O. A.; Müller, K.-R.; Tkatchenko, A. 
        J. Phys. Chem. Lett. 2015, 6 (12), 2326–2331.
        """
        if str(cmat)=="":
            cmat = self.cmat
        sorted_cmat, indexlist, cvector = self.sort_cmat(cmat,)
        rand_sort_cmat_lst = []
        t0_gen_permute_cmat = time.time()
        for i in range(nrep):
            noise_cvector, indexlist = self.noise_cvector(cvector, sigma=sigma)
            permuted_cmat = self.permute_cmat(cmat, indexlist)
            #print("permuted_cmat", i + 1)
            #print(permuted_cmat)
            rand_sort_cmat_lst.append(permuted_cmat)
        rand_sort_cmat_lst = np.array(rand_sort_cmat_lst)
        t1_gen_permute_cmat = time.time()
        dt_gen_permute_cmat = t1_gen_permute_cmat - t0_gen_permute_cmat
        #print("Time needed to get nrep random permutations of coulomb matrix:")
        #print(dt_gen_permute_cmat)
        return rand_sort_cmat_lst

    ####################################


##########################################

class Atom_descriptors(Atoms_descriptors):
    """class of an atom object which additionally stores other information. Parent is always an atoms object.
    """
    def __init__(self, parent, name, idnumber, position, atomtype, nuclcharge):
        self.parent = parent
        self.name = self.parent.name
        self.idnumber = idnumber
        self.position = position
        self.atomtype = atomtype
        self.nuclear_charge = nuclcharge
        self.cutoff = self.parent.cutoff
        self.totalrdf = ""
        self.rdf_atom_type_lst = []
        self.rdf_type_lst = []
        self.intrdf_type_lst = []
        return None

    def get_rdf(self, smearing=False):
        """Returns the total rdf (not volume-scaled) for a single atom.
        """
        self.parent.calc_dmat()
        drow = self.parent.dmat[self.idnumber, :][np.nonzero(self.parent.dmat[self.idnumber])]
        npairs = len(drow)
        if smearing:
            rdf = self.do_convolution(drow, self.parent.var, self.parent.accuracy, self.parent.maximum)
            rdf = rdf / npairs
            self.totalrdf = rdf
        else:
            hist = np.histogram(drow, int(self.parent.maximum / self.parent.accuracy), range=(0, self.parent.maximum))
            self.totalrdf = hist[0]  # only take the y values
        return self.totalrdf


    def get_typerdf(self, smearing=False):
        """Returns the type rdf (not volume-scaled) for a single atom.
        """
        self.parent.calc_dmat()
        drow = self.dmat[self.idnumber, :][np.nonzero(self.dmat[self.idnumber])]
        for atomtype in set(self.atomtypes):
            print(atomtype)
            x_list = self.atomtypes == atomtype
            subdrow = drow[x_list, :]
            npairs = len(subdrow)
            if smearing:
                rdf = self.do_convolution(subdrow, self.parent.var, self.parent.accuracy, self.parent.maximum)
                rdf = rdf / npairs
                self.rdf_type_lst.append(rdf)
            else:
                hist = np.histogram(subdrow, int(self.maximum / self.accuracy), range=(0, self.maximum))
                self.rdf_type_lst.append(hist[1])  # only take the y values
            self.rdf_atom_type_lst.append(atomtype)
        return self.rdf_type_lst, self.rdf_atom_type_lst

    def write_rdf(self, function, typerdf=False, appendix = ""):
        """Takes an existing total or type rdf and writes a 1D-array together with variables information into a file. Returns None.
        """
        x = np.arange(0, self.parent.maximum, step = self.parent.accuracy)
        if typerdf:
            rdf, atypes = function
            for (single_rdf, atomtype) in zip(rdf, atypes):
                np.savetxt(self.name + appendix + "_atom" + str(self.atomtype) + "_" + str(atomtype) + ".dat", single_rdf, header = "Maximum = " + str(self.parent.maximum) + " Accuracy = " + str(self.parent.accuracy) + " Variance = " + str(self.parent.var) + " atom type: " + str(atomtype))
                plt.plot(x[:len(single_rdf)], single_rdf, "-")
                plt.savefig(self.name  + appendix + "_atom" + str(i) + ".png")
                plt.close()
        else:
            rdf = function
            filename = self.name + appendix + "_atom" + str(self.idnumber)
            np.savetxt(filename + ".dat", rdf, header = "total rdf " + "Maximum = " + str(self.parent.maximum) + " Accuracy = " + str(self.parent.accuracy) + " Variance = " + str(self.parent.var))
            plt.plot(x[:len(rdf)], rdf, "-")
            plt.savefig(filename + ".png")
            plt.close()
        return None

    def get_intrdf(self, typerdf=False):
        """Returns the integral of the given radial distribution function.
        Can be either discrete or gaussian smeared.
        """
        if typerdf==False:
            rdf = self.totalrdf
        else:
            rdf = self.rdf_type_lst
        if list(rdf)==[] or str(rdf)=="":
            return "Calculate rdf first"
        if typerdf:
            for single_rdf in rdf[0]:
                x = np.arange(0, self.parent.maximum, step = self.parent.accuracy)
                x = x[:len(single_rdf)]
                intrdf = scipy.integrate.cumtrapz(single_rdf, x, axis=-1, initial=0)
                self.intrdf_type_lst.append(intrdf)
            return self.intrdf_type_lst, self.rdf_atom_type_lst
        else:
            x = np.arange(0, self.parent.maximum, step = self.parent.accuracy)
            x = x[:len(rdf)]
            self.intrdf = scipy.integrate.cumtrapz(rdf, x, axis=-1, initial=0)
            return self.intrdf

    def fit2_ax2(self, function):
        """Takes a 1D function which is fitted to a * x ** 2.
        Returns a. """
        x = np.arange(0, self.parent.maximum, step = self.parent.accuracy)
        x = x[:len(function)]
        y = function
        self.sqfit = np.polynomial.polynomial.polyfit(x, y, deg=[2], rcond=None, full=False, w=None)
        print(self.sqfit)
        return self.sqfit

    def get_local_cmat(self, cut_dist="", size=""):
        """Building up on the distance matrix and the charges, the coulomb matrix is
        returned. Takes only neighbours within a certain distance into account. Fills up the local coulomb matrix with
        invisible atoms up to a certain size"""
        if str(cut_dist)=="":
            cut_dist = self.cutoff
        dmat = self.parent.dmat
        cmat = self.parent.cmat
        local_pos = self.position.reshape(1, self.position.shape[0])
        all_pos = self.parent.positions
        distances = cdist(local_pos,all_pos)

        loc_nn = np.nonzero(distances <= cut_dist)[1] # indices of nearest neighbours and itself

        self.local_cmat = cmat[loc_nn, : ][: , loc_nn]

        if size:
            zeros = np.zeros((size,size))
            zeros[:self.local_cmat.shape[0], :self.local_cmat.shape[1]] = self.local_cmat
            self.local_cmat = zeros # invisible atoms have been added     
        #print(self.local_cmat)
        return self.local_cmat


##########################################

class Descriptor_pair:
    """
    """
    def __init__(self, object1, object2):
        return


##########################################


