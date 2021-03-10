/*Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <iostream>
#include "soap.h"
#include "soapGTO.h"
#include "soapGeneral.h"
#include "soapGTODevX.h"
#include "geometry.h"

using namespace std;

SOAPGTO::SOAPGTO(
    double rcut,
    int nmax,
    int lmax,
    double eta,
    py::array_t<int> species,
    bool periodic,
    bool crossover,
    string average,
    double cutoff_padding,
    py::array_t<double> alphas,
    py::array_t<double> betas
)
    : Descriptor(periodic, average, rcut+cutoff_padding)
    , rcut(rcut)
    , nmax(nmax)
    , lmax(lmax)
    , eta(eta)
    , species(species)
    , crossover(crossover)
    , cutoff_padding(cutoff_padding)
    , alphas(alphas)
    , betas(betas)
{
}

void SOAPGTO::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<double> centers
) const
{
    // Extend system if periodicity is requested.
    auto pbc_u = pbc.unchecked<1>();
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    if (is_periodic) {
        ExtendedSystem system_extended = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
        positions = system_extended.positions;
        atomic_numbers = system_extended.atomic_numbers;
    }
    this->create(out, positions, atomic_numbers, centers);
}

void SOAPGTO::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers
) const
{
    // Calculate neighbours with a cell list
    CellList cell_list(positions, this->cutoff);
    this->create(out, positions, atomic_numbers, centers, cell_list);
}

void SOAPGTO::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers,
    CellList cell_list
) const
{
    soapGTO(
        out,
        positions,
        centers,
        this->alphas,
        this->betas,
        atomic_numbers,
        this->species,
        this->rcut,
        this->cutoff_padding,
        this->nmax,
        this->lmax,
        this->eta,
        this->crossover,
        this->average,
        cell_list
    );
}

void SOAPGTO::create_cartesian(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<double> xd,
    py::array_t<double> yd,
    py::array_t<double> zd,
    py::array_t<double> cd,
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<double> centers,
    py::array_t<int> center_indices,
    py::array_t<int> indices,
    const bool return_descriptor
) const
{
    int n_atoms = atomic_numbers.shape(0); // Should be saved before extending the system
    int n_centers = centers.shape(0);

    // Extend system if periodicity is requested.
    auto pbc_u = pbc.unchecked<1>();
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    if (is_periodic) {
        ExtendedSystem system_extended = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
        positions = system_extended.positions;
        atomic_numbers = system_extended.atomic_numbers;
    }

    soapGTODevX(
        derivatives,
        descriptor,
        xd,
        yd,
        zd,
        cd,
        positions,
        centers,
        center_indices,
        this->alphas,
        this->betas,
        atomic_numbers,
        this->species,
        this->rcut,
        this->cutoff_padding,
        n_atoms,
        this->nmax,
        this->lmax,
        n_centers,
        this->eta,
        this->crossover,
        indices,
        return_descriptor,
        false
    );
}

int SOAPGTO::get_number_of_features() const
{
    int n_species = this->species.shape(0);
    return this->crossover
        ? (n_species*this->nmax)*(n_species*this->nmax+1)/2*(this->lmax+1) 
        : n_species*(this->lmax+1)*((this->nmax+1)*this->nmax)/2;
}

void SOAPGTO::derivatives_analytical(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<double> xd,
    py::array_t<double> yd,
    py::array_t<double> zd,
    py::array_t<double> cd,
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<double> centers,
    py::array_t<int> center_indices,
    py::array_t<int> indices,
    const bool return_descriptor
) const
{
    int n_atoms = atomic_numbers.shape(0); // Should be saved before extending the system
    int n_centers = centers.shape(0);

    // Extend system if periodicity is requested.
    auto pbc_u = pbc.unchecked<1>();
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    if (is_periodic) {
        ExtendedSystem system_extended = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
        positions = system_extended.positions;
        atomic_numbers = system_extended.atomic_numbers;
    }

    soapGTODevX(
        derivatives,
        descriptor,
        xd,
        yd,
        zd,
        cd,
        positions,
        centers,
        center_indices,
        this->alphas,
        this->betas,
        atomic_numbers,
        this->species,
        this->rcut,
        this->cutoff_padding,
        n_atoms,
        this->nmax,
        this->lmax,
        n_centers,
        this->eta,
        this->crossover,
        indices,
        return_descriptor,
        true
    );
}

SOAPPolynomial::SOAPPolynomial(
    double rcut,
    int nmax,
    int lmax,
    double eta,
    py::array_t<int> species,
    bool periodic,
    bool crossover,
    string average,
    double cutoff_padding,
    py::array_t<double> rx,
    py::array_t<double> gss
)
    : Descriptor(periodic, average, rcut+cutoff_padding)
    , rcut(rcut)
    , nmax(nmax)
    , lmax(lmax)
    , eta(eta)
    , species(species)
    , crossover(crossover)
    , cutoff_padding(cutoff_padding)
    , rx(rx)
    , gss(gss)
{
}

void SOAPPolynomial::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<double> centers
) const
{
    // Extend system if periodicity is requested.
    auto pbc_u = pbc.unchecked<1>();
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    if (is_periodic) {
        ExtendedSystem system_extended = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
        positions = system_extended.positions;
        atomic_numbers = system_extended.atomic_numbers;
    }
    this->create(out, positions, atomic_numbers, centers);
}

void SOAPPolynomial::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers
) const
{
    // Calculate neighbours with a cell list
    CellList cell_list(positions, this->cutoff);
    this->create(out, positions, atomic_numbers, centers, cell_list);
}

void SOAPPolynomial::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers,
    CellList cell_list
) const
{
    soapGeneral(
        out,
        positions,
        centers,
        atomic_numbers,
        this->species,
        this->rcut,
        this->cutoff_padding,
        this->nmax,
        this->lmax,
        this->eta,
        this->rx,
        this->gss,
        this->crossover,
        this->average,
        cell_list
    );
}

int SOAPPolynomial::get_number_of_features() const
{
    int n_species = this->species.shape(0);
    return this->crossover
        ? (n_species*this->nmax)*(n_species*this->nmax+1)/2*(this->lmax+1) 
        : n_species*(this->lmax+1)*((this->nmax+1)*this->nmax)/2;
}
