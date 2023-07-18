/*Copyright 2019 DScrie developers

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
#include "soap.h"
#include "soapGeneral.h"
#include "soapGTO.h"
#include "geometry.h"

using namespace std;

SOAPGTO::SOAPGTO(
    double r_cut,
    int n_max,
    int l_max,
    double eta,
    py::dict weighting,
    bool crossover,
    string average,
    double cutoff_padding,
    py::array_t<double> alphas,
    py::array_t<double> betas,
    py::array_t<int> species,
    bool periodic
)
    : Descriptor(periodic, average, r_cut+cutoff_padding)
    , r_cut(r_cut)
    , n_max(n_max)
    , l_max(l_max)
    , eta(eta)
    , weighting(weighting)
    , crossover(crossover)
    , cutoff_padding(cutoff_padding)
    , alphas(alphas)
    , betas(betas)
    , species(species)
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
    // Empty mock arrays since we are not calculating the derivatives
    py::array_t<double> xd({1, 1, 1, 1, 1});
    py::array_t<double> yd({1, 1, 1, 1, 1});
    py::array_t<double> zd({1, 1, 1, 1, 1});
    py::array_t<double> derivatives({1, 1, 1, 1});
    py::array_t<int> indices({1});
    py::array_t<int> center_indices({1});

    soapGTO(
        derivatives,
        out,
        xd,
        yd,
        zd,
        positions,
        centers,
        center_indices,
        this->alphas,
        this->betas,
        atomic_numbers,
        this->species,
        this->r_cut,
        this->cutoff_padding,
        this->n_max,
        this->l_max,
        this->eta,
        this->weighting,
        this->crossover,
        this->average,
        indices,
        false,
        true,
        false,
        cell_list
    );
}

int SOAPGTO::get_number_of_features() const
{
    int n_species = this->species.shape(0);
    return this->crossover
        ? (n_species*this->n_max)*(n_species*this->n_max+1)/2*(this->l_max+1) 
        : n_species*(this->l_max+1)*((this->n_max+1)*this->n_max)/2;
}

void SOAPGTO::derivatives_analytical(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<double> xd,
    py::array_t<double> yd,
    py::array_t<double> zd,
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<double> centers,
    py::array_t<int> center_indices,
    py::array_t<int> indices,
    const bool attach,
    const bool return_descriptor
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

    // Calculate neighbours with a cell list
    CellList cell_list(positions, this->cutoff);

    soapGTO(
        derivatives,
        descriptor,
        xd,
        yd,
        zd,
        positions,
        centers,
        center_indices,
        this->alphas,
        this->betas,
        atomic_numbers,
        this->species,
        this->r_cut,
        this->cutoff_padding,
        this->n_max,
        this->l_max,
        this->eta,
        this->weighting,
        this->crossover,
        this->average,
        indices,
        attach,
        return_descriptor,
        true,
        cell_list
    );
}

SOAPPolynomial::SOAPPolynomial(
    double r_cut,
    int n_max,
    int l_max,
    double eta,
    py::dict weighting,
    bool crossover,
    string average,
    double cutoff_padding,
    py::array_t<double> rx,
    py::array_t<double> gss,
    py::array_t<int> species,
    bool periodic
)
    : Descriptor(periodic, average, r_cut+cutoff_padding)
    , r_cut(r_cut)
    , n_max(n_max)
    , l_max(l_max)
    , eta(eta)
    , weighting(weighting)
    , crossover(crossover)
    , cutoff_padding(cutoff_padding)
    , rx(rx)
    , gss(gss)
    , species(species)
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
        this->r_cut,
        this->cutoff_padding,
        this->n_max,
        this->l_max,
        this->eta,
        this->weighting,
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
        ? (n_species*this->n_max)*(n_species*this->n_max+1)/2*(this->l_max+1) 
        : n_species*(this->l_max+1)*((this->n_max+1)*this->n_max)/2;
}
