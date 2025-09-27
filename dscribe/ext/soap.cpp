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
    string average,
    double cutoff_padding,
    py::array_t<int> species,
    py::array_t<double> species_weights,
    bool periodic,
    string compression,
    py::array_t<double> alphas,
    py::array_t<double> betas
)
    : DescriptorLocal(periodic, average, r_cut+cutoff_padding)
    , r_cut(r_cut)
    , n_max(n_max)
    , l_max(l_max)
    , eta(eta)
    , weighting(weighting)
    , cutoff_padding(cutoff_padding)
    , species(species)
    , species_weights(species_weights)
    , compression(compression)
    , alphas(alphas)
    , betas(betas)
{
}

void SOAPGTO::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<int> centers,
    CellList cell_list
)
{
}

void SOAPGTO::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers,
    CellList cell_list
)
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
        this->species_weights,
        this->r_cut,
        this->cutoff_padding,
        this->n_max,
        this->l_max,
        this->eta,
        this->weighting,
        this->average,
        this->compression,
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
    if ( this->compression == "mu1nu1" ){
        return (n_species*this->n_max*this->n_max) * (this->l_max+1);
    } else if ( this->compression == "mu2" ){
        return (this->n_max * (this->n_max+1) * (this->l_max+1) / 2);
    } else if ( this->compression == "crossover" ){
        return n_species*(this->l_max+1)*((this->n_max+1)*this->n_max)/2;
    } else{
        return (n_species*this->n_max)*(n_species*this->n_max+1)*(this->l_max+1)/2;
    }
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
        System system_extended = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
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
        this->species_weights,
        this->r_cut,
        this->cutoff_padding,
        this->n_max,
        this->l_max,
        this->eta,
        this->weighting,
        this->average,
        this->compression,
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
    string average,
    double cutoff_padding,
    py::array_t<int> species,
    py::array_t<double> species_weights,
    bool periodic,
    string compression,
    py::array_t<double> rx,
    py::array_t<double> gss
)
    : DescriptorLocal(periodic, average, r_cut+cutoff_padding)
    , r_cut(r_cut)
    , n_max(n_max)
    , l_max(l_max)
    , eta(eta)
    , weighting(weighting)
    , cutoff_padding(cutoff_padding)
    , species(species)
    , species_weights(species_weights)
    , compression(compression)
    , rx(rx)
    , gss(gss)
{
}

void SOAPPolynomial::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<int> centers,
    CellList cell_list
)
{
}
void SOAPPolynomial::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers,
    CellList cell_list
)
{
    soapGeneral(
        out,
        positions,
        centers,
        atomic_numbers,
        this->species,
        this->species_weights,
        this->r_cut,
        this->cutoff_padding,
        this->n_max,
        this->l_max,
        this->eta,
        this->weighting,
        this->rx,
        this->gss,
        this->average,
        this->compression,
        cell_list
    );
}

int SOAPPolynomial::get_number_of_features() const
{
    int n_species = this->species.shape(0);
    if ( this->compression == "mu1nu1" ){
        return (n_species*this->n_max*this->n_max) * (this->l_max+1);
    } else if ( this->compression == "mu2" ){
        return (this->n_max * (this->n_max+1) * (this->l_max+1) / 2);
    } else if ( this->compression == "crossover" ){
        return n_species*(this->l_max+1)*((this->n_max+1)*this->n_max)/2;
    } else{
        return (n_species*this->n_max)*(n_species*this->n_max+1)/2*(this->l_max+1);
    }
}
