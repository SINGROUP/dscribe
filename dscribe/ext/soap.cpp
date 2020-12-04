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
#include "soap.h"
#include "soapGTO.h"
#include "soapGeneral.h"

using namespace std;

SOAPGTO::SOAPGTO(
    double rcut,
    int nmax,
    int lmax,
    double eta,
    py::array_t<int> species,
    bool crossover,
    string average,
    double cutoff_padding,
    py::array_t<double> alphas,
    py::array_t<double> betas
)
    : Descriptor(average)
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
    py::array_t<double> centers
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
        this->average
    );
}

int SOAPGTO::get_number_of_features() const
{
    int n_species = this->species.shape(0);
    return this->crossover
        ? (n_species*this->nmax)*(n_species*this->nmax+1)/2*(this->lmax+1) 
        : n_species*(this->lmax+1)*((this->nmax+1)*this->nmax)/2;
}

SOAPPolynomial::SOAPPolynomial(
    double rcut,
    int nmax,
    int lmax,
    double eta,
    py::array_t<int> species,
    bool crossover,
    string average,
    double cutoff_padding,
    py::array_t<double> rx,
    py::array_t<double> gss
)
    : Descriptor(average)
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
    py::array_t<double> centers
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
        this->average
    );
}

int SOAPPolynomial::get_number_of_features() const
{
    int n_species = this->species.shape(0);
    return this->crossover
        ? (n_species*this->nmax)*(n_species*this->nmax+1)/2*(this->lmax+1) 
        : n_species*(this->lmax+1)*((this->nmax+1)*this->nmax)/2;
}
