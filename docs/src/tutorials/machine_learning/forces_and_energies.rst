Predicting forces and energies a.k.a. training a ML force-field
===============================================================

This tutorial covers how descriptors can be effectively used as an input for a
machine learning model that will predict energies and forces. There are several
choices that you have to make in building a full-blown ML force field. For the
sake of simplicity we have decided on the following setup:

    - We will use a dataset of two atoms interacting through a Lennard-Jones
      potential. This pretty much as simple as it gets. Real systems will be
      much more complicated thus requiring a more complicated machine learning
      model and longer training times.
    - We will use the SOAP descriptor calculated directly between the two
      atoms. Once again this is for simplicity and in real systems you would
      have many more centers, possibly on top of each atom.
    - We will use a fully connected neural network to perform the prediction.
      In principle any machine learning method will do, but with neural
      networks it is very convenient to analytically calculate the derivatives
      of the output with respect to the input. This allows us to train an
      energy prediction model from which we will automatically get the forces
      as long as we also know the derivatives of the descriptor with respect to
      the atomic positions. This is exactly what the :code:`derivatives`-function
      provided by DScribe returns. 

Principle
---------
We will use a dataset of feature vectors :math:`\mathbf{D}`, their derivatives
as a Jacobian matrix :math:`\nabla \mathbf{D}` and the associated system energies :math:`E` and
forces :math:`\mathbf{F}` for training. We will use a neural network :math:`f`
to predict the energies: :math:`\hat{E} = f(\mathbf{D})`, while the predicted
forces can be directly computed as the negative gradient with respect to the
atomic positions: :math:`\hat{\mathbf{F}} = -\nabla f(\mathbf{D})`. Here
variables with a "hat" on top indicate predicted quantities to distinguish them
from the real values.

The loss function for the neural network will contain the sum of mean squared
error of both energies and forces. In order to better equalize the contribution
of these two properties in the loss function their values are scaled by their
variance in the training set.

Dataset generation
------------------
The following script generates our training dataset. You can find it in the
examples folder.

.. literalinclude:: ../../../../examples/forces_and_energies/dataset.py
    :language: python

The energies will look like this:

.. image:: /_static/img/lj.png
   :alt: Lennard-Jones energies
   :align: center

Training
--------

Terminology
-----------

Analysis
--------
