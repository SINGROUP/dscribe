Unsupervised Learning: Clustering
=================================

In this tutorial we take a look at how the descriptors can be used to perform a
common unsupervised learning task called *clustering*. In clustering we are
using an unlabeled dataset of input values -- in this case the feature vectors
that DScribe outputs -- to train a model that organizes these inputs into
meaningful groups/clusters.

Setup
-----
We will try to find structurally similar locations on top of an copper
FCC(111)-surface. To do this, we will first calculate a set of SOAP vectors on
top of the surface. To simplify things, we will only consider a single plane 1
Å above the topmost atoms. This set of feature vectors will be our dataset.

.. figure:: /_static/img/fcc111.png
   :alt: FCC(111) surface
   :align: center
   :width: 50%

   The used copper FCC(111) surface as viewed from above.

This dataset will be used as input for a clustering model. We will use one of
the most common and simplest models: k-means clustering. The goal is to use
this model to categorize all of the sampled sites into a fixed subset of
clusters. We will fix the number of clusters to ten, but this could be changed
or even determined dynamically if we used some other clustering algorithm.

As with all forms of unsupervised learning, we do not have the "correct"
answers that we could optimize our model againsts. There are certain `ways to
measure the clustering model performance
<https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment>`_
even without correctly labeled data, but in this simple example we will simply
use a setup that provides a reasonable result in our opinion: this is
essentially biasing our model.

Dataset generation
------------------
The following script generates our training dataset:

.. literalinclude:: ../../../../examples/clustering/dataset.py
    :language: python

Training
--------
Let's load the dataset and fit our model:

.. literalinclude:: ../../../../examples/clustering/training.py
    :language: python
    :lines: 1-20

Analysis
--------
When the training is done (takes few seconds), we can visually examine the
clustering. Here we simply plot the sampled points and colour them based on the
cluster that was assigned by our model.

.. literalinclude:: ../../../../examples/clustering/training.py
   :start-at: # Visualize clusters in a plot
   :language: python
   :lines: 1-

The resulting clustering looks like this:

.. figure:: /_static/img/clustering.png
   :alt: Lennard-Jones energies
   :align: center
   :width: 90%

   The k-means clustering result.

We can see that our simple clustering setup is able to determine similar
regions in our sampling plane. Effectively we have reduced the plane into ten
different regions, from which we could select e.g. one representative point per
region for further sampling. This provides a powerful tool for pre-selecting
informative samples containing chemically and structurally dinstinct sites for
e.g. supervised training.
