Mixtures of von Mises-Fisher Distributions (movMF)
===========

movMF is a distribution model for D-dimensional spherical data. movMF is analogous to Gaussian distribution but in spherical space. This code is based on the article [Clustering on the Unit Hypersphere using
von Mises-Fisher Distributions](http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf) of A. Banerjee.



Install
-------------

Get and Compile (you need boost):

```
git clone https://github.com/mrouvier/movMF
cd movMF
make
```

Generate randomly data:

```
python data/generate_data.py > data/vec
```

or

```
make generate_date
```


Program usage
-------------

To train a statistical mixture model :

```
./bin/movmf_train --nb_mixture 128 --nb_iteration_em 10  --train data/vec --save mixture.txt
```

where *nb_mixture* is the number of mixture, *nb_iteration_em* is the number of iteration of Exepctation Maximization (algorithm used to train the mixture model), *train* is the training file and *save* is a file, the program save the model to this file.


To test the mixture model :


```
./bin/movmf_test data/vec mixture.txt
```

Given a mixture model (mixture.txt) and a testing file (data/vec), the program caculate the log likelihood for each vector.

