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


Program usage
-------------

```
./bin/movmf data/vec
```

