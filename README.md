
# deepregression

[![R build status](https://github.com/davidruegamer/deepregression/workflows/R-CMD-check/badge.svg)](https://github.com/davidruegamer/deepregression/actions)
[![Codecov test coverage](https://codecov.io/gh/davidruegamer/deepregression/branch/master/graph/badge.svg)](https://codecov.io/gh/davidruegamer/deepregression?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

# Installation

To install the package, use the following command:
``` r
devtools::install_github("davidruegamer/deepregression")
```
Note that the installation requires additional packages (see below) and their installation is currently forced by `deepregression`.

# Requirements

The requirements are given in the `DESCRIPTION`. If you load the package manually using `devtools::load_all`, make sure the following packages are availabe:

  - Matrix
  - dplyr
  - keras
  - mgcv
  - reticulate
  - tensorflow
  - tfprobability

If you set up a Python environment for the first time, install `reticulate` and run the `check_and_install` function from the `deepregression` package. This tries to install miniconda, TF 2.1, TFP 0.9 and keras 2.4.3, which seems to be the most reliable setup for `deepregression` at the moment.

# How to cite this?

Until published, please cite the following preprint:

    @article{rugamer2020unifying,
      title={A Unifying Network Architecture for Semi-Structured Deep Distributional Learning},
      author={R{\"u}gamer, David and Kolb, Chris and Klein, Nadja},
      journal={arXiv preprint arXiv:2002.05777},
      year={2020}
    }

# How to use this?

See [the tutorial](vignettes/tutorial.md) for a detailed introduction.

# Python version

A Python version of the package is available [here](https://github.com/davidruegamer/PySDDR). 

# Related literature

The following works are based on the ideas implemented in this package:

* [https://arxiv.org/abs/2002.05777](Original Semi-Structured Deep Distributional Regression Proposal)
* [https://arxiv.org/abs/2010.06889](Neural Mixture Distributional Regression)
* [https://arxiv.org/abs/2010.07860](Deep Conditional Transformation Models)
* [https://arxiv.org/abs/2101.00661](Combining Graph Neural Networks and Spatio-temporal Disease Models to Predict COVID-19 Cases in Germany)
