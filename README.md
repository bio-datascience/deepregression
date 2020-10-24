
# deepregression

[![R build status](https://github.com/davidruegamer/deepregression/workflows/R-CMD-check/badge.svg)](https://github.com/davidruegamer/deepregression/actions)
[![Codecov test coverage](https://codecov.io/gh/davidruegamer/deepregression/branch/master/graph/badge.svg)](https://codecov.io/gh/davidruegamer/deepregression?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

# Installation

Since the repository is still private, clone the repository to your
local machine and either install the package locally or use
``` r
devtools::load_all("davidruegamer/deepregression")
```
which is often more reliable due to some open issues.

# Requirements

The requirements are given in the `DESCRIPTION`. If you load the package manually using `devtools`, make sure the following packages are availabe:

  - Matrix
  - dplyr
  - keras
  - mgcv
  - reticulate
  - tensorflow
  - tfprobability

If you setup a Python environment for the first time, install `reticulate` and run the `check_and_install` function from the `deepregression` package. This tries to install miniconda, TF 2.0, TFP 0.8 and keras 2.3, which seems to be the most reliable setup for `deepregression` at the moment.

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
