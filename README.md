

<picture align="center">
  <img alt="DataShifts Logo" src="logo/datashifts.svg" width="900">
</picture>



--------------------------------------------------------------------------------

# <span style="font-size:94%;">DataShifts — A Toolkit for Quantifying Distribution Shifts</span>

[![PyPI version](https://img.shields.io/pypi/v/datashifts?color=blue)](https://pypi.org/project/datashifts/) [![PyPI downloads](https://pepy.tech/badge/datashifts?color=green)](https://pypi.org/project/datashifts/) [![License](https://img.shields.io/pypi/l/datashifts.svg)](https://github.com/DataShifts/datashifts/blob/main/LICENSE)

DataShifts is a Python package that makes it simple to **measure and analyze the distribution shifts from labeled samples**. It can be used with tensor computation frameworks such as [PyTorch](https://github.com/pytorch/pytorch), [NumPy](https://github.com/numpy/numpy) and [KeOps](https://github.com/getkeops/keops). It is designed for data science practitioners who need a principled way to answer questions such as:

* *How far has my production data shifted from the training set?*

* *How do the model’s representations shift in a new domain, and are they robust to distribution shifts?*

* *Are the distribution shifts mainly in the inputs (covariate shift) or in the labels (concept shift)?*

* *How do these distribution shifts affect model performance?*

In analysis, distribution shift is often decomposed into **covariate shift ($X$ shift)** and **concept shift ($Y|X$ shift)**. The general theory below shows that the error bound **scales linearly with** these two shifts. With a single call, **DataShifts** estimates these two shifts from labeled samples, providing a rigorous and general tool for quantifying and analyzing distribution shift.

---

## Core Theory — General Learning Bound under Distribution Shifts

Let the covariate and label spaces be metric spaces $(\mathcal{X} ,\rho _{\mathcal{X}}),(\mathcal{Y} ,\rho _{\mathcal{Y}})$, and $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$ are two joint distributions of covariates and labels on $\mathcal{X}\times\mathcal{Y}$. If the hypothesis $h:\mathcal{X} \rightarrow \mathcal{Y}'$ is $L _h$-Lipschitz continuous, loss $\ell :\mathcal{Y} \times \mathcal{Y} '\rightarrow \mathbb{R}$ is separately $(L _{\ell},L _{\ell}')$-Lipschitz continuous, then:

$$
\LARGE
\epsilon _B(h)\le \epsilon _A(h)+L _hL _{\ell}'S _{Cov}+L _{\ell}S _{Cpt}^{\gamma ^*}
$$

where $\epsilon _A(h), \epsilon _B(h)$ are the errors of hypothesis $h$ under the distributions $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$, respectively. $S _{Cov}, S _{Cpt}^{\gamma ^*}$ are **covariate shift** (= $X$ shift, distribution shift of covariates) and **concept shift** (= $Y|X$ shift, distribution shift of labels conditioned on covariates) between $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$. Both shifts are defined in closed form via **entropic optimal transport**.

This elegant theory shows how distribution shifts affect the error, and has the following advantages:

* **General**: Because the theory assumes no particular loss or space, it applies broadly to losses and tasks—including regression, classification, and multi-label problems, as long as the covariate and label space of the problem can define metrics. Moreover, depending on whether the covariate space is the raw feature space or the model’s representation space, the theory can measure shifts in either the original data or the learned representations.

* **Estimable**: Both covariate shift $S _{Cov}$ and concept shift $S _{Cpt}^{\gamma ^*}$ in the theory can be rigorously estimated from finite samples drawn from the two distributions—**which is the core capability of this package**.

For further theoretical details, please see our [original paper](https://arxiv.org/abs/2506.12829).

---
## Installation

Just use the following command to install DataShifts package:
```shell
pip install datashifts
```


---

## Quick Example

```python
import torch
from datashifts import DataShifts

# Generate data from two different distributions (take labels originating from pure noise as an example)
N=10000       #Number of samples
x_dim=200     #Feature dimensions
y_dim=10      #Label dimensions
x_shift=10.0  #True covariate shift
device="cuda" #Device

random_directions=torch.randn(1, x_dim, device=device)
x_shift_vector=random_directions/((random_directions**2).sum()**(1/2))*x_shift
# First distribution
x1 = torch.randn(N, x_dim, device=device)
y1= torch.rand(N, y_dim, device=device)
# Second distribution
x2 = torch.randn(N, x_dim, device=device)+x_shift_vector
y2= torch.rand(N, y_dim, device=device)

# Using DataShifts to quantify covariate and concept shifts
covariate_shift, concept_shift=DataShifts(x1, x2, y1, y2)
print("Covariate shift: ", covariate_shift)
print("Concept shift: ",   concept_shift  )
```

Typical output

```
The sample size of (x1,y1,w1) is larger than parameter 'N_max'=5000, sampling strategy is used.
The sample size of (x2,y2,w2) is larger than parameter 'N_max'=5000, sampling strategy is used.
Covariate shift: tensor(9.9608, device='cuda:0')
Concept shift:   tensor(1.2627, device='cuda:0')
```

---

## `datashifts.DataShifts` —  <span style="font-size:90%;">Measure Covariate  &  Concept Shift between Distributions from Samples</span>

`datashifts.DataShifts` is the core method of the DataShifts package, which estimates covariate shift and concept shift from finite labeled samples `(x1,y1), (x2,y2)` drawn from two distributions, with automatic sub‑sampling for scalability and GPU acceleration.

```python
covariate_shift, concept_shift = DataShifts(
            x1, x2, y1, y2,                    # required
            weights1=None, weights2=None,      # optional importance weights
            eps=0.01,                          # entropic regularisation
            N_max=5000,                        # max points kept per distribution
            device=None,                       # "cpu", "cuda" or None (auto)
            seed=None,                         # random seed for reproducibility
            verbose=True                       # print progress messages
)
```

*Note (temporary): For now, Euclidean distance is the only built-in metric. Custom metrics are planned.*

### Parameters

| name                   | type                                  | default | description                                                  |
| ---------------------- | ------------------------------------- | ------- | ------------------------------------------------------------ |
| `x1`, `x2`             | `torch.Tensor` **or** `numpy.ndarray` | —       | Covariates of the samples drawn from two distributions.<br>Shapes accepted:`(Batch_size, Num_samples, Dim_x)` or `(Num_samples, Dim_x)` |
| `y1`, `y2`             | `torch.Tensor` **or** `numpy.ndarray` | —       | Corresponding labels.<br>Shapes accepted:`(Batch_size, Num_samples, Dim_y)` or `(Num_samples, Dim_y)`. Must match `x*` in `Batch_size` and `Num_samples` dimensions. |
| `weights1`, `weights2` | `torch.Tensor` **or** `numpy.ndarray` | `None`  | Sample weights.<br/>Shapes accepted:`(Batch_size, Num_samples)` or `(Num_samples)`. Must match `x*` in `Batch_size` and `Num_samples` dimensions. |
| `eps`                  | `float`                               | `0.01`  | Entropic regularisation for optimal transport. Smaller => more precise but slower. |
| `N_max`                | `int`                                 | `5000`  | Upper bound on samples per distribution kept for optimal transport. If `N>N_max`, the function resamples without replacement to speed up the solution (weighted if `weights*` provided). Larger => more precise but slower. |
| `device`               | `str`                                 | `None`  | Running device.<br/>`"cpu"`, `"cuda"`/`"gpu"`, or `None`(= automatically use GPU if available). |
| `seed`                 | `int`                                 | `None`  | Random seed for shuffling and sampling.                      |
| `verbose`              | `bool`                                | `True`  | Whether to print progress messages (sampling or automatic device choice). |


### Returns

```python
covariate_shift : torch.Tensor
concept_shift   : torch.Tensor
```

Returned objects are PyTorch tensors placed on the chosen `device`.

---
## Licensing, Citation, Academic Use

This package is released under the [MIT License](https://en.wikipedia.org/wiki/MIT_License). See the [LICENSE](https://github.com/DataShifts/datashifts/blob/main/LICENSE) file for full details.

If you use this package in a research paper, **please cite** our [original paper](https://arxiv.org/abs/2506.12829):

```latex
@article{chen2025general,
  title={General and Estimable Learning Bound Unifying Covariate and Concept Shifts},
  author={Chen, Hongbo and Xia, Li Charlie},
  journal={arXiv preprint arXiv:2506.12829},
  year={2025}
}
```

---

> **Contributions & issues** welcome at https://github.com/DataShifts/datashifts/issues
