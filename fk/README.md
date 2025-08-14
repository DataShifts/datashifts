---

## Core Theory — General Learning Bound under Distribution Shifts

Let the covariate and label spaces be metric spaces  $(\mathcal {X} ,\rho _{\mathcal {X}}), (\mathcal {Y} ,\rho _{\mathcal {Y}})$ , and  distribution shift of labels conditioned on covariates) between $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$. are two joint distributions of covariates and labels on  $\mathcal{X}\times\mathcal{Y}$ . If the hypothesis  $h:\mathcal{X} \rightarrow \mathcal{Y} ^{'}$  is $L_h$ -Lipschitz continuous, loss  $\ell :\mathcal {Y} \times \mathcal {Y} ^{'} \rightarrow \mathbb {R} $  is separately  $(L_{\ell},L_{\ell}^{'})$ -Lipschitz continuous, then:


$$
\epsilon _B(h)\le \epsilon _A(h)+L_hL_{\ell}^{'}\,S_{Cov}+L_{\ell}\,S_{Cpt}^{\gamma ^*}
$$

where  $\epsilon _A(h), \epsilon _B(h)$  are the errors of hypothesis  $h$  under the distributions  $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$ , respectively.  $S_{Cov}, S_{Cpt}^{\gamma ^*}$  are **covariate shift** (= $X$ shift, distribution shift of covariates) and **concept shift** (= $Y|X$ shift, distribution shift of labels conditioned on covariates) between $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$. Both shifts are defined in closed form via **entropic optimal transport**.

This elegant theory shows how distribution shifts affect the error, and has the following advantages:

* **General**: Because the theory assumes no particular loss or space, it applies broadly to losses and tasks—including regression, classification, and multi-label problems, as long as the covariate and label space of the problem can define metrics. Moreover, depending on whether the covariate space is the raw feature space or the model’s representation space, the theory can measure shifts in either the original data or the learned representations.
* **Estimable**: Both covariate shift $S_{Cov}$ and concept shift $S_{Cpt}^{\gamma ^*}$ in the theory can be rigorously estimated from finite samples drawn from the two distributions—**which is the core capability of this package**.

For further theoretical details, please see our [original paper](https://arxiv.org/abs/2506.12829).
