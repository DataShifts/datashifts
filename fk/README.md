---

## Core Theory — General Learning Bound under Distribution Shifts

Let the covariate and label spaces be metric spaces $(\mathcal{X} ,\rho _{\mathcal{X}}),(\mathcal{Y} ,\rho _{\mathcal{Y}})$, and $\mathcal{D} _{XY}^{A}, \mathcal{D} _{XY}^{B}$ are two joint distributions of covariates and labels on $\mathcal{X}\times\mathcal{Y}$. If the hypothesis $h:\mathcal{X} \rightarrow \mathcal{Y}'$ is $L_h$-Lipschitz continuous, loss $\ell :\mathcal{Y} \times \mathcal{Y} '\rightarrow \mathbb{R}$ is separately $(L _\ell, L _\ell')$ -Lipschitz continuous, then:
$$
\epsilon _B(h)\le \epsilon _A(h)+L _hL _{\ell}^{'}\,S _{Cov}+L _{\ell}\,S _{Cpt}^{\gamma ^*}
$$


