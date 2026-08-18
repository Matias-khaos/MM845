# Tutorial 3 — Variational Problems, Spectra, and Linear Models on Geometric Data

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 3: Linear Models, Optimisation & Regularisation**

---

## The idea

Lecture 3 is the one lecture where everything can be proved: least squares is an
orthogonal projection, the loss is convex, the minimiser is unique and closed-form,
and the convergence rate of gradient descent is a ratio of eigenvalues. This
tutorial spends that certainty on geometry.

It covers two syllabus items, and they turn out to be the same idea twice.

**Part I — *Variational Problems and Function Approximation on Manifolds*.**
Approximating a function on $S^2$ *is* orthogonal projection in $L^2(S^2)$. Choose
the eigenbasis of the Laplace–Beltrami operator and the projection becomes diagonal,
the condition number drops to $1$, and Tikhonov regularisation becomes literally a
penalty on the Dirichlet energy $\int_M|\nabla f|^2$. We also recover that
eigenbasis from nothing but a point cloud, using a graph Laplacian.

**Part II — *Linear Models for Geometric Classification and Regression*.** We build
a family of random plane curves, label them with quantities computable in closed
form (area, length, convexity), and predict those labels by linear and logistic
regression. The theme is Lecture 3's design principle: the feature map is worth more
than the model.

## Files

| File | What it is |
|---|---|
| [`variational_and_linear_models.ipynb`](variational_and_linear_models.ipynb) | The tutorial. 14 sections, 11 figures, 8 exercises. |
| `README.md` | This file. |

## Running it

You need the `aigeo` environment from [Tutorial 1](../tutorial_01/README.md).
NumPy and SciPy throughout; **PyTorch is used in §13 only**.

```bash
$ conda activate aigeo
$ jupyter lab variational_and_linear_models.ipynb
```

The whole notebook runs top to bottom in about a minute, most of it in §3 (a dense
eigendecomposition) and §13 (two PyTorch fits). **Kernel → Restart Kernel and Run
All Cells** should complete without error before you start editing.

## Contents

| § | Topic | Lecture 3 connection |
|---|---|---|
| **Part I** | | |
| 1 | Approximation as orthogonal projection; empirical vs exact inner product | slide 3 |
| 2 | The Laplace–Beltrami eigenbasis, built by Gram–Schmidt | slide 9 |
| 3 | Recovering the eigenbasis from a point cloud (graph Laplacian) | — |
| 4 | Spectral decay: how fast can you approximate? | slide 3 |
| 5 | Ridge $=$ Dirichlet energy $=$ a spectral filter | slide 8 |
| 6 | Conditioning: the same subspace in two bases | slides 5–6 |
| **Part II** | | |
| 7 | A dataset of random plane curves, with exact labels | slide 10 |
| 8 | Symmetry: $SO(2)$ on Fourier modes, and invariant features | slide 9 |
| 9 | Regression I — area: the model recovers $\pi$ | slides 3, 11 |
| 10 | Regression II — length: identifiability and $\kappa$ | slides 5–6 |
| 11 | Classification: convex vs non-convex | slide 4 |
| 12 | Ridge and lasso; feature selection as conjecture generation | slide 8 |
| 13 | PyTorch: the canonical training loop | slide 11 |
| 14 | Summary | |

## Results worth watching for

**§2–3 — the spectrum of $S^2$, twice, from two directions.** Running Gram–Schmidt
on monomials in order of degree produces an orthonormal basis whose blocks have
sizes $1, 3, 5, 7, 9, 11, 13$ — that is $\dim\mathcal{H}_\ell = 2\ell+1$, obtained
without ever writing down a spherical harmonic. Then a graph Laplacian built from
*pairwise distances alone* returns eigenvalues $0, 2, 6, 12, 20$ with exactly those
multiplicities, and its eigenspaces agree with the analytic $\mathcal{H}_\ell$ to
within $0.08^\circ$ of principal angle. The multiplicities are the convincing part:
they are integers, so no fitted constant can fake them.

The eigenvalues themselves are biased low, and §3 shows the bias vanishing as
$O(t)$ — halve the kernel bandwidth, halve the error.

**§5 — regularisation is geometry.** In the harmonic basis the Dirichlet energy is
$\sum_k \lambda_k c_k^2$, so weighted ridge with $w_k = \ell(\ell+1)$ *is* a penalty
on $\int|\nabla f|^2$. The fitted coefficients then satisfy
$\hat c_k = \hat c_k^{\mathrm{LS}}/(1+\lambda\lambda_k)$ — verified to $3\times10^{-4}$
— and the measured shrinkage factors fall exactly on that curve. The Dirichlet
penalty beats plain ridge by about 35% in test MSE, because it knows the noise is
spread across the spectrum while the signal is not.

**§6 — the condition number, on a scale.** The monomial and harmonic bases span the
*same* subspace, give the same projection and the same answer. But
$\kappa \approx 1.2\times10^4$ versus $\kappa \approx 1.5$: about 28,000 gradient
steps to gain one digit, versus about 4.

**§9 — a model that recovers $\pi$.** By Parseval the area of a polar curve is
exactly $\pi a_0^2 + \frac{\pi}{2}\sum_k(a_k^2+b_k^2)$, so it is *linear* in the
rotation-invariant power spectrum. Least squares on those features returns
$R^2 = 1.000000000000$, an intercept equal to $\pi$ to ten decimals, and eight
weights all equal to $\pi/2$ to thirteen. On the raw Fourier coefficients — more
features, same data — it returns $R^2 = 0.003$.

**§10 — the weights you may not quote.** Fitting length on the same features
recovers the second-order prediction $\frac{\pi}{2}k^2$ for low $k$ and wanders for
high $k$. That is not error but *unidentifiability*: the feature $p_k$ has standard
deviation falling like $k^{-5}$, so by $k=8$ it varies too little to determine its
own coefficient. Bootstrap error bars make the distinction visible, and it is the
distinction between a result and a number.

Incidentally, the weights $k^2$ are the eigenvalues of $-d^2/d\theta^2$: Part I's
Dirichlet energy reappears, unprompted, as the answer to a regression about arc
length.

**§13 — a small loss is not a right answer.** The canonical PyTorch loop, run on the
area regression for 20000 epochs, reaches a training MSE of $3\times10^{-11}$ and
gets the last three weights badly wrong. Re-run with a flatter spectrum — same
optimiser, same epochs, same target, only the conditioning changed — and it returns
$\pi/2$ in every coordinate to $2\times10^{-16}$. Training longer does not fix the
first case; nothing does, because those directions of parameter space are flat.

## Exercises

Eight, spread through the notebook.

| # | § | Topic |
|---|---|---|
| 1 | 1 | The projection theorem and Pythagoras, numerically |
| 2 | 3 | Breaking the symmetry: degeneracies split on an ellipsoid; Weyl's law |
| 3 | 5 | Other regularisers, coloured noise, and the rate of $\lambda^\star \to 0$ |
| 4 | 6 | Watching the zig-zag: GD rates from $\kappa$ |
| **5** | 9 | **What invariance buys: augmentation vs invariant features** |
| 6 | 11 | Better features for convexity; polynomial feature maps |
| 7 | 12 | Ridge vs lasso paths; instability of selection; elastic net |
| 8 | 13 | The condition number, visible in training curves |

Exercise 5 is the one to do if you do only one — it is the Part II analogue of
Tutorial 2's orbit-splitting trap.

## What to take away

- **Least squares is orthogonal projection**, on a manifold as much as in
  $\mathbb{R}^n$; the design matrix is a discretised Gram matrix, and having only
  samples means having only an approximate inner product.
- **The Laplacian's eigenbasis is the good basis** — it diagonalises the problem,
  makes $\kappa\approx1$, and is recoverable from a bare point cloud.
- **Regularisation is geometry.** Choosing a penalty is choosing which functions you
  consider plausible; on a manifold that choice has a name and a formula.
- **Approximation error is coefficient decay**, hence regularity. No optimiser
  repairs a poor ansatz.
- **The feature map is the model.** Invariant features made the area regression
  exact and its weights readable; raw coordinates, with more parameters, gave
  nothing.
- **Read the error bars, not the point estimates.** Coefficients are trustworthy
  only where their features vary.

## Next

**Lecture 4** replaces the fixed feature map $\phi$ with a *learned* one — neural
networks — giving up convexity, closed forms and guarantees in exchange for
expressive power. Because this tutorial pinned down exactly what those guarantees
were worth, Tutorial 4 can say precisely what the trade bought.

## Further reading

- Atkinson & Han, *Spherical Harmonics and Approximation on the Unit Sphere* (Springer, 2012) — Part I in full rigour.
- Belkin & Niyogi, "Laplacian eigenmaps for dimensionality reduction and data representation", *Neural Computation* **15** (2003) — the construction in §3.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 3 — ridge, lasso, and the ball picture. Free online.
- Trefethen, *Approximation Theory and Approximation Practice* (SIAM, 2013), ch. 7–8 — why smoothness governs decay, in one dimension and beautifully.
- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning* (2021), ch. 3 — invariant features as the principled version of §8.
