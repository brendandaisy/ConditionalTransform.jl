# ConditionalTransform.jl

A helper package written in Julia to sample from various conditional distributions. If $P(X)$ is a known, joint probability density function, and $u = f(X)$ is a transformation function, sampling from $P(X \mid u)$ for two types of transformations are currently implemented:

* If $u$ is a scalar, sample $x \sim P(X)$ such that $f(x) = u$. Use function `sample_cond_f`
* If $u$ is a boolean, sample $x \sim P(X)$ such that $f(x)$ returns `false`. Use function `sample_trunc_f`
