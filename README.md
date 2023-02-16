# ConditionalTransform.jl

A helper package written in Julia to sample from various conditional distributions. If $P(X)$ is a known, joint probability distribution function, and $u = f(X)$ is a transformation function, sampling from the following transformations is currently implemented:

* $P(X \mid u)$, i.e. sample $x \sim P(X)$ such that $f(x) = u$. Use function `sample_cond_f`
* Sample $x \sim P(X)$ such that $f(x)$ returns `false`. Use function `sample_trunc_f`
