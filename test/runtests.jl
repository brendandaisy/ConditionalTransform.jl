using Test
using Revise
using Distributions
using SampleTransform

X = Normal(0, 1)
Y = Normal(0, 1)
u = 3
f_inv(x, s) = s - x
d_f_inv(x, s) = 1

Xcond, Ycond = sample_cond_f([X, Y], u, f_inv, d_f_inv; pivot=2, nsamples=20_000)
Xcond2 = rand((u + Normal(0, sqrt(2))) / 2, 20_000)
@test abs(mean(Xcond) - mean(Xcond2)) <= 0.01
@test abs(mean(Xcond.^2) - 2.75) <= 0.01
