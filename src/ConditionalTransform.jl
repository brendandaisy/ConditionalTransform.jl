module ConditionalTransform

#TODO: how about...PDFTransform? ProbabilityTransform?

using Distributions

export ARSampler, accept_reject, sample_trunc_f, sample_cond_f

struct ARSampler
    target::Function
    proposal::Distribution
end

ARSampler(f, p...) = ARSampler(f, product_distribution(p...))

# target(s, ar::ARSampler) = ar.target(s)

function accept_reject(ar::ARSampler; acc_rate=100, nsamples=100, silent=false)
    ret = []
    attempts = 0
    for _ in 1:nsamples
        s = rand(ar.proposal)
        u = rand()
        attempts += 1
        while u >= ar.target(s) / (acc_rate*pdf(ar.proposal, s))
            s = rand(ar.proposal)
            u = rand()
            attempts += 1
        end
        push!(ret, s)
    end
    if !silent
        @info "Acceptance rate was $(nsamples/attempts)"
    end
    return [getindex.(ret, i) for i=1:length(ar.proposal)]
end

function sample_trunc_f(dists::AbstractVector, cond::Function; arkwargs...)
    dprod = product_distribution(dists...)
    target_pdf = x -> cond(x...) ? 0 : pdf(dprod, x)
    ar = ARSampler(target_pdf, dprod)
    accept_reject(ar; arkwargs...)
end

sample_trunc_f(dists::NamedTuple, cond::Function; arkwargs...) = sample_trunc_f(vcat(dists...), cond; arkwargs...)

"""
Assumes the order of parameters in `dists`, `f_inv`, and `d_f_inv` are the same
"""
function sample_cond_f(dists::AbstractVector, u, f_inv, d_f_inv; pivot=1, cond=nothing, arkwargs...)
    dpivot = dists[pivot]
    deleteat!(dists, pivot)

    target_pdf = x -> begin
        cond !== nothing && cond(x..., u) && return 0
        J = abs(d_f_inv(x..., u))
        prod(pdf.(dists, x)) * J * pdf(dpivot, f_inv(x..., u))
    end

    ar = ARSampler(target_pdf, dists...)
    samps = accept_reject(ar; arkwargs...)
    pivot_samps = f_inv.(samps..., u)
    return insert!(samps, pivot, pivot_samps)
end

function sample_cond_f(dists::NamedTuple, u, f_inv, d_f_inv; pivot=1, kwargs...)
    if pivot isa Symbol
        pivot_idx = findall(x->x == pivot, keys(dists))[1]
    end
    sample_cond_f(vcat(dists...), u, f_inv, d_f_inv; pivot=pivot_idx, kwargs...)
end

end # module
