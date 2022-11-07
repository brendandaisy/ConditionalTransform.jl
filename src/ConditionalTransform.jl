module ConditionalTransform

#TODO: how about...PDFTransform? ProbabilityTransform?

using Distributions

export ARSampler, accept_reject, sample_cond_f

struct ARSampler
    target::Function
    proposal::Distribution
end

ARSampler(f, p...) = ARSampler(f, product_distribution(p...))

# target(s, ar::ARSampler) = ar.target(s)

function accept_reject(ar::ARSampler; acc_rate=100, nsamples=100)
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
    @info "Acceptance rate was $(nsamples/attempts)"
    return [getindex.(ret, i) for i=1:length(ar.proposal)]
end

function sample_cond_f(dists::AbstractVector, u, f_inv, d_f_inv; pivot=1, acc_rate=100, nsamples=100)
    dpivot = dists[pivot]
    deleteat!(dists, pivot)
    target = x -> prod(pdf.(dists, x)) * abs(d_f_inv(x..., u)) * pdf(dpivot, f_inv(x..., u))
    ar = ARSampler(target, dists...)
    samps = accept_reject(ar; acc_rate, nsamples)
    pivot_samps = f_inv.(samps..., u)
    return push!(samps, pivot_samps)
end

function sample_cond_f(dists::NamedTuple, u, f_inv, d_f_inv; pivot=1, kwargs...)
    if pivot isa Symbol
        pivot = getindex(dists, pivot)
    end
    sample_cond_f(values(dists), u, f_inv, d_f_inv; pivot, kwargs...)
end

end # module
