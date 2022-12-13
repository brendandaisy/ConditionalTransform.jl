module ConditionalTransform

using Distributions

export AcceptReject, accept_reject, sample_trunc_f, sample_cond_f

# struct ARSampler
#     target::Function
#     proposal::Distribution
# end

struct AcceptReject
    acc_rate::Float64
end

# ARSampler(f, p...) = ARSampler(f, product_distribution(p...))

# target(s, ar::ARSampler) = ar.target(s)

function accept_reject(target, proposal, nsamples, sampler; silent=false)
    ret = []
    attempts = 0    
    for _ in 1:nsamples
        s = rand(proposal)
        u = rand()
        attempts += 1
        while u >= target(s) / (sampler.acc_rate*pdf(proposal, s))
            s = rand(proposal)
            u = rand()
            attempts += 1
        end
        push!(ret, s)
    end
    if !silent
        @info "Acceptance rate was $(nsamples/attempts)"
    end
    return [getindex.(ret, i) for i=1:length(proposal)]
end

"""
Sample from P(θ) truncated in the region where cond(θ) is true, i.e. P(θ∣cond(θ) == false)
"""
function sample_trunc_f(dists::AbstractVector, cond::Function; nsamples=100, sampler=AcceptReject(100))
    dprod = product_distribution(dists...)
    target_pdf = x -> cond(x...) ? pdf(dprod, x) : 0
    accept_reject(target_pdf, dprod, nsamples, sampler)
end

sample_trunc_f(dists::NamedTuple, cond; kwargs...) = sample_trunc_f(vcat(dists...), cond; kwargs...)

"""
Assumes the order of parameters in `dists`, `f_inv`, and `d_f_inv` are the same
"""
function sample_cond_f(
    dists::AbstractVector, u, f_inv, d_f_inv; 
    pivot=1, cond=nothing, nsamples=100, sampler=AcceptReject(100)
)
    dpivot = dists[pivot]
    deleteat!(dists, pivot)

    target_pdf = x -> begin
        cond !== nothing && cond(x..., u) && return 0
        J = abs(d_f_inv(x..., u))
        prod(pdf.(dists, x)) * J * pdf(dpivot, f_inv(x..., u))
    end

    samps = accept_reject(target_pdf, product_distribution(dists), nsamples, sampler)
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
