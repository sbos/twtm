using Distributions
using Debug

function gen_phi(V::Integer, K::Integer, beta::FloatingPoint)
    dir = Dirichlet(ones(V) * beta)
    return rand(dir, K)'
end

function gen_theta(T::Int64, alpha::Array{Float64, 1}, p0::FloatingPoint)
    K = size(alpha, 1)

    theta = zeros(T, K)
    
    switch = Bernoulli(p0)
    dir = Dirichlet(alpha)
    
    theta[1, :] = rand(dir, 1)
    for t=2:T
        if rand(switch, 1)[1] == 1
            theta[t, :] = theta[t-1, :]
        else
            theta[t, :] = rand(dir, 1)
        end
    end

    return theta
end

function gen_docs(N::Integer, theta::Array{Float64, 2}, phi::Array{Float64, 2})
    T, K = size(theta)
    K, V = size(phi)
    topics = [Categorical(vec(phi[k,:])) for k=1:K]
    v = [1:T]
    I = [v[div(i,N)+1] for i=0:N*length(v)-1]
    J = zeros(Int, N * T)
    w = ones(Int, N * T)
    for t=1:T
        mult = Multinomial(N, vec(theta[t, :]))
        n = rand(mult, 1)
        for k in 1:K
            if n[k] <= 0
                continue
            end
            wk = rand(topics[k], n[k])
            J[(t-1)*N + sum(n[1:k-1]) + 1 : (t-1)*N + sum(n[1:k])] = wk
        end
    end 
    return sparse(J, I, w, V, T)
end

export gen_phi, gen_theta, gen_docs
