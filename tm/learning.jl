
function q_z(log_theta, log_phi, word_mask)
    K, V = size(log_phi)
    T = size(log_theta, 1)
    z = Array(SparseMatrixCSC, T)
    for t = 1:T
        N = size(word_mask[t], 1)
        I = repmat([1:K], N)
        J = vcat({ones(K) * v for v in word_mask[t]})
        data = zeros(N * K)
        for v in 1:N
            w = word_mask[v]
            z_t = exp(log_theta[t, :] + log_phi[:, w])
            z[v*K:(v+1)*K] = z_t ./ z_t.sum()
        end
        z[t] = sparse(J, I, data, V, K)
    end
    return z
end

function q_theta(n, alpha, p0, L, resample=true, smooth=true) 
    T, V = size(n)
    K = size(alpha, 1)
    
    theta = zeros(T, L, K)
    log_theta = zeros(T, L, K)
    pt = zeros(T, L)
    w = zeros(T, L)

    let dir = Dirichlet(alpha + n[1, :]') 
        theta[1, :] = rand(dir, L)
        pt[1, :] = 0 
        w[1, :] = vcat([logpdf(dir, theta[1, j]) for j=1:L]...)
    end

    for i=2:T
        let dir_t = Dirichlet(alpha + n[t, :]')
    end
end

function M_step(n, beta)
    K, V = size(n)
    phi = n + beta - 1

    return bsxfun(./, phi, sum(phi, 2))
end

function VEM(w, alpha, beta)

end
