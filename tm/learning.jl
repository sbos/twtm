require("Options")
using OptionsMod

function q_z(log_theta, log_phi, docptr, wordval)
    K, V = size(log_phi)
    T = size(log_theta, 1)
    z = Array(SparseMatrixCSC, T)
    for t = 1:T
        doc = wordval[docptr[t] : docptr[t+1] - 1]
        N = length(doc)
        I = vec(repmat([1:K], N, 1))
        J = vcat([ones(Int, K) * w for w in doc]...)
        data = zeros(N * K)
        for v in 1:N
            w = doc[v]
            z_t = exp(log_theta[t, :]' + log_phi[:, w])
            data[(v-1)*K + 1 : v*K] = z_t ./ sum(z_t)
        end
        z[t] = sparse(J, I, data, V, K)
    end
    return z
end

function count_z(docs, z)
    V, T = size(docs)
    V, K = size(z[1])
    n = zeros(T, K)

    for t=1:T
        w = docs.nzval[docs.colptr[t]:docs.colptr[t+1]-1]
        for k=1:K
            z_k = z[t].nzval[z[t].colptr[k]:z[t].colptr[k+1]-1]
            n[t, k] = dot(w, z_k)
        end
    end
    return n
end

function q_theta(n, alpha, p0, L, opts::Options)
    @defaults opts smoothing=true
    T, V = size(n)
    K = size(alpha, 1)
    
    theta = zeros(T, L, K)
    log_theta = zeros(T, L, K)
    pt = zeros(T, L)
    w = zeros(T, L)

    function resample(w)
        dist = Categorical(w)
        return rand(dist, length(w))
    end

    let q = Dirichlet(alpha + n[1, :]'),
        pr = Dirichlet(alpha),
        N = sum(n[1, :])
        theta[1, :] = rand(q, L)
        pt[1, :] = 0 
        w[1, :] = logpdf(pr, theta[1, :, :]) + vcat([logpdf(Multinomial(N, theta[1, j, :]'), n[1, :]') - logpdf(q, theta[1, j, :]) for j=1:L]...)
    end

    for t=2:T
        idx = resample(exp(w[t-1, :]))
        theta[t-1, :, :] = theta[t-1, idx, :]
        w[t-1, :]        = w[t-1, idx]
        pt[t-1, :]       = pt[t-1, idx]

        let n = n[t, :]'
            function pt(th)
                a = p0 * prod(th .^ n)
                b = (1-p0) * exp(log_dirmult(alpha) - log_dirmult(alpha + n))
                return a / (a + b)
            end

            dir = Dirichlet(alpha + n)
            function q(th)
                pt = pt(th)
                if rand(Binomial(1, pt), 1)[1] == 1
                    return th, log(pt), pt
                end
                sample = vec(rand(dir, 1))
                return sample, log(1-pt) + logpdf(dir, sample), pt
            end

            N = sum(n)
            pdir = Dirichlet(alpha)
            function g(th)
                return sum(log(theta) .* n)
            end

            function f(currth, prevth)
                if sum(abs(currth - prevth)) < 1e-10
                    return log(p0)
                end
                return log(1-p0) + logpdf(pdir, currth)
            end

            for j=1:L
                theta[t, j, :], w[t, j], pt[t, j] = propose(theta[t-1, j, :])
                w[t, j] = w[t-1, j] + f(theta[t, j, :], theta[t-1, j, :]) + g(theta[t, j, :]) - w[t, j]
            end
        end
    end

    if smoothing == true
        for t=N:-1:1
            
        end
    end

    log_theta    = zeros(T, K)
    sample_theta = zeros(T, K)
    sample_pt    = zeros(T)
    for t=1:T
        w = exp(w[t, :])
        log_theta[t, :]    = sum(log(theta[t, :, :]) .* w)
        sample_theta[t, :] = sum(    theta[t, :, :]  .* w)
        sample_pt[t]       = sum(       pt[t, :]     .* w)
    end

    return log_theta, sample_theta, sample_pt
end

function M_step(n, beta)
    K, V = size(n)
    phi = n + beta - 1

    return bsxfun(./, phi, sum(phi, 2))
end

export q_z, count_z, q_theta, M_step
