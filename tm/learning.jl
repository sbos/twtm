require("Options")
using OptionsMod

function q_z(log_theta, log_phi, docptr, wordval)
    K, V = size(log_phi)
    T = size(log_theta, 1)
    z = Array(SparseMatrixCSC, T)
    for t = 1:T
        doc = wordval[docptr[t] : docptr[t+1] - 1]
        N = length(doc)
        I = squeeze(repmat([1:K], N, 1))
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
    @defaults opts filtering=true smoothing=false
    T, V = size(n)
    K = size(alpha, 1)
    
    theta = zeros(T, L, K)
    log_theta = zeros(T, L, K)
    pt = zeros(T, L)
    w = None
    if filtering == true
        w = zeros(T, L)
    end
    
    let q = Dirichlet(alpha + vec(n[1, :])),
        pr = Dirichlet(alpha),
        n_t = vec(n[1, :])

        pt[1, :] = 0
        for j = 1:L
            th = rand(q)
            if filtering == true
                w[1, j] = logpdf(pr, th) + dot(log(th), n_t) - logpdf(q, th)
            end
            theta[1, j, :] = th
        end

        if filtering == true
            w[1, :] = adjust(w[1, :])
        end
    end

    pdir = Dirichlet(alpha)

    function f(currth, prevth)
        if sum((currth - prevth) .^ 2) < 1e-10
            return log(p0)
        end
        return log(1-p0) + logpdf(pdir, currth)
    end

    for t=2:T
        if filtering == true
            idx              = resample(exp(vec(w[t-1, :])))
            assert(length(idx) == L)
            theta[t-1, :, :] = theta[t-1, idx, :]
            w[t-1, :]        = -log(L)
            pt[t-1, :]       = pt[t-1, idx]
        end

        let n_t = vec(n[t, :])
            function pt_t(th)
                a = p0 * prod(th .^ n_t)
                b = (1 - p0) * exp(log_dirmult(alpha) - log_dirmult(alpha + n_t))
                return a / (a + b)
            end

            dir = Dirichlet(alpha + n_t)
            function q(th)
                pt_j = pt_t(th)
                if rand(Bernoulli(pt_j)) == 1
                    return th, log(pt_j), pt_j
                end
                sample = rand(dir)
                return sample, log(1-pt_j) + logpdf(dir, sample), pt_j
            end

            function g(th)
                return dot(log(th), n_t)
            end
            
            for j=1:L
                th_prev = vec(theta[t-1, j, :])
                theta[t, j, :], q_t_j, pt[t, j] = q(th_prev)
                if filtering == true
                    th_curr = vec(theta[t, j, :])
                    w[t, j] = w[t-1, j] + f(th_curr, th_prev) + g(th_curr) - q_t_j
                end
            end

            if filtering == true
                w[t, :] = adjust(w[t, :])
            end
        end
    end

    if filtering == true && smoothing == true
        idx = resample(exp(vec(w[T, :])))
        theta[T, :, :] = theta[T, idx, :]
        pt[T, :] = pt[T, idx]

        for t=T-1:-1:1
            for j=1:L
                w[t, j] = w[t, j] + f(vec(theta[t+1, j, :]), vec(theta[t, j, :]))
            end

            w[t, :] = adjust(w[t, :])
            idx = resample(exp(vec(w[t, :])))
            assert(length(idx) == L)
            theta[t, :, :] = theta[t, idx, :]
            pt[t, :] = pt[t, idx]
        end
    end

    log_theta    = zeros(T, K)
    sample_theta = zeros(T, K)
    sample_pt    = zeros(T)
    for t=1:T
        th = squeeze(theta[t, :, :])
        if filtering == true
            w_t = vec(exp(w[t, :]))
            assert(abs(sum(w_t) - 1) < 1e-5)
            log_theta[t, :]    = sum(bsxfun(.*, log(th), w_t), 1)
            sample_theta[t, :] = sum(bsxfun(.*,      th, w_t), 1)
            sample_pt[t]       = sum(bsxfun(.*, vec(pt[t, :]), w_t))    
        else
            log_theta[t, :]    = sum(log(th), 1) / L
            sample_theta[t, :] = sum(th, 1) / L
            sample_pt[t]       = mean(pt[t, :])
        end
        assert(abs(sum(sample_theta[t, :]) - 1) < 1e-5)
        assert(sample_pt[t] >= 0.0 && sample_pt[t] - 1 < 1e-5)
    end

    @check_used opts

    return log_theta, sample_theta, sample_pt
end

function E_step(docs, alpha, phi, p0::Float64, L::Int, maxiter::Int, opts::Options)
    @defaults opts filtering=true smoothing=false
    V, T = size(docs)
    K = length(alpha)
    n = rand(Dirichlet(alpha), T)
    log_theta, theta, pt = q_theta(n, alpha, p0, L, opts)

    log_phi = log(phi)
    z = null
    for iter=1:maxiter
        z = q_z(log_theta, log_phi, docs.colptr, docs.rowval)
        n = count_z(docs, z)
        log_theta, theta, pt = q_theta(n, alpha, p0, L, opts)
    end

    @check_used opts

    return theta, z, pt
end

function M_step(docs, z, beta::FloatingPoint)
    T = length(z)
    V, K = size(z[1])
    phi = ones(K, V) * beta - 1

    for t=1:T
        idx = docs.colptr[t]:docs.colptr[t+1]-1
        wc = docs.nzval[idx]
        w = docs.rowval[idx] 
        for k=1:K
            z_k = z[t].nzval[z[t].colptr[k]:z[t].colptr[k+1]-1]
            phi[k, w] += dot(wc, z_k)
        end
    end

    for k=1:K
        for v=1:V
            if phi[k, v] < 0.0
                phi[k, v] = 0.0
            end
        end
    end

    phi = bsxfun(./, phi, sum(phi, 2))
    return phi
end

function VEM(docs, alpha, beta, p0, L, eiter, totaliter, opts::Options)
    @defaults opts filtering=true smoothing=false

    K = length(alpha)
    V, T = size(docs)
    phi = gen_phi(V, K, 1.0)
    theta, z, pt = None, None, None
    for iter=1:totaliter
        theta, z, pt = E_step(docs, alpha, phi, p0, L, eiter, opts)
        phi = M_step(docs, z, beta)
        println("VEM iter=", iter, "/", totaliter)
    end

    @check_used opts

    return z, theta, phi, pt
end

export q_z, count_z, q_theta, E_step, M_step, VEM
