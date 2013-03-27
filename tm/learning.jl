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

    function adjust(w)
        w -= max(w)
        w -= log(sum(exp(w)))
        return w
    end

    let q = Dirichlet(alpha + squeeze(n[1, :])),
        pr = Dirichlet(alpha),
        n_t = squeeze(n[1, :])

        pt[1, :]    = 0
        for j = 1:L
            th = rand(q)
            w[1, j] = logpdf(pr, th) + dot(log(th), n_t) - logpdf(q, th)
            theta[1, j, :] = th
        end

        w[1, :] = adjust(w[1, :])
    end

    pdir = Dirichlet(alpha)

    function f(currth, prevth)
        if sum(abs(currth - prevth)) < 1e-10
            return log(p0)
        end
        return log(1-p0) + logpdf(pdir, currth)
    end

    for t=2:T
        idx = resample(exp(vec(w[t-1, :])))
        theta[t-1, :, :] = theta[t-1, idx, :]
        w[t-1, :]        = -log(L)
        pt[t-1, :]       = pt[t-1, idx]

        let n_t = squeeze(n[t, :])
            function pt_t(th)
                a = p0 * prod(th .^ n_t)
                b = (1-p0) * exp(log_dirmult(alpha) - log_dirmult(alpha + n_t))
                return a / (a + b)
            end

            dir = Dirichlet(alpha + n_t)
            function q(th)
                pt_j = pt_t(th)
                if rand(Binomial(1, pt_j), 1)[1] == 1
                    return th, log(pt_j), pt_j
                end
                sample = squeeze(rand(dir, 1))
                return sample, log(1-pt_j) + logpdf(dir, sample), pt_j
            end

            function g(th)
                return dot(log(th), n_t)
            end
            
            for j=1:L
                th_prev = vec(theta[t-1, j, :])
                theta[t, j, :], w[t, j], pt[t, j] = q(th_prev)
                th_curr = vec(theta[t, j, :])
                w[t, j] = w[t-1, j] + f(th_curr, th_prev) + g(th_curr) - w[t, j]
            end

            w[t, :] = adjust(w[t, :])
            w_t = vec(exp(w[t, :]))
        end
    end

    if smoothing == true
        idx = resample(exp(vec(w[T, :])))
        theta[T, :, :] = theta[T, idx, :]
        pt[T, :] = pt[T, idx]

        for t=T-1:-1:1
            for j=1:L
                w[t, j] = w[t, j] + f(vec(theta[t+1, j, :]), vec(theta[t, j, :]))
            end

            w[t, :] = adjust(w[t, :])
            idx = resample(exp(vec(w[t, :])))
            theta[t, :, :] = theta[t, idx, :]
            pt[t, :] = pt[t, idx]
        end
    end

    log_theta    = zeros(T, K)
    sample_theta = zeros(T, K)
    sample_pt    = zeros(T)
    for t=1:T
        w_t = vec(exp(w[t, :]))
        th = squeeze(theta[t, :, :])
        log_theta[t, :]    = sum(bsxfun(.*, log(th), w_t), 1)
        sample_theta[t, :] = sum(bsxfun(.*,      th, w_t), 1)
        sample_pt[t, :]    = sum(bsxfun(.*, squeeze(pt[t, :]), w_t))
    end

    @check_used opts

    return log_theta, sample_theta, sample_pt
end

function M_step(n, beta)
    K, V = size(n)
    phi = n + beta - 1

    return bsxfun(./, phi, sum(phi, 2))
end

export q_z, count_z, q_theta, M_step
