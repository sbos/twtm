function q_z(log_theta, log_phi, docs)
    K, V = size(log_phi)
    T = size(log_theta, 1)
    z = Array(SparseMatrixCSC{Float64, Int}, T)
    for t = 1:T
        doc = spcolidx(docs, t)
        N = length(doc)
        I = vec(repmat([1:K], N, 1))
        J = vcat([ones(Int, K) * w for w in doc]...)
        data = zeros(N * K)
        for v in 1:N
            w = doc[v]
            z_t = exp(log_theta[t, :]' + log_phi[:, w])

            assert(sum(isnan(z_t)) == 0, "oops, there is NaN somewhere")

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
        w = spcolval(docs, t)
        for k=1:K
            z_k = spcolval(z[t], k)
#println(z_k)
            n[t, k] = dot(w, z_k)
        end
    end
    
    return n
end

function q_theta(n, alpha, p0, L, opts::Options)
    @defaults opts filtering=true smoothing=false

    assert(0 <= p0 <= 1.0, "p0 must be probability")

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
                assert(0.0 <= pt_j <= 1.0, "something is wrong with p_t: $pt_j")
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
        th = squeeze(theta[t, :, :], [1])
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
    n = bsxfun(.*, rand(Dirichlet(alpha), T), float(vec(sum(docs, 1))))
    log_theta, theta, pt = q_theta(n, alpha, p0, L, opts)

    log_phi = log(phi)
    z = null
    for iter=1:maxiter
        z = q_z(log_theta, log_phi, docs)
        n = count_z(docs, z)
        log_theta, theta, pt = q_theta(n, alpha, p0, L, opts)
    end

    @check_used opts

    return theta, z, pt
end

function estimate_phi(feeds::Vector{Feed}, zs::Vector{WordAssignment}, beta::Float64)
    F = length(feeds)
    V, K = size(zs[1][1])

    assert(F == length(zs))

    phi = ones(K, V) * beta - 1

    function pjob(f)
        local_phi = zeros(K, V)
        docs = feeds[f]
        z = zs[f]
        V, T = size(docs)

        for t=1:T
            word_idx = spcolidx(docs, t)
            word_count = spcolval(docs, t)

            for k=1:K
                z_t_k = spcolval(z[t], k) #vec(dense(z[t][:, k]))
                assert(length(z_t_k) == length(word_count))
                local_phi[k, word_idx] += (word_count .* z_t_k)' #(word_count .* z_t_k[word_idx])'
            end
        end

        println("estimating phi, f=$f/$F")
        return local_phi
    end

    phi += @parallel (+) for f=1:F
        pjob(f)
    end

    phi[phi .< 0.0] = realmin(Float64)

    phi = bsxfun(./, phi, sum(phi, 2))

    assert(sum(isnan(phi)) == 0, "Some parts of phi is NaN")

    return phi
end

function estimate_phi(feed::Feed, z::WordAssignment, beta::Float64)
    feeds = Array(Feed, 0)
    push!(feeds, feed)
    
    zs = Array(WordAssignment, 0)
    push!(zs, z)

    return estimate_phi(feeds, zs, beta)
end
      
function VEM(feeds::Vector{Feed}, alpha::Vector{Float64}, beta::FloatingPoint, p0::FloatingPoint, L::Int, eiter::Int, totaliter::Int, opts::Options)
    @defaults opts filtering=true smoothing=false

    K = length(alpha)
    F = length(feeds)
    V, T = size(feeds[1])
    phi = gen_phi(V, K, 1.0)

    thetas = Array(Array{Float64, 2}, F)
    zs = Array(WordAssignment, F)
    pts = Array(Array{Float64, 1}, F)

    for iter=1:totaliter
        for f=1:F
            thetas[f], zs[f], pts[f] = E_step(feeds[f], alpha, phi, p0, L, eiter, opts)
            println("E-step feed=", f, "/", F)
        end
        phi = estimate_phi(feeds, zs, beta)
        println("VEM iter=", iter, "/", totaliter)
    end

    @check_used opts

    return zs, thetas, phi, pts
end

function PVEM(feeds::Vector{Feed}, alpha::Vector{Float64}, beta::FloatingPoint, p0::FloatingPoint, L::Int, eiter::Int, totaliter::Int, opts::Options)
    @defaults opts filtering=true smoothing=false

    K = length(alpha)
    F = length(feeds)
    V, T = size(feeds[1])
    phi = gen_phi(V, K, 1.0)

    thetas = Array(Array{Float64, 2}, F)
    zs = Array(WordAssignment, F)
    pts = Array(Array{Float64, 1}, F)

    for iter=1:totaliter
        function apply_e_step(f)
            expectations = E_step(feeds[f], alpha, phi, p0, L, eiter, opts)
            println("E-step feed=$f/$F")
            return expectations
        end

        result = pmap(apply_e_step, [1:F])

        for f=1:F
            thetas[f], zs[f], pts[f] = result[f]
        end

        phi = estimate_phi(feeds, zs, beta)
        println("VEM iter=", iter, "/", totaliter, " joint_prob=", joint_prob(feeds, thetas, zs, phi, p0, alpha, beta))
    end

    @check_used opts

    return zs, thetas, phi, pts
end


function likelihood(z, phi)
    log_phi = log(phi)
    T = length(z)
    value = 0.0
    for t=1:T
         
    end
    return value
end

export q_z, count_z, q_theta, E_step, estimate_phi, PVEM, VEM, likelihood
