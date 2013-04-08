function joint_prob(feeds, thetas, zs, phi, p0, alpha, beta)
    prob = 0.0
    
    K, V = size(phi)
    for k=1:K
        prob += logpdf(Dirichlet(beta * ones(V)), vec(phi[k, :]))
    end

    F = length(feeds)
    prob += @parallel (+) for f=1:F
        docs = feeds[f]
        theta = thetas[f]
        z = zs[f]

        V, T = size(docs)
        log_phi = log(phi)
        n = count_z(docs, z)


        fprob = 0.0
        for t=1:T
            if t > 1
                fprob += log(p0 * pdf(Dirichlet(alpha), vec(theta[t-1, :])) + (1-p0) * pdf(Dirichlet(alpha), vec(theta[t, :])))
            else
                fprob += logpdf(Dirichlet(alpha), vec(theta[t, :]))
            end

            words = spcolidx(docs, t)
            word_count = spcolval(docs, t)

            for k=1:K
                fprob += n[t, k] * log(theta[t, k])

                z_t_k = spcolval(z[t], k)
                assert(length(z_t_k) == length(words))

                for v=1:length(words)
                    fprob += z_t_k[v] * log_phi[k, words[v]] * word_count[v]
                end
            end
        end

        fprob
    end

    return prob
end

