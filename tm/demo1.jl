include("twtm.jl")
using TwitterTopicModeling

require("Options")
using OptionsMod

K = 3
V = 20
alpha = ones(K) * 0.1
beta = 0.01
T = 100
p0 = 0.8

phi = gen_phi(V, K, beta)

theta, pt = gen_theta(T, alpha, p0)
docs, n = gen_docs(20, theta, phi)

function print_docs()
    println("generated documents")
    for t=1:T
        println("t=", t, " ", "theta=", vec(theta[t, :]))
        println("w[t]=", vec(dense(docs[:, t])))
    end
end

z = q_z(log(theta), log(phi), docs.colptr, docs.rowval)
est_n = count_z(docs, z)

est_theta = zeros(T, K)
for t=1:T
    est_theta[t, :] = est_n[t, :] / sum(est_n[t, :])
end
println("theta(z) reconstruction error (z estimated from true theta):")
println(mean(sum(abs(est_theta - theta), 2)))
println("n estimation error: ", mean(sum(abs(est_n - n), 2)))

L = 50
fopts = @options filtering=true smoothing=true
est_log_theta, est_theta, est_pt = q_theta(n, alpha, p0, L, fopts)

println("switch detection error: ", mean(abs(est_pt - pt)))

println("theta(z) estimation error (z's are true; pfilter is used):")
println(mean(sum(abs(est_theta - theta), 2)))

est_theta, est_z, est_pt = E_step(docs, alpha, phi, p0, L, 20, fopts)

println("theta(z) estimation error (full E-step, phi is true):")
println(mean(sum(abs(est_theta - theta), 2)))

est_phi = estimate_phi(docs, z, beta)
println("phi estimation error (z's are from E-step): ", mean(abs(est_phi - phi)))

est_z, est_theta, est_phi, est_pt = VEM(docs, alpha, beta, p0, L, 10, 30, fopts)

println(phi)
println(est_phi)
