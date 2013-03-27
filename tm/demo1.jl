include("twtm.jl")
using TwitterTopicModeling

require("Options")
using OptionsMod

K = 3
V = 100
alpha = ones(K) * 0.1
beta = 0.01
T = 20
p0 = 0.8

phi = gen_phi(K, V, beta)
theta = gen_theta(T, alpha, p0)
docs = gen_docs(17, theta, phi)

z = q_z(log(theta), log(phi), docs.colptr, docs.rowval)
n = count_z(docs, z)

est_theta = zeros(T, K)
for t=1:T
    est_theta[t, :] = n[t, :] / sum(n[t, :])
end
println("theta(z) reconstruction error (z estimated from true theta):")
println(sum(abs(est_theta - theta), 1) / T)

L = 30
est_log_theta, est_theta, est_pt = q_theta(n, alpha, p0, L, Options())

println("theta(z) estimation error (z's are true; pfilter is used):")
println(sum(abs(est_theta - theta), 1) / T)

for t=1:T
#println(t)
#println(est_theta[t, :])
#println(theta[t, :])
end
