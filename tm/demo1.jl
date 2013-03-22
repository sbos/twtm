include("twtm.jl")
using TwitterTopicModeling

K = 3
V = 15
alpha = ones(K) * 0.1
beta = 0.01
T = 100
p0 = 0.8

phi = gen_phi(K, V, beta)
theta = gen_theta(T, alpha, p0)
docs = gen_docs(17, theta, phi)
