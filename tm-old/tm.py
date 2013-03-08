import numpy as np
from numpy.random.mtrand import dirichlet
from numpy.random import binomial, multinomial, random
import scipy as sp
from scipy.stats import rv_discrete, pearsonr
from misc import test_topics, reverse_dict, dir_logpdf, dir_mult, dir_pdf
from pylab import *
from pdb import set_trace   
import warnings
from cPickle import dump

def generate_docs(phi, ndocs, nwords_per_doc, alpha=0.1, p0=0.8):
    K, V = phi.shape

    theta = np.zeros((ndocs, K), dtype=float)

    switch = np.append([0], binomial(1, p0, ndocs - 1))
    switch = switch == 0

    samples = dirichlet([alpha] * K, size=int(switch.sum()))
    theta[switch] = samples

    last_theta = None
    for t in xrange(0, ndocs):
        if switch[t] == True:
            last_theta = theta[t]
            continue

        theta[t] = last_theta

    def gen_z(theta):
        z = np.repeat(np.arange(K),
            multinomial(nwords_per_doc, theta, size=1)[0])
        np.random.shuffle(z)
        return z 

    z = np.apply_along_axis(gen_z, 1, theta)

    def gen_w(z):
        return np.random.multinomial(1, phi[z]).nonzero()[0][0]

    w = np.vectorize(gen_w)(z)

    return w, z, theta, switch

def gen_phi(beta, K):
    return dirichlet(beta, size=K)

def q_theta(N, alpha, p0, L, resampling=True, smoothing=False):
    T, K = N.shape
    theta = np.zeros((T, L, K), dtype=float)

    pt = np.zeros((T, L), dtype=float)
    w = np.zeros((T, L), dtype=float)

    def trans_prob(curr, prev):
        if np.power(curr - prev, 2).sum() < 1e-10:
            return np.log(p0)
        return np.log(1-p0) + dir_logpdf(curr, alpha)

    def emiss_prob(n, theta):
        return (np.log(theta) * n).sum(axis=1)

    def prior_prob(theta):
        return dir_logpdf(theta, alpha)

    trans_prob_v = vectorize(trans_prob)

    def pt_t(th, n, p):
        a = p * (np.power(th, n)).prod()
        b = (1 - p) * dir_mult(alpha) / dir_mult(alpha + n)
        return a / (a + b)

    def theta_t(th, n, p):
        pt = pt_t(th, n, p)
        if binomial(1, pt) == 1:
            return (th, pt, np.log(pt))

        tt = dirichlet(alpha + n, 1)[0]
        return (tt, pt, np.log(1-pt) + dir_logpdf(tt, alpha + n))

    def resample(L, w):
        return np.repeat(np.arange(L), multinomial(L, np.exp(w)))

    def adjust(w):
        w -= w.max() 
        w -= np.log(np.exp(w).sum())
        return w

    for j in xrange(L):
        theta[0,j], pt[0,j], w[0,j] = theta_t(theta[-1,j], N[0], 0.0)
    if resampling:
        w[0] = np.apply_along_axis(prior_prob, 1, theta[0]) - w[0] +\
            emiss_prob(N[0, np.newaxis], theta[0])

    for t in xrange(1, T):
        if resampling:
            resample_idx = resample(L, w[t-1])
            w[t-1] = -np.log(L)
            theta[t-1] = theta[t-1, resample_idx]
            pt[t-1] = pt[t-1, resample_idx]

        for j in xrange(L):
            theta[t,j], pt[t,j], w[t,j] = theta_t(theta[t-1,j], N[t], p0)

        if resampling:
            w[t] = w[t-1] - w[t] + np.apply_along_axis(trans_prob, 1, theta[t], theta[t-1])
            w[t] += emiss_prob(N[t, np.newaxis], theta[t])
            w[t] = adjust(w[t])

    log_theta = None
    if resampling and smoothing:
        resample_idx = resample(L, w[T-1])
        theta[T-1] = theta[T-1, resample_idx]
        pt[T-1] = pt[T-1, resample_idx]

        for t in xrange(T-2, -1, -1):
            pw = w[t] + np.apply_along_axis(trans_prob, 1, theta[t+1], theta[t])
            pw = adjust(pw)
            resample_idx = resample(L, pw)
            theta[t] = theta[t, resample_idx]
            pt[t] = pt[t, resample_idx]
            w[t] = pw

    if resampling:
        wx = np.exp(w)
        log_theta = np.log(theta) * wx[:, :, np.newaxis]
        theta = theta * wx[:, :, np.newaxis]
        pt = pt * wx

        theta = theta.sum(axis=1)
        pt = pt.sum(axis=1)
        log_theta = log_theta.sum(axis=1)
    else:
        theta = np.apply_along_axis(np.mean, 1, theta)
        pt = np.apply_along_axis(np.mean, 1, pt)
    return (theta, pt, log_theta)

def q_z(w, log_theta, phi):
    (K, V) = phi.shape
    (T, N) = w.shape
    z = zeros((T, N, K))

    for k in xrange(K):
        z[:, :, k] = np.exp(log_theta[:, k, np.newaxis]) * phi[k, w]

    z = z / z.sum(axis=2)[:, :, np.newaxis]
    return z

def q_z_alt(w, theta, phi):
    (K, V) = phi.shape
    (T, N) = w.shape
    z = zeros((T, N, K))

    for k in xrange(K):
        z[:, :, k] = theta[:, k, np.newaxis] * phi[k, w]

    z = z / z.sum(axis=2)[:, :, np.newaxis]
    return z

def likelihood(w, z, theta, phi):
    (K, V) = phi.shape
    p = 0.0
    for k in xrange(K):
        p += (phi[k, w] * z[:, :, k]).sum()
    p += (theta * z.sum(axis=1)).sum()
    return p

def likelihood_strong(w, z, log_phi):
    return log_phi[z, w].sum() 

def count_z(z):
    K = z.max()+1
    def count(z_k):
        def count_k(k):
            return (z_k == k).sum()
        return vectorize(count_k)(np.arange(K))
    return np.apply_along_axis(count, 1, z)

def E_step(w, phi, alpha, beta, p0, L, n=None, theta=None, maxiter=100, resampling=True, smoothing=False):
    T, N = w.shape
    K, V = phi.shape

    log_theta = None

    if n == None:
        n = dirichlet(alpha, size=T) * N
    if theta == None:
        theta, pt, log_theta = q_theta(n, alpha, p0, L)
    # if theta == None:
    #     theta = dirichlet(alpha, size=T)
    #     log_theta = np.log(theta)
    # if n == None:
    #     n = theta * N

    log_phi = np.log(phi)

    pt = None
    likelihood_log = np.zeros(maxiter, dtype=float)
    theta_log = np.zeros((maxiter, T, K), dtype=float)

    for iteration in xrange(maxiter):
        z = q_z(w, log_theta, phi)
        #z = q_z_alt(w, theta, phi)
        n = z.sum(axis=1)
        new_theta, pt, new_log_theta = q_theta(n, alpha, p0, L, resampling=resampling,
            smoothing=smoothing) 

        #set_trace()
        diff = np.abs(theta - new_theta)
        avg_diff = diff.mean()
        max_diff = diff.max()

        likelihood_log[iteration] = likelihood(w, z, np.log(new_theta), log_phi)
        print 'iteration %d. avg diff: %f. max diff: %f. likelihood: %f' %\
         (iteration, avg_diff, max_diff, likelihood_log[iteration])

        theta_log[iteration] = theta
        log_theta = new_log_theta
        theta = new_theta

    return z, theta, pt, likelihood_log, theta_log

def M_step(w, z, beta):
    return None

def display_z(z, est_z, perm=None):
    n = count_z(z)
    if perm == None:
        perm = np.arange(n.shape[1])
    est_n = est_z.sum(axis=1)
    for t in xrange(T):
        print '%s %s %s' % (n[t], est_n[t, perm].round(), switch[t])

if __name__ == '__main__':
    phi, word2idx = test_topics()

    p0 = 0.8
    alpha = 0.1
    K, V = phi.shape
    K, V = 8, 100
    beta = 0.01
    phi = gen_phi(np.array([beta] * V), K)
    T = 100
    L = 100
    maxiter = 30

    w, z, theta, switch = generate_docs(phi, T, 20, alpha=alpha, p0=p0)
    alpha = np.array([alpha] * K)

    # print 'truth likelihood %f' % likelihood_strong(w, z, np.log(phi))
    # est_z = q_z(w, np.log(theta), phi)
    # display_z(z, est_z)
    # est_theta, pt, est_log_theta = q_theta(n, alpha, p0, L, resampling=True, smoothing=False)
    est_z, est_theta, pt, likelihood_log, theta_log = E_step(w, phi, alpha, beta, p0, L, maxiter=maxiter,
        resampling=True, smoothing=True)
    est_n = est_z.sum(axis=1)
    print 'mean theta error %f' % np.abs(est_theta - theta).sum(axis=1).mean()

    n = count_z(z)
    print 'mean n error %f' % np.abs(est_n - n).sum(axis=1).mean()

    best_iter = likelihood_log.argmax()
    est_theta = theta_log[best_iter]
    print 'best theta error %f' % np.abs(est_theta - theta).sum(axis=1).mean()

    not_switch = np.logical_not(switch)

    #with open("e_step_1", "w") as output:
    #    dump((alpha, beta, phi), output)
    #    dump((w, z, theta, switch), output)
    #    dump((est_n, est_theta, pt), output)

    X = np.arange(maxiter)
    #X = np.arange(T)

    theta_error = np.zeros((maxiter), dtype=float)
    for iteration in xrange(maxiter):
        theta_error[iteration] = np.abs(theta_log[iteration] - theta).sum(axis=1).mean()
    print str.format('corellation: {0}', pearsonr(theta_error, likelihood_log))

    #plot(X, likelihood_log, color="blue")
    plot(X, theta_error, color="red")
    #scatter(X, theta_error)
    #scatter(X[not_switch], pt[not_switch], color="blue")
    #scatter(X[switch], pt[switch], color="red")

    show()
