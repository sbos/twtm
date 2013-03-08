import numpy as np
from scipy.special import gammaln


def test_topics():
    w = {}

    def add_word(word):
        w[word] = len(w)

    add_word('beer')
    add_word('player')
    add_word('ball')
    add_word('girl')
    add_word('fun')
    add_word('holiday')
    add_word('game')
    add_word('burger')
    add_word('boobs')
    add_word('legs')
    add_word('some')
    add_word('lot')
    add_word('one')

    V = len(w)
    K = 3
    phi = np.zeros((K, V), dtype=float)

    def set_prob(k, word, p):
        phi[k, w[word]] = p

    set_prob(0, 'beer', 5)
    set_prob(0, 'player', 9)
    set_prob(0, 'ball', 9)
    set_prob(0, 'girl', 4)
    set_prob(0, 'fun', 6)
    set_prob(0, 'holiday', 3)
    set_prob(0, 'game', 8.5)
    set_prob(0, 'burger', 4)
    set_prob(0, 'boobs', 1)
    set_prob(0, 'legs', 5)

    set_prob(1, 'beer', 3)
    set_prob(1, 'player', 0.5)
    set_prob(1, 'ball', 0.3)
    set_prob(1, 'girl', 11)
    set_prob(1, 'fun', 7)
    set_prob(1, 'holiday', 2)
    set_prob(1, 'game', 2)
    set_prob(1, 'burger', 2)
    set_prob(1, 'boobs', 7)
    set_prob(1, 'legs', 8)

    set_prob(2, 'beer', 8)
    set_prob(2, 'player', 2)
    set_prob(2, 'ball', 4)
    set_prob(2, 'girl', 5)
    set_prob(2, 'fun', 6)
    set_prob(2, 'holiday', 10)
    set_prob(2, 'game', 3)
    set_prob(2, 'burger', 6)
    set_prob(2, 'boobs', 0.9)
    set_prob(2, 'legs', 2)

    for k in xrange(K):
        set_prob(k, 'some', 1)
        set_prob(k, 'one', 1)
        set_prob(k, 'lot', 1)

    phi = phi / np.sum(phi, axis=1)[:, np.newaxis]

    return (phi, w)


def reverse_dict(words):
    return dict((idx, word) for word, idx in words.items())

def dir_logmult(alpha):
    return gammaln(alpha.sum()) - gammaln(alpha).sum()

def dir_mult(alpha):
    return np.exp(dir_logmult(alpha))

def dir_logpdf(theta, alpha):
    return dir_logmult(alpha) + (np.log(theta) * (alpha - 1)).sum()

def dir_pdf(theta, alpha):
    return np.exp(dir_logpdf(theta, alpha))

