log_dirmult(alpha) = lgamma(sum(alpha)) - sum(lgamma(alpha))

function resample(w)
    dist = Categorical(w)
    return rand(dist, length(w))
end

function adjust(w)
    w -= max(w)
    w -= log(sum(exp(w)))
    return w
end

