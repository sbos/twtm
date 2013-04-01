log_dirmult(alpha) = lgamma(sum(alpha)) - sum(lgamma(alpha))

function resample(w)
    dist = Categorical(w)
    return rand(dist, length(w))
end

spcolidx(m, col) = m.rowval[m.colptr[col]:m.colptr[col+1]-1]

spcolval(m, col) =  m.nzval[m.colptr[col]:m.colptr[col+1]-1]

function adjust(w)
    w -= max(w)
    w -= log(sum(exp(w)))
    return w
end

