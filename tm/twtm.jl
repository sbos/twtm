module TwitterTopicModeling

Feed = SparseMatrixCSC{Int, Int}
WordAssignment = Array{SparseMatrixCSC{Float64, Int}, 1}

include("util.jl")
include("generate.jl")
include("learning.jl")
include("load.jl")

end
