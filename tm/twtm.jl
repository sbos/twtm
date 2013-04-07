module TwitterTopicModeling

Feed = SparseMatrixCSC{Int, Int}
WordAssignment = Vector{SparseMatrixCSC{Float64, Int}}

include("util.jl")
include("generate.jl")
include("learning.jl")
include("load.jl")
include("save.jl")

export Feed, WordAssignment
end
