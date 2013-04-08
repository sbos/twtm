module TwitterTopicModeling

require("Options")
using OptionsMod

Feed = SparseMatrixCSC{Int, Int}
WordAssignment = Vector{SparseMatrixCSC{Float64, Int}}

include("util.jl")
include("generate.jl")
include("score.jl")
include("learning.jl")
include("load.jl")
include("save.jl")

export Feed, WordAssignment
end
