mutable struct TypeMP
    type::String
    col::Bool        
    method::String

    function TypeMP(;type::String="full",col::Bool=false,method::String="sum")
        if !(type in ["full","trunc"])
            error("Invalid algorithm type option")
        end
        if type == "trunc"
            if col
                if !(method in ["inorder","reverse","alternate","random"])
                    error("Invalid type.method parameter")
                end
            else
                if !(method in ["sum","random"])
                    error("Invalid type.method parameter")
                end
            end
        end
        new(type,col,method)
    end
end


mutable struct ZinfoInstance
    yk_index::Vector{Int64}
    V_index::Vector{Int64}
    end_V::Int64

    function ZinfoInstance(nmaxits,end_V)
        new(zeros(Int64,nmaxits), zeros(Int64,nmaxits), end_V)
    end
end

mutable struct Cell
    array::Vector{Any}

    function Cell(length::Int64) 
        new(Vector{Any}(undef,length))
    end
end

function Base.getindex(C::Cell, i::Int)
    return C.array[i]
end

function Base.setindex!(C::Cell, v, i::Int64)
    return C.array[i] = v
end

Base.firstindex(C::Cell) = 1

Base.lastindex(C::Cell) = length(C.array)



