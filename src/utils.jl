#Base.@kwdef 
mutable struct TypeMP
    type::String
    col::Bool        
    method::String

    function TypeMP(type::String="full",col::Bool=false,method::String="sum")
        if !(type in ["full","trunc"])
            error("Invalid algorithm type option")
        end
        if type == "trunc"
            if col
                if !(type in ["inorder","reverse","alternate","random"])
                    error("Invalid type.method parameter")
                end
            else
                if !(type in ["sum","random"])
                    error("Invalid type.method parameter")
                end
            end
        end
        new(type,col,method)
    end
end

