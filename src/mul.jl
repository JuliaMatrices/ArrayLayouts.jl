struct Mul{StyleA, StyleB, AA, BB}
    A::AA
    B::BB
end

@inline Mul{StyleA,StyleB}(A::AA, B::BB) where {StyleA,StyleB,AA,BB} =
    Mul{StyleA,StyleB,AA,BB}(A,B)

@inline Mul(A, B) = Mul{typeof(MemoryLayout(A)), typeof(MemoryLayout(B))}(A, B)

@inline _mul_eltype(A) = A
@inline _mul_eltype(A, B) = Base.promote_op(*, A, B)
@inline _mul_eltype(A, B, C, D...) = _mul_eltype(Base.promote_op(*, A, B), C, D...)

@inline eltype(::Mul{StyleA,StyleB,AA,BB}) where {StyleA,StyleB,AA,BB} = _mul_eltype(eltype(AA), eltype(BB))

size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]
length(M::Mul) = prod(size(M))
size(M::Mul) = map(length,axes(M))
axes(M::Mul{<:Any,<:Any,<:AbstractMatrix,<:AbstractVector}) = (axes(M.A,1),)
axes(M::Mul{<:Any,<:Any,<:AbstractMatrix,<:AbstractMatrix}) = (axes(M.A,1),axes(M.B,2))

"""
   mulreduce(M::Mul)

returns a lower level lazy multiplication object such as `MulAdd`, `Lmul` or `Rmul`.
The Default is `MulAdd`
"""
mulreduce(M::Mul) = MulAdd(M)

similar(M::Mul, ::Type{T}, axes) where {T,N} = similar(mulreduce(M), T, axes)
similar(M::Mul, ::Type{T}) where T = similar(mulreduce(M), T)
similar(M::Mul) = similar(mulreduce(M))

check_mul_axes(A) = nothing
_check_mul_axes(::Number, ::Number) = nothing
_check_mul_axes(::Number, _) = nothing
_check_mul_axes(_, ::Number) = nothing
_check_mul_axes(A, B) = axes(A,2) == axes(B,1) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and first axis of B, $(axes(B,1)) must match"))
function check_mul_axes(A, B, C...) 
    _check_mul_axes(A, B)
    check_mul_axes(B, C...)
end

# we need to special case AbstractQ as it allows non-compatiple multiplication
function check_mul_axes(A::AbstractQ, B, C...) 
    axes(A.factors, 1) == axes(B, 1) || axes(A.factors, 2) == axes(B, 1) ||  
        throw(DimensionMismatch("First axis of B, $(axes(B,1)) must match either axes of A, $(axes(A))"))
    check_mul_axes(B, C...)
end

function instantiate(M::Mul)
    @boundscheck check_mul_axes(M.A, M.B)
    M
end

materialize(M::Mul) = copy(instantiate(M))
@inline mul(A::AbstractArray, B::AbstractArray) = copy(Mul(A,B))

copy(M::Mul) = copy(mulreduce(M))
@inline copyto!(dest::AbstractArray, M::Mul) = copyto!(dest, mulreduce(M))
mul!(dest::AbstractArray, A::AbstractArray, B::AbstractArray) = copyto!(dest, Mul(A,B))


broadcastable(M::Mul) = M


macro veclayoutmul(Typ)
    ret = quote
        Base.:*(A::AbstractMatrix, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::Adjoint{<:Any,<:AbstractMatrix{T}}, B::$Typ{S}) where {T,S} = ArrayLayouts.mul(A,B)
        Base.:*(A::Transpose{<:Any,<:AbstractMatrix{T}}, B::$Typ{S}) where {T,S} = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec{T}, B::$Typ{T}) where T<:Real = ArrayLayouts.mul(A,B)

        Base.:*(A::LinearAlgebra.AbstractQ, B::$Typ) = ArrayLayouts.mul(A,B)
    end
    for Struc in (:AbstractTriangular, :Diagonal)
        ret = quote
            $ret

            Base.:*(A::LinearAlgebra.$Struc, B::$Typ) = ArrayLayouts.mul(A,B)
        end
    end
    for Mod in (:Adjoint, :Transpose)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                ArrayLayouts.mul!(dest,A,b)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LinearAlgebra.AbstractTriangular) = ArrayLayouts.mul(A,B)
        end
    end

    esc(ret)
end

macro layoutmul(Typ)
    ret = quote
        LinearAlgebra.mul!(dest::AbstractVector, A::$Typ, b::AbstractVector) =
            ArrayLayouts.mul!(dest,A,b)

        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::AbstractMatrix) =
            ArrayLayouts.mul!(dest,A,b)
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Typ) =
            ArrayLayouts.mul!(dest,A,b)

        Base.:*(A::$Typ, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::AbstractMatrix) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::AbstractVector) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::ArrayLayouts.LayoutVector) = ArrayLayouts.mul(A,B)
        Base.:*(A::AbstractMatrix, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)

        Base.:*(A::LinearAlgebra.AbstractQ, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::LinearAlgebra.AbstractQ) = ArrayLayouts.mul(A,B)
    end
    for Struc in (:AbstractTriangular, :Diagonal)
        ret = quote
            $ret

            Base.:*(A::LinearAlgebra.$Struc, B::$Typ) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Typ, B::LinearAlgebra.$Struc) = ArrayLayouts.mul(A,B)
        end
    end
    for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Mod{<:Any,<:AbstractMatrix}) =
                ArrayLayouts.mul!(dest,A,b)

            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                ArrayLayouts.mul!(dest,A,b)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix) = ArrayLayouts.mul(A,B)
            Base.:*(A::AbstractMatrix, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::ArrayLayouts.LayoutVector) = ArrayLayouts.mul(A,B)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Typ, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal) = ArrayLayouts.mul(A,B)
            Base.:*(A::Diagonal, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)

            Base.:*(A::LinearAlgebra.AbstractTriangular, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LinearAlgebra.AbstractTriangular) = ArrayLayouts.mul(A,B)
        end
    end

    esc(ret)
end

@veclayoutmul LayoutVector


###
# Dot
###

struct Dot{StyleA,StyleB,ATyp,BTyp}
    A::ATyp
    B::BTyp
end

@inline Dot(A::ATyp,B::BTyp) where {ATyp,BTyp} = Dot{typeof(MemoryLayout(ATyp)), typeof(MemoryLayout(BTyp)), ATyp, BTyp}(A, B)
@inline copy(d::Dot{<:Any,<:Any,<:AbstractArray,<:AbstractArray}) = Base.invoke(dot, Tuple{AbstractArray,AbstractArray}, d.A, d.B)
@inline materialize(d::Dot) = copy(instantiate(d))
@inline Dot(M::Mul{<:DualLayout,<:Any,<:AbstractMatrix,<:AbstractVector}) = materialize(Dot(M.A', M.B))
@inline mulreduce(M::Mul{<:DualLayout,<:Any,<:AbstractMatrix,<:AbstractVector}) = Dot(M)

@inline dot(a::LayoutArray, b::AbstractArray) = copy(Dot(a,b))
@inline dot(a::SubArray{<:Any,N,<:LayoutArray}, b::AbstractArray) where N = materialize(Dot(a,b))

