for Typ in (:Ldiv, :Rdiv)
    @eval begin
        struct $Typ{StyleA, StyleB, AType, BType}
            A::AType
            B::BType
        end

        @inline $Typ{StyleA, StyleB}(A::AType, B::BType) where {StyleA,StyleB,AType,BType} =
            $Typ{StyleA,StyleB,AType,BType}(A,B)

        @inline $Typ(A::AType, B::BType) where {AType,BType} =
            $Typ{typeof(MemoryLayout(AType)),typeof(MemoryLayout(BType)),AType,BType}(A, B)

        @inline BroadcastStyle(::Type{<:$Typ}) = ApplyBroadcastStyle()
        @inline broadcastable(M::$Typ) = M

        similar(A::$Typ, ::Type{T}, axes) where T = similar(Array{T}, axes)
        similar(A::$Typ, ::Type{T}) where T = similar(A, T, axes(A))
        similar(A::$Typ) = similar(A, eltype(A))

        @inline copy(M::$Typ) = copyto!(similar(M), M)
        @inline materialize(M::$Typ) = copy(instantiate(M))
    end
end

@inline _ldivaxes(::Tuple{}, ::Tuple{}) = ()
@inline _ldivaxes(::Tuple{}, Bax::Tuple) = Bax
@inline _ldivaxes(::Tuple{<:Any}, ::Tuple{<:Any}) = ()
@inline _ldivaxes(::Tuple{<:Any}, Bax::Tuple{<:Any,<:Any}) = (OneTo(1),last(Bax))
@inline _ldivaxes(Aax::Tuple{<:Any,<:Any}, ::Tuple{<:Any}) = (last(Aax),)
@inline _ldivaxes(Aax::Tuple{<:Any,<:Any}, Bax::Tuple{<:Any,<:Any}) = (last(Aax),last(Bax))

@inline ldivaxes(A, B) = _ldivaxes(axes(A), axes(B))

@inline axes(L::Ldiv) = ldivaxes(L.A, L.B)
@inline size(L::Ldiv) = map(length, axes(L))
@inline length(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =size(L.A, 2)

@inline size(L::Rdiv) = (size(L.A, 1),size(L.B,1))
@inline axes(L::Rdiv) = (axes(L.A, 1),axes(L.B,1))

@inline ndims(L::Ldiv) = ndims(L.B)
@inline ndims(L::Rdiv) = 2
@inline eltype(M::Ldiv) = promote_type(Base.promote_op(inv, eltype(M.A)), eltype(M.B))
@inline eltype(M::Rdiv) = promote_type(eltype(M.A), Base.promote_op(inv, eltype(M.B)))

# Lazy getindex

getindex(L::Ldiv, k...) = _getindex(indextype(L), L, k)
concretize(L::AbstractArray) = convert(Array,L)
concretize(L::Ldiv) = ldiv(concretize(L.A), concretize(L.B))
_getindex(::Type{Tuple{I}}, L::Ldiv, (k,)::Tuple{I}) where I = concretize(L)[k]
_getindex(::Type{Tuple{I,J}}, L::Ldiv, (k,j)::Tuple{Colon,J}) where {I,J} = Ldiv(L.A, L.B[:,j])
_getindex(::Type{Tuple{I,J}}, L::Ldiv, (k,j)::Tuple{I,J}) where {I,J} = L[:,j][k]

check_ldiv_axes(A, B) =
    axes(A,1) == axes(B,1) || throw(DimensionMismatch("First axis of A, $(axes(A,1)), and first axis of B, $(axes(B,1)) must match"))

check_rdiv_axes(A, B) =
    axes(A,2) == axes(B,2) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and second axis of B, $(axes(B,2)) must match"))



@inline function instantiate(L::Ldiv)
    check_ldiv_axes(L.A, L.B)
    Ldiv(instantiate(L.A), instantiate(L.B))
end

@inline function instantiate(L::Rdiv)
    check_rdiv_axes(L.A, L.B)
    Rdiv(instantiate(L.A), instantiate(L.B))
end

__ldiv!(::Mat, ::Mat, B) where Mat = error("Overload materialize!(::Ldiv{$(typeof(MemoryLayout(Mat))),$(typeof(MemoryLayout(B)))})")
__ldiv!(::Mat, ::Mat, B::LayoutArray) where Mat = error("Overload materialize!(::Ldiv{$(typeof(MemoryLayout(Mat))),$(typeof(MemoryLayout(B)))})")
__ldiv!(_, F, B) = LinearAlgebra.ldiv!(F, B)
@inline _ldiv!(A, B) = __ldiv!(A, factorize(A), B)
@inline _ldiv!(A::Factorization, B) = LinearAlgebra.ldiv!(A, B)
@inline _ldiv!(A::Factorization, B::LayoutArray) = error("Overload materialize!(::Ldiv{$(typeof(MemoryLayout(A))),$(typeof(MemoryLayout(B)))})")

@inline _ldiv!(dest, A, B) = ldiv!(dest, factorize(A), B)
@inline _ldiv!(dest, A::Factorization, B) = LinearAlgebra.ldiv!(dest, A, B)
@inline _ldiv!(dest, A::Transpose{<:Any,<:Factorization}, B) = LinearAlgebra.ldiv!(dest, A, B)
@inline _ldiv!(dest, A::Adjoint{<:Any,<:Factorization}, B) = LinearAlgebra.ldiv!(dest, A, B)

@inline ldiv(A, B) = materialize(Ldiv(A,B))
@inline rdiv(A, B) = materialize(Rdiv(A,B))

@inline ldiv!(A, B) = materialize!(Ldiv(A,B))
@inline rdiv!(A, B) = materialize!(Rdiv(A,B))

@inline ldiv!(C, A, B) = copyto!(C, Ldiv(A,B))
@inline rdiv!(C, A, B) = copyto!(C, Rdiv(A,B))

@inline materialize!(M::Ldiv) = _ldiv!(M.A, M.B)
@inline materialize!(M::Rdiv) = ldiv!(M.B', M.A')'
@inline copyto!(dest::AbstractArray, M::Rdiv) = copyto!(dest', Ldiv(M.B', M.A'))'
@inline copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, copy(M.B))

const MatLdivVec{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractVector{V}}
const MatLdivMat{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatLdivVec{styleA, styleB, T<:BlasFloat} = MatLdivVec{styleA, styleB, T, T}
const BlasMatLdivMat{styleA, styleB, T<:BlasFloat} = MatLdivMat{styleA, styleB, T, T}

const MatRdivMat{styleA, styleB, T, V} = Rdiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatRdivMat{styleA, styleB, T<:BlasFloat} = MatRdivMat{styleA, styleB, T, T}

materialize!(M::Ldiv{ScalarLayout}) = Base.invoke(LinearAlgebra.ldiv!, Tuple{Number,AbstractArray}, M.A, M.B)
materialize!(M::Rdiv{<:Any,ScalarLayout}) = Base.invoke(LinearAlgebra.rdiv!, Tuple{AbstractArray,Number}, M.A, M.B)

function materialize!(M::Ldiv{ScalarLayout,<:SymmetricLayout})
    ldiv!(M.A, symmetricdata(M.B))
    M.B
end
function materialize!(M::Ldiv{ScalarLayout,<:HermitianLayout})
    ldiv!(M.A, hermitiandata(M.B))
    M.B
end
function materialize!(M::Rdiv{<:SymmetricLayout,ScalarLayout})
    rdiv!(symmetricdata(M.A), M.B)
    M.A
end
function materialize!(M::Rdiv{<:HermitianLayout,ScalarLayout})
    rdiv!(hermitiandata(M.A), M.B)
    M.A
end

macro _layoutldiv(Typ)
    ret = quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = ArrayLayouts.ldiv!(A,x)
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = ArrayLayouts.ldiv!(A,x)
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = ArrayLayouts.ldiv!(A,x)
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = ArrayLayouts.ldiv!(A,x)

        LinearAlgebra.ldiv!(A::Factorization, x::$Typ) = ArrayLayouts.ldiv!(A,x)

        LinearAlgebra.ldiv!(A::Bidiagonal, B::$Typ) = ArrayLayouts.ldiv!(A,B)


        Base.:\(A::$Typ, x::AbstractVector) = ArrayLayouts.ldiv(A,x)
        Base.:\(A::$Typ, x::AbstractMatrix) = ArrayLayouts.ldiv(A,x)

        Base.:\(x::AbstractMatrix, A::$Typ) = ArrayLayouts.ldiv(x,A)
        Base.:\(x::UpperTriangular, A::$Typ) = ArrayLayouts.ldiv(x,A)
        Base.:\(x::LowerTriangular, A::$Typ) = ArrayLayouts.ldiv(x,A)
        Base.:\(x::Diagonal, A::$Typ) = ArrayLayouts.ldiv(x,A)

        Base.:\(A::Bidiagonal{<:Number}, B::$Typ{<:Number}) = ArrayLayouts.ldiv(A,B)
        Base.:\(A::Bidiagonal, B::$Typ) = ArrayLayouts.ldiv(A,B)
        Base.:\(transA::Transpose{<:Number,<:Bidiagonal{<:Number}}, B::$Typ{<:Number}) = ArrayLayouts.ldiv(transA,B)
        Base.:\(transA::Transpose{<:Any,<:Bidiagonal}, B::$Typ) = ArrayLayouts.ldiv(transA,B)
        Base.:\(adjA::Adjoint{<:Number,<:Bidiagonal{<:Number}}, B::$Typ{<:Number}) = ArrayLayouts.ldiv(adjA,B)
        Base.:\(adjA::Adjoint{<:Any,<:Bidiagonal}, B::$Typ) = ArrayLayouts.ldiv(adjA,B)

        Base.:\(x::$Typ, A::$Typ) = ArrayLayouts.ldiv(x,A)

        Base.:/(A::$Typ, x::AbstractVector) = ArrayLayouts.rdiv(A,x)
        Base.:/(A::$Typ, x::AbstractMatrix) = ArrayLayouts.rdiv(A,x)

        Base.:/(x::AbstractMatrix, A::$Typ) = ArrayLayouts.rdiv(x,A)
        Base.:/(x::Diagonal, A::$Typ) = ArrayLayouts.rdiv(x,A)

        Base.:/(x::$Typ, A::$Typ) = ArrayLayouts.rdiv(x,A)
    end
    if Typ ≠ :LayoutVector
        ret = quote
            $ret
            Base.:\(A::$Typ, x::LayoutVector) = ArrayLayouts.ldiv(A,x)
        end
    end
    if Typ ≠ :LayoutMatrix
        ret = quote
            $ret
            Base.:\(A::$Typ, x::LayoutMatrix) = ArrayLayouts.ldiv(A,x)
            Base.:/(x::LayoutMatrix, A::$Typ) = ArrayLayouts.rdiv(x,A)
        end
    end
    esc(ret)
end

macro layoutldiv(Typ)
    esc(quote
        ArrayLayouts.@_layoutldiv $Typ
        ArrayLayouts.@_layoutldiv UpperTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutldiv UnitUpperTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutldiv LowerTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutldiv UnitLowerTriangular{T, <:$Typ{T}} where T

        ArrayLayouts.@_layoutldiv UpperTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
        ArrayLayouts.@_layoutldiv UnitUpperTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
        ArrayLayouts.@_layoutldiv LowerTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
        ArrayLayouts.@_layoutldiv UnitLowerTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
    end)
end

@_layoutldiv LayoutVector