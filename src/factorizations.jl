abstract type AbstractQRLayout <: MemoryLayout end

"""
   QRCompactWYLayout{SLAY,TLAY}()

represents a Compact-WY QR factorization whose 
factors are stored with layout SLAY and τ stored with layout TLAY
"""
struct QRCompactWYLayout{SLAY,TLAY} <: AbstractQRLayout end
"""
    QRPackedLayout{SLAY,TLAY}()

represents a Packed QR factorization whose 
factors are stored with layout SLAY and τ stored with layout TLAY
"""
struct QRPackedLayout{SLAY,TLAY} <: AbstractQRLayout end

MemoryLayout(::Type{<:LinearAlgebra.QRCompactWY{<:Any,MAT}}) where MAT = 
    QRCompactWYLayout{typeof(MemoryLayout(MAT)),DenseColumnMajor}()
MemoryLayout(::Type{<:LinearAlgebra.QR{<:Any,MAT}}) where MAT = 
    QRPackedLayout{typeof(MemoryLayout(MAT)),DenseColumnMajor}()

function materialize!(L::Ldiv{<:QRCompactWYLayout,<:Any,<:Any,<:AbstractVector})
    A,b = L.A, L.B
    ldiv!(UpperTriangular(A.R), view(lmul!(adjoint(A.Q), b), 1:size(A, 2)))
    b
end

function materialize!(L::Ldiv{<:QRCompactWYLayout,<:Any,<:Any,<:AbstractMatrix})
    A,B = L.A, L.B
    ldiv!(UpperTriangular(A.R), view(lmul!(adjoint(A.Q), B), 1:size(A, 2), 1:size(B, 2)))
    B
end

# Julia implementation similar to xgelsy
function materialize!(Ldv::Ldiv{<:QRPackedLayout,<:Any,<:Any,<:AbstractMatrix{T}}) where T
    A,B = Ldv.A,Ldv.B
    m, n = size(A)
    minmn = min(m,n)
    mB, nB = size(B)
    lmul!(adjoint(A.Q), view(B, 1:m, :))
    mB ≥ n || throw(DimensionMismatch("Number of rows in B must exceed number of columns in A"))
    R = A.R
    @inbounds begin
        if n > m # minimum norm solution
            τ = zeros(T,m)
            for k = m:-1:1 # Trapezoid to triangular by elementary operation
                x = view(R, k, [k; m + 1:n])
                τk = reflector!(x)
                τ[k] = conj(τk)
                for i = 1:k - 1
                    vRi = R[i,k]
                    for j = m + 1:n
                        vRi += R[i,j]*x[j - m + 1]'
                    end
                    vRi *= τk
                    R[i,k] -= vRi
                    for j = m + 1:n
                        R[i,j] -= vRi*x[j - m + 1]
                    end
                end
            end
        end
        ldiv!(UpperTriangular(view(R, :, 1:minmn)), view(B, 1:minmn, :))
        if n > m # Apply elementary transformation to solution
            B[m + 1:mB,1:nB] .= zero(T)
            for j = 1:nB
                for k = 1:m
                    vBj = B[k,j]
                    for i = m + 1:n
                        vBj += B[i,j]*R[k,i]'
                    end
                    vBj *= τ[k]
                    B[k,j] -= vBj
                    for i = m + 1:n
                        B[i,j] -= R[k,i]*vBj
                    end
                end
            end
        end
    end
    return B
end
function materialize!(Ldv::Ldiv{<:QRPackedLayout,<:Any,<:Any,<:AbstractVector{T}}) where T
    ldiv!(Ldv.A, reshape(Ldv.B, length(Ldv.B), 1))
    Ldv.B
end



abstract type AbstractQLayout <: MemoryLayout end
struct QRPackedQLayout{SLAY,TLAY} <: AbstractQLayout end
struct AdjQRPackedQLayout{SLAY,TLAY} <: AbstractQLayout end
struct QRCompactWYQLayout{SLAY,TLAY} <: AbstractQLayout end
struct AdjQRCompactWYQLayout{SLAY,TLAY} <: AbstractQLayout end

MemoryLayout(::Type{<:LinearAlgebra.QRPackedQ{<:Any,S}}) where {S,T} = 
    QRPackedQLayout{typeof(MemoryLayout(S)),DenseColumnMajor}()
MemoryLayout(::Type{<:LinearAlgebra.QRCompactWYQ{<:Any,S}}) where {S,T} = 
    QRCompactWYQLayout{typeof(MemoryLayout(S)),DenseColumnMajor}()

adjointlayout(::Type, ::QRPackedQLayout{SLAY,TLAY}) where {SLAY,TLAY} = AdjQRPackedQLayout{SLAY,TLAY}()
adjointlayout(::Type, ::QRCompactWYQLayout{SLAY,TLAY}) where {SLAY,TLAY} = AdjQRCompactWYQLayout{SLAY,TLAY}()

copy(M::Lmul{<:AbstractQLayout}) = copyto!(similar(M), M)
mulreduce(M::Mul{<:AbstractQLayout,<:AbstractQLayout}) = MulAdd(M)
mulreduce(M::Mul{<:AbstractQLayout}) = Lmul(M)
mulreduce(M::Mul{<:Any,<:AbstractQLayout}) = Rmul(M)

function copyto!(dest::AbstractArray{T}, M::Lmul{<:AbstractQLayout}) where T
    A,B = M.A,M.B
    if size(dest,1) == size(B,1) 
        copyto!(dest, B)
    else
        copyto!(view(dest,1:size(B,1),:), B)
        zero!(@view(dest[size(B,1)+1:end,:]))
    end
    lmul!(A,dest)
end

function copyto!(dest::AbstractArray, M::Ldiv{<:AbstractQLayout})
    A,B = M.A,M.B
    copyto!(dest, B)
    ldiv!(A,dest)
end

materialize!(M::Lmul{LAY}) where LAY<:AbstractQLayout = error("Overload materialize!(::Lmul{$(LAY)})")
materialize!(M::Rmul{LAY}) where LAY<:AbstractQLayout = error("Overload materialize!(::Rmul{$(LAY)})")

materialize!(M::Ldiv{<:AbstractQLayout}) = materialize!(Lmul(M.A',M.B))

materialize!(M::Lmul{<:QRPackedQLayout{<:AbstractColumnMajor,<:AbstractColumnMajor},<:AbstractColumnMajor,<:AbstractMatrix{T},<:AbstractVecOrMat{T}}) where T<:BlasFloat = 
    LAPACK.ormqr!('L','N',M.A.factors,M.A.τ,M.B)    

materialize!(M::Lmul{<:QRCompactWYQLayout{<:AbstractColumnMajor,<:AbstractColumnMajor},<:AbstractColumnMajor,<:AbstractMatrix{T},<:AbstractVecOrMat{T}}) where T<:BlasFloat = 
    LAPACK.gemqrt!('L','N',M.A.factors,M.A.T,M.B)

function materialize!(M::Lmul{<:QRCompactWYQLayout{<:AbstractColumnMajor,<:AbstractColumnMajor},<:AbstractRowMajor,<:AbstractMatrix{T},<:AbstractMatrix{T}}) where T<:BlasReal 
    LAPACK.gemqrt!('R','T',M.A.factors,M.A.T,transpose(M.B))
    M.B
end
function materialize!(M::Lmul{<:QRCompactWYQLayout{<:AbstractColumnMajor,<:AbstractColumnMajor},<:ConjLayout{<:AbstractRowMajor},<:AbstractMatrix{T},<:AbstractMatrix{T}}) where T<:BlasComplex
    LAPACK.gemqrt!('R','C',M.A.factors,M.A.T,(M.B)')
    M.B
end


function materialize!(M::Lmul{<:QRPackedQLayout})
    A,B = M.A, M.B
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    @inbounds begin
        for k = min(mA,nA):-1:1
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:mB
                    vBj += conj(Afactors[i,k])*B[i,j]
                end
                vBj = A.τ[k]*vBj
                B[k,j] -= vBj
                for i = k+1:mB
                    B[i,j] -= Afactors[i,k]*vBj
                end
            end
        end
    end
    B
end


### QcB
materialize!(M::Lmul{<:AdjQRPackedQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractStridedLayout,<:AbstractMatrix{T},<:AbstractVecOrMat{T}}) where T<:BlasFloat =
    (A = M.A.parent; LAPACK.ormqr!('L','T',A.factors,A.τ,M.B))
materialize!(M::Lmul{<:AdjQRPackedQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractStridedLayout,<:AbstractMatrix{T},<:AbstractVecOrMat{T}}) where T<:BlasComplex = 
    (A = M.A.parent; LAPACK.ormqr!('L','C',A.factors,A.τ,M.B))
materialize!(M::Lmul{<:AdjQRCompactWYQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractStridedLayout,<:AbstractMatrix{T},<:AbstractVecOrMat{T}}) where T<:BlasFloat =
    (A = M.A.parent; LAPACK.gemqrt!('L','T',A.factors,A.T,M.B))
materialize!(M::Lmul{<:AdjQRCompactWYQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractStridedLayout,<:AbstractMatrix{T},<:AbstractVecOrMat{T}}) where T<:BlasComplex = 
    (A = M.A.parent; LAPACK.gemqrt!('L','C',A.factors,A.T,M.B))
function materialize!(M::Lmul{<:AdjQRPackedQLayout})
    adjA,B = M.A, M.B
    require_one_based_indexing(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    @inbounds begin
        for k = 1:min(mA,nA)
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:mB
                    vBj += conj(Afactors[i,k])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = k+1:mB
                    B[i,j] -= Afactors[i,k]*vBj
                end
            end
        end
    end
    B
end

## AQ
materialize!(M::Rmul{<:AbstractStridedLayout,<:QRPackedQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractVecOrMat{T},<:AbstractMatrix{T}}) where T<:BlasFloat =
    LAPACK.ormqr!('R', 'N', M.B.factors, M.B.τ, M.A)
materialize!(M::Rmul{<:AbstractStridedLayout,<:QRCompactWYQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractVecOrMat{T},<:AbstractMatrix{T}}) where T<:BlasFloat =
    LAPACK.gemqrt!('R','N', M.B.factors, M.B.T, M.A)
function materialize!(M::Rmul{<:Any,<:QRPackedQLayout})
    A,Q = M.A,M.B
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    @inbounds begin
        for k = 1:min(mQ,nQ)
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:mQ
                    vAi += A[i,j]*Qfactors[j,k]
                end
                vAi = vAi*Q.τ[k]
                A[i,k] -= vAi
                for j = k+1:nA
                    A[i,j] -= vAi*conj(Qfactors[j,k])
                end
            end
        end
    end
    A
end

### AQc
materialize!(M::Rmul{<:AbstractStridedLayout,<:AdjQRPackedQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractVecOrMat{T},<:AbstractMatrix{T}}) where T<:BlasReal =
    (B = M.B.parent; LAPACK.ormqr!('R','T',B.factors,B.τ,M.A))
materialize!(M::Rmul{<:AbstractStridedLayout,<:AdjQRPackedQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractVecOrMat{T},<:AbstractMatrix{T}}) where T<:BlasComplex =
    (B = M.B.parent; LAPACK.ormqr!('R','C',B.factors,B.τ,M.A))
materialize!(M::Rmul{<:AbstractStridedLayout,<:AdjQRCompactWYQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractVecOrMat{T},<:AbstractMatrix{T}}) where T<:BlasReal =
    (B = M.B.parent; LAPACK.gemqrt!('R','T',B.factors,B.T,M.A))
materialize!(M::Rmul{<:AbstractStridedLayout,<:AdjQRCompactWYQLayout{<:AbstractStridedLayout,<:AbstractStridedLayout},<:AbstractVecOrMat{T},<:AbstractMatrix{T}}) where T<:BlasComplex =
    (B = M.B.parent; LAPACK.gemqrt!('R','C',B.factors,B.T,M.A))
function materialize!(M::Rmul{<:Any,<:AdjQRPackedQLayout})
    A,adjQ = M.A,M.B
    Q = adjQ.parent
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    @inbounds begin
        for k = min(mQ,nQ):-1:1
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:mQ
                    vAi += A[i,j]*Qfactors[j,k]
                end
                vAi = vAi*conj(Q.τ[k])
                A[i,k] -= vAi
                for j = k+1:nA
                    A[i,j] -= vAi*conj(Qfactors[j,k])
                end
            end
        end
    end
    A
end


__qr(layout, lengths, A; kwds...) = Base.invoke(qr, Tuple{AbstractMatrix{eltype(A)}}, A; kwds...)
_qr(layout, axes, A; kwds...) = __qr(layout, map(length, axes), A; kwds...)
_qr(layout, axes, A, pivot::P; kwds...) where P = Base.invoke(qr, Tuple{AbstractMatrix{eltype(A)},P}, A, pivot; kwds...)
_qr!(layout, axes, A, args...; kwds...) = error("Overload _qr!(::$(typeof(layout)), axes, A)")
_lu(layout, axes, A; kwds...) = Base.invoke(lu, Tuple{AbstractMatrix{eltype(A)}}, A; kwds...)
_lu(layout, axes, A, pivot::P; kwds...) where P = Base.invoke(lu, Tuple{AbstractMatrix{eltype(A)},P}, A, pivot; kwds...)
_lu!(layout, axes, A, args...; kwds...) = error("Overload _lu!(::$(typeof(layout)), axes, A)")
_cholesky(layout, axes, A, ::Val{false}=Val(false); check::Bool = true) = cholesky!(cholcopy(A); check = check)
_cholesky(layout, axes, A, ::Val{true}; tol = 0.0, check::Bool = true) = cholesky!(cholcopy(A), Val(true); tol = tol, check = check)
_cholesky!(layout, axes, A, v::Val{tf}; kwds...) where tf = Base.invoke(cholesky!, Tuple{LinearAlgebra.RealHermSymComplexHerm,Val{tf}}, A, v; kwds...)
_factorize(layout, axes, A) = qr(A) # Default to QR

# Cholesky factorization without pivoting (copied from stdlib/LinearAlgebra).
function _cholesky!(layout, axes, A::LinearAlgebra.RealHermSymComplexHerm, ::Val{false}; check::Bool = true)
    C, info = LinearAlgebra._chol!(A.data, A.uplo == 'U' ? UpperTriangular : LowerTriangular)
    check && LinearAlgebra.checkpositivedefinite(info)
    return Cholesky(C.data, A.uplo, info)
end

_inv_eye(_, ::Type{T}, axs::NTuple{2,Base.OneTo{Int}}) where T = Matrix{T}(I, map(length,axs)...)
function _inv_eye(A, ::Type{T}, (rows,cols)) where T
    dest = zero!(similar(A, T, (cols,rows)))
    dest[diagind(dest)] .= one(T)
    dest
end

function _inv(layout, axes, A)
    T = eltype(A)
    (rows,cols) = axes
    n = checksquare(A)
    S = typeof(zero(T)/one(T))      # dimensionful
    S0 = typeof(zero(T)/oneunit(T)) # dimensionless
    dest = _inv_eye(A, S0, (cols,rows))
    ldiv!(factorize(convert(AbstractMatrix{S}, A)), dest)
end


macro _layoutfactorizations(Typ)
    esc(quote
        LinearAlgebra.cholesky(A::$Typ, args...; kwds...) = ArrayLayouts._cholesky(ArrayLayouts.MemoryLayout(A), axes(A), A, args...; kwds...)
        LinearAlgebra.cholesky!(A::$Typ, v::Val{false}=Val(false); check::Bool = true) = ArrayLayouts._cholesky!(ArrayLayouts.MemoryLayout(A), axes(A), A, v; check=check)
        LinearAlgebra.qr(A::$Typ, args...; kwds...) = ArrayLayouts._qr(ArrayLayouts.MemoryLayout(A), axes(A), A, args...; kwds...)
        LinearAlgebra.qr!(A::$Typ, args...; kwds...) = ArrayLayouts._qr!(ArrayLayouts.MemoryLayout(A), axes(A), A, args...; kwds...)
        LinearAlgebra.lu(A::$Typ, pivot::Union{Val{false}, Val{true}}; kwds...) = ArrayLayouts._lu(ArrayLayouts.MemoryLayout(A), axes(A), A, pivot; kwds...)
        LinearAlgebra.lu(A::$Typ{T}; kwds...) where T = ArrayLayouts._lu(ArrayLayouts.MemoryLayout(A), axes(A), A; kwds...)
        LinearAlgebra.lu!(A::$Typ, args...; kwds...) = ArrayLayouts._lu!(ArrayLayouts.MemoryLayout(A), axes(A), A, args...; kwds...)
        LinearAlgebra.factorize(A::$Typ) = ArrayLayouts._factorize(ArrayLayouts.MemoryLayout(A), axes(A), A)
        LinearAlgebra.inv(A::$Typ) = ArrayLayouts._inv(ArrayLayouts.MemoryLayout(A), axes(A), A)
    end)
end

macro layoutfactorizations(Typ)
    esc(quote
        ArrayLayouts.@_layoutfactorizations $Typ
        ArrayLayouts.@_layoutfactorizations SubArray{<:Any,2,<:$Typ}
        ArrayLayouts.@_layoutfactorizations LinearAlgebra.RealHermSymComplexHerm{<:Real,<:$Typ}
        ArrayLayouts.@_layoutfactorizations LinearAlgebra.RealHermSymComplexHerm{<:Real,<:SubArray{<:Real,2,<:$Typ}}
    end)
end

function ldiv!(C::Cholesky{<:Any,<:AbstractMatrix}, B::LayoutArray)
    if C.uplo == 'L'
        return ldiv!(adjoint(LowerTriangular(C.factors)), ldiv!(LowerTriangular(C.factors), B))
    else
        return ldiv!(UpperTriangular(C.factors), ldiv!(adjoint(UpperTriangular(C.factors)), B))
    end
end


