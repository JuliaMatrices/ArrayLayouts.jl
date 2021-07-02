using ArrayLayouts, LinearAlgebra, Test
import ArrayLayouts: sub_materialize
using BandedMatrices: BandedMatrix

struct MyMatrix <: LayoutMatrix{Float64}
    A::Matrix{Float64}
end

Base.getindex(A::MyMatrix, k::Int, j::Int) = A.A[k,j]
Base.setindex!(A::MyMatrix, v, k::Int, j::Int) = setindex!(A.A, v, k, j)
Base.size(A::MyMatrix) = size(A.A)
Base.strides(A::MyMatrix) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyMatrix) where T = Base.unsafe_convert(Ptr{T}, A.A)
MemoryLayout(::Type{MyMatrix}) = DenseColumnMajor()

struct MyVector <: LayoutVector{Float64}
    A::Vector{Float64}
end

Base.getindex(A::MyVector, k::Int) = A.A[k]
Base.setindex!(A::MyVector, v, k::Int) = setindex!(A.A, v, k)
Base.size(A::MyVector) = size(A.A)
Base.strides(A::MyVector) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyVector) where T = Base.unsafe_convert(Ptr{T}, A.A)
MemoryLayout(::Type{MyVector}) = DenseColumnMajor()

# These need to test dispatch reduces to ArrayLayouts.mul, etc.
@testset "LayoutArray" begin
    @testset "LayoutVector" begin
        a = MyVector([1.,2,3])
        B = randn(3,3)
        b = randn(3)

        @test a == a.A == Vector(a)
        @test a[1:3] == a.A[1:3]
        @test a[:] == a
        @test (a')[1,:] == (a')[1,1:3] == a
        @test stringmime("text/plain", a) == "3-element MyVector:\n 1.0\n 2.0\n 3.0"
        @test B*a ≈ B*a.A
        @test B'*a ≈ B'*a.A
        @test transpose(B)*a ≈ transpose(B)*a.A
        @test b'a ≈ transpose(b)a ≈ a'b ≈ transpose(a)b ≈ b'a.A
        @test qr(B).Q*a ≈ qr(B).Q*a.A

        @test a'a == transpose(a)a == dot(a,a) == dot(a,a.A) == dot(a.A,a) == 14
        v = view(a,1:3)
        @test copy(v) == sub_materialize(v) == a[1:3]
        @test dot(v,a) == dot(v,a.A) == dot(a,v) == dot(a.A,v) == dot(v,v) == 14

        V = view(a',:,1:3)
        @test copy(V) == sub_materialize(V) == (a')[:,1:3]

        s = SparseVector(3, [1], [2])
        @test a's == s'a == dot(a,s) == dot(s,a) == dot(s,a.A)
    end

    @testset "LayoutMatrix" begin
        A = MyMatrix(randn(5,5))
        for (kr,jr) in ((1:2,2:3), (:,:), (:,1:2), (2:3,:), ([1,2],3:4), (:,[1,2]), ([2,3],:),
                        (2,:), (:,2), (2,1:3), (1:3,2))
            @test A[kr,jr] == A.A[kr,jr]
        end
        b = randn(5)
        for Tri in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
            @test ldiv!(Tri(A), copy(b)) ≈ ldiv!(Tri(A.A), copy(b))
            @test lmul!(Tri(A), copy(b)) ≈ lmul!(Tri(A.A), copy(b))
        end

        @test copyto!(MyMatrix(Array{Float64}(undef,5,5)), A) == A
        @test copyto!(Array{Float64}(undef,5,5), A) == A
        @test copyto!(MyMatrix(Array{Float64}(undef,5,5)), A.A) == A
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),1:3,1:3), view(A,1:3,1:3)) == A[1:3,1:3]
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),:,:), A) == A
        @test copyto!(MyMatrix(Array{Float64}(undef,3,3)), view(A,1:3,1:3)) == A[1:3,1:3]
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),:,:), A.A) == A
        @test copyto!(Array{Float64}(undef,3,3), view(A,1:3,1:3)) == A[1:3,1:3]
        @test copyto!(MyMatrix(Array{Float64}(undef,5,5)), A') == A'
        @test copyto!(MyMatrix(Array{Float64}(undef,5,5)), view(A',:,:)) == A'
        @test copyto!(Array{Float64}(undef,5,5), A') == A'
        @test copyto!(Array{Float64}(undef,5,5), view(A',:,:)) == A'
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),:,:), A') == A'

        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),:,:), view(A',:,:)) == A'

        @testset "factorizations" begin
            @test qr(A).factors ≈ qr(A.A).factors
            @test qr(A,Val(true)).factors ≈ qr(A.A,Val(true)).factors
            @test lu(A).factors ≈ lu(A.A).factors
            @test lu(A,Val(true)).factors ≈ lu(A.A,Val(true)).factors
            @test_throws ErrorException qr!(A)
            @test_throws ErrorException lu!(A)

            @test qr(A) isa LinearAlgebra.QRCompactWY
            @test inv(A) ≈ inv(A.A)

            S = Symmetric(MyMatrix(reshape(inv.(1:25),5,5) + 10I))
            @test cholesky(S).U ≈ cholesky!(deepcopy(S)).U
            @test cholesky(S,Val(true)).U ≈ cholesky(Matrix(S),Val(true)).U

            B = Symmetric(BandedMatrix(reshape(inv.(1:25),5,5) + 10I))
            @test @inferred(cholesky(B)).U ≈ @inferred(cholesky!(deepcopy(B))).U
        end
        Bin = randn(5,5)
        B = MyMatrix(copy(Bin))
        muladd!(1.0, A, A, 2.0, B)
        @test all(B .=== A.A^2 + 2Bin)

        @testset "tiled_blasmul!" begin
            B = MyMatrix(copy(Bin))
            muladd!(1.0, Ones(5,5), A, 2.0, B)
        end

        @testset "mul!" begin
            @test mul!(B, A, A) ≈ A*A
            @test mul!(B, A', A) ≈ A'*A
            @test mul!(B, A, A') ≈ A*A'
            @test mul!(B, A', A') ≈ A'*A'
            @test mul!(B, A, Bin) ≈ A*Bin

            @test mul!(copy(B), A, A, 2, 3) ≈ 2A*A + 3B
            @test mul!(copy(B), A', A, 2, 3) ≈ 2A'*A + 3B
            @test mul!(copy(B), A, A', 2, 3) ≈ 2A*A' + 3B
            @test mul!(copy(B), A', A', 2, 3) ≈ 2A'*A' + 3B
            @test mul!(copy(B), Bin, A, 2, 3) ≈ 2Bin*A + 3B
            @test mul!(copy(B), Bin, A', 2, 3) ≈ 2Bin*A' + 3B
            @test mul!(copy(B), A, Bin, 2, 3) ≈ 2A*Bin + 3B
            @test mul!(copy(B), A', Bin, 2, 3) ≈ 2A'*Bin + 3B
            @test mul!(copy(B), A, Bin, 2, 3) ≈ 2A*Bin + 3B
            @test mul!(copy(B), A, Bin', 2, 3) ≈ 2A*Bin' + 3B
            @test mul!(copy(B), Bin', A, 2, 3) ≈ 2Bin'*A + 3B
        end

        @testset "generic_blasmul!" begin
            A = BigFloat.(randn(5,5))
            Bin = BigFloat.(randn(5,5))
            B = copy(Bin)
            muladd!(1.0, Ones(5,5), A, 2.0, B)
            @test B == Ones(5,5)*A + 2.0Bin
        end

        C = MyMatrix([1 2; 3 4])
        @test stringmime("text/plain", C) == "2×2 MyMatrix:\n 1.0  2.0\n 3.0  4.0"

        @testset "layoutldiv" begin
            A = MyMatrix(randn(5,5))
            x = randn(5)
            X = randn(5,5)
            t = view(randn(10),[1,3,4,6,7])
            T = view(randn(10,5),[1,3,4,6,7],:)
            t̃ = copy(t)
            T̃ = copy(T)
            B = Bidiagonal(randn(5),randn(4),:U)
            @test ldiv!(A, copy(x)) ≈ A\x
            @test A\t ≈ A\t̃
            # QR is not general enough
            @test_broken ldiv!(A, t) ≈ A\t
            @test ldiv!(A, copy(X)) ≈ A\X
            @test A\T ≈ A\T̃
            @test_broken A/T ≈ A/T̃
            @test_broken ldiv!(A, T) ≈ A\T
            @test B\A ≈ B\Matrix(A)
            @test transpose(B)\A ≈ transpose(B)\Matrix(A) ≈ Transpose(B)\A ≈ Adjoint(B)\A
            @test B'\A ≈ B'\Matrix(A)
            @test A\A ≈ I
            @test_broken A/A ≈ I
            @test A\MyVector(x) ≈ A\x
            @test A\MyMatrix(X) ≈ A\X
        end

        @testset "dot" begin
            A = MyMatrix(randn(5,5))
            b = randn(5)
            @test dot(b, A, b) ≈ b'*(A*b) ≈ b'A*b
        end

        @testset "dual vector * symmetric (#40)" begin
            A = randn(3,3)
            x = rand(3)
            @test x' * Symmetric(MyMatrix(A)) ≈ x'Symmetric(A)
            @test transpose(x) * Symmetric(MyMatrix(A)) ≈ transpose(x)Symmetric(A)
        end

        @testset "map(copy, ::Diagonal)" begin
            # this is needed in BlockArrays
            D = Diagonal([MyMatrix(randn(2,2)), MyMatrix(randn(2,2))])
            @test map(copy, D) == D
        end

        @testset "permutedims(::Diagonal)" begin
            D = Diagonal(MyVector(randn(5)))
            @test permutedims(D) ≡ D
        end
    end

    @testset "l/rmul!" begin
        b = MyVector(randn(5))
        A = MyMatrix(randn(5,5))
        @test lmul!(2, deepcopy(b)) == rmul!(deepcopy(b), 2) == 2b
        @test lmul!(2, deepcopy(A)) == rmul!(deepcopy(A), 2) == 2A
        @test lmul!(2, deepcopy(A)') == rmul!(deepcopy(A)', 2) == 2A'
        @test lmul!(2, transpose(deepcopy(A))) == rmul!(transpose(deepcopy(A)), 2) == 2transpose(A)
        @test lmul!(2, Symmetric(deepcopy(A))) == rmul!(Symmetric(deepcopy(A)), 2) == 2Symmetric(A)
        @test lmul!(2, Hermitian(deepcopy(A))) == rmul!(Hermitian(deepcopy(A)), 2) == 2Hermitian(A)

        C = randn(ComplexF64,5,5)
        @test ArrayLayouts.lmul!(2, Hermitian(copy(C))) == ArrayLayouts.rmul!(Hermitian(copy(C)), 2) == 2Hermitian(C)

        
        @test ldiv!(2, deepcopy(b)) == rdiv!(deepcopy(b), 2) == 2\b
        @test ldiv!(2, deepcopy(A)) == rdiv!(deepcopy(A), 2) == 2\A
        @test ldiv!(2, deepcopy(A)') == rdiv!(deepcopy(A)', 2) == 2\A'
        @test ldiv!(2, transpose(deepcopy(A))) == rdiv!(transpose(deepcopy(A)), 2) == 2\transpose(A)
        @test ldiv!(2, Symmetric(deepcopy(A))) == rdiv!(Symmetric(deepcopy(A)), 2) == 2\Symmetric(A)
        @test ldiv!(2, Hermitian(deepcopy(A))) == rdiv!(Hermitian(deepcopy(A)), 2) == 2\Hermitian(A)
        @test ArrayLayouts.ldiv!(2, Hermitian(copy(C))) == ArrayLayouts.rdiv!(Hermitian(copy(C)), 2) == 2\Hermitian(C)
    end

    @testset "pow/I" begin
        A = randn(2,2)
        B = MyMatrix(A)
        @test B^2 ≈ A^2
        @test B^2.3 ≈ A^2.3
        @test B^(-1) ≈ inv(A)
        @test B + I ≈ I + B ≈ A + I
        @test B - I ≈ A - I
        @test I - B ≈ I - B
    end

    @testset "Diagonal" begin
        D = Diagonal(MyVector(randn(5)))
        D̃ = Diagonal(Vector(D.diag))
        B = randn(5,5)
        B̃ = MyMatrix(B)
        @test D*D ≈ Matrix(D)^2
        @test_broken D^2 ≈ D*D
        @test D*B ≈ Matrix(D)*B
        @test B*D ≈ B*Matrix(D)
        @test D*B̃ ≈ Matrix(D)*B̃
        @test B̃*D ≈ B̃*Matrix(D)
        @test D*D̃ ≈ D̃*D

        @test D\D ≈ I
        @test D\B ≈ Matrix(D)\B
        @test B\D ≈ B\Matrix(D)
        @test D\B̃ ≈ Matrix(D)\B̃
        @test B̃\D ≈ B̃\Matrix(D)
        @test D\D̃ ≈ D̃\D
    end

    @testset "Adj/Trans" begin
        A = MyMatrix(randn(5,5))
        T = UpperTriangular(randn(5,5))
        D = Diagonal(MyVector(randn(5)))

        @test D * A' ≈ D * A.A'
        @test A' * D ≈ A.A' * D

        @test A * Adjoint(T) ≈ A.A * Adjoint(T)
        @test A * Transpose(T) ≈ A.A * Transpose(T)
        @test Adjoint(T) * A ≈ Adjoint(T) * A.A
        @test Transpose(T) * A ≈ Transpose(T) * A.A
        @test Transpose(T)A' ≈ Adjoint(T)A' ≈ Adjoint(T)Transpose(A) ≈ Transpose(T)Transpose(A)
        @test Transpose(A)Adjoint(T) ≈ A'Adjoint(T) ≈ A'Transpose(T) ≈ Transpose(A)Transpose(T)
    end

    @testset "concat" begin
        a = MyVector(randn(5))
        A = MyMatrix(randn(5,6))

        @test [a; a] == [Array(a); a] == [a; Array(a)]
        @test [A; A] == [Array(A); A] == [A; Array(A)]

        @test [Array(a) A A] == [Array(a) Array(A) Array(A)]
        @test [a A A] == [Array(a) Array(A) Array(A)]
        @test [a Array(A) A] == [Array(a) Array(A) Array(A)]
    end
end

struct MyUpperTriangular{T} <: AbstractMatrix{T}
    A::UpperTriangular{T,Matrix{T}}
end

MyUpperTriangular{T}(::UndefInitializer, n::Int, m::Int) where T = MyUpperTriangular{T}(UpperTriangular(Array{T}(undef, n, m)))
MyUpperTriangular(A::AbstractMatrix{T}) where T = MyUpperTriangular{T}(UpperTriangular(Matrix{T}(A)))
Base.convert(::Type{MyUpperTriangular{T}}, A::MyUpperTriangular{T}) where T = A
Base.convert(::Type{MyUpperTriangular{T}}, A::MyUpperTriangular) where T = MyUpperTriangular(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyUpperTriangular}, A::MyUpperTriangular)= A
Base.convert(::Type{AbstractArray{T}}, A::MyUpperTriangular) where T = MyUpperTriangular(convert(AbstractArray{T}, A.A))
Base.convert(::Type{AbstractMatrix{T}}, A::MyUpperTriangular) where T = MyUpperTriangular(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyUpperTriangular{T}}, A::AbstractArray{T}) where T = MyUpperTriangular{T}(A)
Base.convert(::Type{MyUpperTriangular{T}}, A::AbstractArray) where T = MyUpperTriangular{T}(convert(AbstractArray{T}, A))
Base.convert(::Type{MyUpperTriangular}, A::AbstractArray{T}) where T = MyUpperTriangular{T}(A)
Base.getindex(A::MyUpperTriangular, kj...) = A.A[kj...]
Base.getindex(A::MyUpperTriangular, ::Colon, j::AbstractVector) = MyUpperTriangular(A.A[:,j])
Base.setindex!(A::MyUpperTriangular, v, kj...) = setindex!(A.A, v, kj...)
Base.size(A::MyUpperTriangular) = size(A.A)
Base.similar(::Type{MyUpperTriangular{T}}, m::Int, n::Int) where T = MyUpperTriangular{T}(undef, m, n)
Base.similar(::MyUpperTriangular{T}, m::Int, n::Int) where T = MyUpperTriangular{T}(undef, m, n)
Base.similar(::MyUpperTriangular, ::Type{T}, m::Int, n::Int) where T = MyUpperTriangular{T}(undef, m, n)
LinearAlgebra.factorize(A::MyUpperTriangular) = factorize(A.A)

MemoryLayout(::Type{MyUpperTriangular{T}}) where T = MemoryLayout(UpperTriangular{T,Matrix{T}})
triangulardata(A::MyUpperTriangular) = triangulardata(A.A)

@_layoutlmul MyUpperTriangular


@testset "MyUpperTriangular" begin
    A = randn(5,5)
    B = randn(5,5)
    x = randn(5)
    U = MyUpperTriangular(A)

    @test lmul!(U, copy(x)) ≈ U * x
    @test lmul!(U, copy(B)) ≈ U * B

    @test_skip lmul!(U,view(copy(B),collect(1:5),1:5)) ≈ U * B
end
