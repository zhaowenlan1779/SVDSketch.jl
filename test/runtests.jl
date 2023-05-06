using SVDSketch
using Test, LinearAlgebra, SparseArrays, StableRNGs

function testSVDMatch(S1, S2, r=size(S1.S, 1))
    approxEq(a, b) = a ≈ b || a ≈ -b

    @test S1.S[1:r] ≈ S2.S[1:r]
    @testset "singular vectors" begin
        for j = 1:r
            @test approxEq(S1.U[:, j], S2.U[:, j])
            @test approxEq(S1.V[:, j], S2.V[:, j])
        end
    end
end

@testset "dense" begin
    rng = StableRNG(123)
    A = rand(rng, 1:10, 10, 10) # Integer matrix, tests promotion as well
    S1 = svdsketch(A)
    S2 = svd(A)

    testSVDMatch(S1, S2)
    @test_throws ArgumentError svdsketch(A, blocksize=100)
end

@testset "maxrank" begin
    rng = StableRNG(123)
    A = rand(rng, 1:10, 10, 10) # Integer matrix, tests promotion as well
    S1 = svdsketch(A, maxrank=3)

    @test maximum(size(S1.S)) <= 3
end

@testset "sparse" begin
    A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], [2.0, -1.0, 6.1, 7.0, 1.5])
    S1 = svdsketch(A)
    S2 = svd(Array(A))

    testSVDMatch(S1, S2)
    @test_throws ArgumentError svdsketch(A, blocksize=100)
end

@testset "no complex" begin
    # Dense
    A = [1+1im 2+2im; 3+3im 4+4im]
    @test_throws MethodError svdsketch(A)

    # Sparse
    A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], exp.(im*[2.0:2:10;]), 5, 4)
    @test_throws MethodError svdsketch(A)
end

@testset "low rank" begin
    rng = StableRNG(123)
    @testset "rank $r" for r in [2, 5, 10, 100]
        m, n = 3*r, 4*r

        FU = qr(randn(rng, Float64, m, r))
        U = Matrix(FU.Q)
        S = 0.1 .+ sort(rand(rng, r), rev=true)
        FV = qr(randn(rng, Float64, n, r))
        V = Matrix(FV.Q)

        A = U*Diagonal(S)*V'

        @testset "blocksize default" begin
            F = svdsketch(A)

            @test size(F.S, 1) == r
            @test F.S[1:r] ≈ S
            @test F.U'*F.U ≈ Matrix{Float64}(I, r, r)
            @test F.V'*F.V ≈ Matrix{Float64}(I, r, r)
        end

        @testset "blocksize bad" begin
            F = svdsketch(A, blocksize=r-1)

            @test size(F.S, 1) == r
            @test F.S[1:r] ≈ S
            @test F.U'*F.U ≈ Matrix{Float64}(I, r, r)
            @test F.V'*F.V ≈ Matrix{Float64}(I, r, r)
        end
    end
end
