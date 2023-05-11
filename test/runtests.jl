using SVDSketch
using Test, LinearAlgebra, SparseArrays, StableRNGs

function testSVDMatch(r1, r2, rank=size(r1.S, 1))
    approxEq(a, b) = sum(abs, a .* b) ≈ 1

    @test r1.S[1:rank] ≈ r2.S[1:rank]
    @testset "singular vectors" begin
        for j = 1:rank
            @test approxEq(r1.U[:, j], r2.U[:, j])
            @test approxEq(r1.V[:, j], r2.V[:, j])
        end
    end
end

@testset "dense" begin
    rng = StableRNG(123)

    @testset "real" begin
        A = rand(rng, 1:10, 10, 10) # Integer matrix, tests promotion as well
        U, S, Vt, = svdsketch(A)
        r2 = svd(A)

        testSVDMatch(SVD(U, S, Vt), r2)
        @test_throws ArgumentError svdsketch(A, blocksize=100)
    end

    @testset "maxrank" begin
        A = rand(rng, 1:10, 10, 10) # Integer matrix, tests promotion as well
        U, S, = svdsketch(A, maxrank=3)

        @test maximum(size(S)) <= 3
    end

    @testset "complex" begin
        A = rand(rng, 1:10, 10, 10) + rand(rng, 1:10, 10, 10) * im
        U, S, Vt, = svdsketch(A)
        r2 = svd(A)

        testSVDMatch(SVD(U, S, Vt), r2)
        @test_throws ArgumentError svdsketch(A, blocksize=100)
    end
end

# Following test cases are borrowed from Arpack.jl
@testset "sparse" begin
    @testset "real" begin
        A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], [2.0, -1.0, 6.1, 7.0, 1.5])
        U, S, Vt, = svdsketch(A)
        r2 = svd(Array(A))

        testSVDMatch(SVD(U, S, Vt), r2)
        @test_throws ArgumentError svdsketch(A, blocksize=100)
    end

    @testset "maxrank" begin
        A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], [2.0, -1.0, 6.1, 7.0, 1.5])
        U, S, = svdsketch(A, maxrank=2)

        @test maximum(size(S)) <= 2
    end

    @testset "complex" begin
        A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], exp.(im*[2.0:2:10;]), 5, 4)
        U, S, Vt, = svdsketch(A, blocksize=1)
        r2 = svd(Array(A))

        testSVDMatch(SVD(U, S, Vt), r2)
    end
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

        @testset "blocksize $b" for b in [0, r-1, r+1]
            @testset "tol $t" for t in [eps(Float64)^(1/4), eps(Float64)^(1/3), 0.01]
                U, S, Vt, = b == 0 ? svdsketch(A, t) : svdsketch(A, t, blocksize=b)

                @test size(S, 1) == r
                @test S[1:r] ≈ S
                @test U'*U ≈ Matrix{Float64}(I, r, r)
                @test Vt*Vt' ≈ Matrix{Float64}(I, r, r)
            end
        end
    end
end
