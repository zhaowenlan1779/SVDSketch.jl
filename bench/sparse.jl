using MKLSparse
using BenchmarkTools
using .SVDSketch
using Arpack
using LinearAlgebra
using SparseArrays
using Scanf

function readmatrix()
    cnt = 948464
    I = Vector{Int64}(undef, cnt)
    J = Vector{Int64}(undef, cnt)
    V = Vector{Float64}(undef, cnt)
    open("bench/SNAP.dat", "r") do io
        for i=1:cnt
            r, I[i], J[i], V[i] = @scanf(io, "%d %d %lf", Int64, Int64, Float64)
        end
    end
    return sparse(I, J, V)
end

tol = 0.5

function checkrank(normA, S)
    r = length(S)
    errqb = sqrt.(1 .- cumsum(S.^2) ./ normA)
    rT = searchsortedfirst(errqb, tol, rev=true)
    println("Initial rank=$r, Truncated rank=$rT")
end

function benchsparse(A)
    m, n = size(A)
    b = max(convert(Integer, floor(min(m, n) / 100)), 20)
    normA = sum(abs2, A)

    println("farPCA, P=1")
    display(@benchmark checkrank($normA, svdsketch($A, $tol, blocksize=$b, poweriter=1)[2]))

    println("farPCA, P=5")
    display(@benchmark checkrank($normA, svdsketch($A, $tol, blocksize=$b, poweriter=5)[2]))
end
