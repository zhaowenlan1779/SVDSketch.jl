using BenchmarkTools
using SVDSketch
using Images
using Arpack
using LinearAlgebra

function readimage()
    # Read image
    img = Array{Float64}(channelview(load("image1.jpg")))
    p, m, n = size(img)
    A = m > n ? [img[1, :, :] img[2, :, :] img[3, :, :]] : [img[1, :, :]; img[2, :, :]; img[3, :, :]]
    return A
end

tol = 0.1

function checkrank(normA, S)
    r = length(S)
    errqb = sqrt.(1 .- cumsum(S.^2) ./ normA)
    rT = searchsortedfirst(errqb, tol, rev=true)
    println("Initial rank=$r, Truncated rank=$rT")
end

function benchimage(A)
    m, n = size(A)
    b = max(convert(Integer, floor(min(m, n) / 100)), 20)
    normA = sum(abs2, A)

    println("farPCA, P=1")
    display(@benchmark checkrank($normA, svdsketch($A, $tol, blocksize=$b, poweriter=1)[2]))
    # checkrank(normA, svdsketch(A, tol, blocksize=b, poweriter=1))

    println("farPCA, P=5")
    display(@benchmark checkrank($normA, svdsketch($A, $tol, blocksize=$b, poweriter=5)[2]))

    println("svds")
    display(@benchmark checkrank($normA, svds($A, nsv=427)[1].S))
end
