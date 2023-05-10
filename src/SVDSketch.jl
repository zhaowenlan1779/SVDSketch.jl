module SVDSketch

export svdsketch

using LinearAlgebra: BlasFloat, eigen, SVD, tr
using ElasticArrays

eps(T) = Base.eps(T)
eps(::Type{Complex{T}}) where {T} = eps(T)

function eigSVD(A)
    transposed = false
    if size(A, 1) < size(A, 2)
        A = A'
        transposed = true
    end

    B = A' * A;
    D, V = eigen(B)
    D = real(D)

    # Eliminate negative & very small eigenvalues
    idx = searchsortedfirst(D, eps(eltype(A))^(1/2))
    if idx > length(D)
        error("SVDSketch: Could not find eigenvalue for eigSVD. Your `tol` may be too small.")
    end

    D, V = D[idx:end], V[:, idx:end]
    S = sqrt.(D)

    U = A * (V ./ S')
    if transposed
        return V, S, U
    else
        return U, S, V
    end
end

@doc raw"""
    svdsketch(A[, tol]; [maxrank, blocksize, maxiter, poweriter]) -> (U, S, Vt, apxerror)

Returns the singular value decomposition (SVD) of a low-rank matrix sketch of ``A``.
The matrix sketch only reflects the most important features of ``A`` (up to a tolerance),
which enables faster calculation of the SVD of large matrices compared to using `svds`.

The sketch of ``A`` satisfies that ``\|U \Sigma V^T - A\|_F / \|A\|_F \leq \text{tol}``.
The default value for `tol` is `eps(eltype(A))^(1/4)`.

In addition to the SVD, the vector `apxerror` is returned, whose entries represent the
relative approximation error in each iteration, ``\|U \Sigma V^T - A\|_F / \|A\|_F``.
The length of `apxerror` is equal to the number of iterations, and `apxerror[end]` is
the relative approximation error of the output.

# Options
`maxrank`: The rank of the matrix sketch will not exceed `maxrank`.
The default value is `minimum(size(A))`.

`blocksize`: A larger value reduces the number of needed iterations, but might also
result in the result having higher rank than necessary to achieve convergence.
The default value is `min(max(floor(Integer, 0.1*size(A, 1)), 5), maxrank)`.

`maxiter`: The maximum number of iterations for the algorithm.
The default value is `maxrank ÷ blocksize`.

`poweriter`: The number of power iterations performed within each iteration of the algorithm.
Power iterations improve the orthogonality of the ``U`` and ``V`` outputs.
The default value is 1.
"""
svdsketch(A::AbstractMatrix{<:BlasFloat}, tol=eps(eltype(A))^(1/4); kwargs...) = _svdsketch(A, tol; kwargs...)

function svdsketch(A::AbstractMatrix{T}; kwargs...) where T
    Tnew = typeof(zero(T)/sqrt(one(T)))
    svdsketch(convert(AbstractMatrix{Tnew}, A); kwargs...)
end

function svdsketch(A::AbstractMatrix{T}, tol; kwargs...) where T
    Tnew = typeof(zero(T)/sqrt(one(T)))
    svdsketch(convert(AbstractMatrix{Tnew}, A), tol; kwargs...)
end

function _svdsketch(A, tol;
                    maxrank::Integer = minimum(size(A)),
                    blocksize::Integer = min(max(floor(Integer, 0.1*size(A, 1)), 5), maxrank),
                    maxiter::Integer = maxrank ÷ blocksize,
                    poweriter::Integer = 1)

    if blocksize > maxrank
        throw(ArgumentError("Block size cannot be larger than max rank"))
    end
    maxrank = maxrank ÷ blocksize * blocksize

    m, n = size(A)
    normA = sum(abs2, A)

    Z = Matrix{eltype(A)}(undef, 0, 0)
    
    Y = ElasticMatrix{eltype(A)}(undef, m, 0)
    sizehint!(Y, m, 10 * blocksize)
    
    W = ElasticMatrix{eltype(A)}(undef, n, 0)
    sizehint!(W, n, 10 * blocksize)

    WTW = Matrix{eltype(A)}(undef, 0, 0)

    apxerror = Vector{Float64}(undef, maxiter)
    # oldblocksize = blocksize
    sizeB = 0
    for i = 1:maxiter
        w = randn(n, blocksize)

        alpha = 0
        for j = 1:poweriter
            if i > 1
                w = A' * (A * w) - W * (Z \ (W' * w)) - alpha * w
            else
                w = A' * (A * w) - alpha * w
            end
            w, ss, = eigSVD(w)
            if j > 1 && ss[1] > alpha
                alpha = (alpha + ss[1]) / 2
            end
        end

        y = A * w
        w = A' * y
        if i > 1
            ytYtemp = y' * Y
            Z = [Z ytYtemp'; ytYtemp y'*y]
            wtWtemp = w' * W
            WTW = [WTW wtWtemp'; wtWtemp w'*w]
        else
            Z = y' * y
            WTW = w' * w
        end

        append!(Y, y)
        append!(W, w)
        sizeB += blocksize

        apxerror[i] = sqrt(max(1 - real(tr(Z\WTW)) / normA, 0))
        if apxerror[i] < tol || sizeB >= maxrank
            apxerror = apxerror[1:i]
            break
        end

        # Adjust block size
        # if i > 1 && apxerror[i] > apxerror[i - 1] / 2
        #     blocksize += oldblocksize
        # end

        if sizeB + blocksize > maxrank
            blocksize = maxrank - sizeB
        end
    end

    D, V = eigen((Z + Z') / 2)
    d = sqrt.(D)
    VS = V ./ d'
    C = VS' * WTW * VS
    C = (C + C') / 2 
    D2, V2 = eigen(C, sortby=λ->-abs(λ))
    d = sqrt.(D2)
    VS *= V2
    Y = Y * VS
    W = W * VS ./ d'
    return Y, d, W', apxerror
end

end
