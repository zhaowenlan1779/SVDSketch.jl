module SVDSketch

export svdsketch

using LinearAlgebra: BlasReal, eigen, SVD, svd, tr

# function eigSVD(A)
#     transposed = false
#     if size(A, 1) < size(A, 2)
#         A = A'
#         transposed = true
#     end

#     B = A' * A;
#     D, V = eigen(B)
#     S = sqrt.(D)

#     U = A * (V ./ S')
#     if transposed
#         return V, S, U
#     else
#         return U, S, V
#     end
# end

svdsketch(A::AbstractMatrix{<:BlasReal}, tol=eps(eltype(A))^(1/4); kwargs...) = _svdsketch(A, tol; kwargs...)
svdsketch(A::AbstractMatrix{Complex{T}}, tol=missing; kwargs...) where T = throw(MethodError(svdsketch, Any[A,tol,kwargs...]))

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
    threshold = tol^2 * normA

    Z = similar(A, 0, 0)
    Y = similar(A, m, 0)
    W = similar(A, n, 0)
    WTW = similar(A, 0, 0)

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
            w, ss, = svd(w)
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

        Y = [Y y]
        W = [W w]
        sizeB += blocksize

        normB = tr(Z\WTW)
        if normA - normB < threshold
            break
        end

        # TODO: Test out auto adjustment of block size, like what Matlab does
        if sizeB >= maxrank
            break
        end
        if sizeB + blocksize > maxrank
            blocksize = maxrank - sizeB
        end
    end

    D, V = eigen((Z + Z') / 2)

    # Eliminate negative & very small eigenvalues
    idx = searchsortedfirst(D, tol^2)
    D, V = D[idx:end], V[:, idx:end]

    d = sqrt.(D)
    VS = V ./ d'
    C = VS' * WTW * VS
    C = (C + C') / 2 
    D2, V2 = eigen(C, sortby=λ->-abs(λ))
    d = sqrt.(D2)
    Y = Y * (VS * V2)
    W = W * (VS * (V2 ./ d'))
    return SVD(Y, d, W')
end

end
