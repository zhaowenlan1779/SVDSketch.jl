using BenchmarkTools
using .SVDSketch
using Images
using Arpack

function read_image()
    # Read image
    img = Array(rawview(channelview(load("image1.jpg"))))
    p, m, n = size(img)
    A = m > n ? [img[1, :, :] img[2, :, :] img[3, :, :]] : [img[1, :, :]; img[2, :, :]; img[3, :, :]]
    return A    
end

function bench_image(A)
    tol = 0.1
    m, n = size(A)
    b = max(convert(Integer, floor(min(m, n) / 100)), 20)

    display(@benchmark svdsketch($A, $tol, blocksize=$b, poweriter=1))
    display(@benchmark svdsketch($A, $tol, blocksize=$b, poweriter=5))
    display(@benchmark svds($A, nsv=427))
end
