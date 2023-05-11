push!(LOAD_PATH, "../src/")

using Documenter, SVDSketch

makedocs(
    format = Documenter.HTML(
        canonical = "https://blog.zhupengfei.com.cn/SVDSketch.jl/stable/",
    ),
    sitename="SVDSketch.jl"
)

deploydocs(
    repo = "github.com/zhaowenlan1779/SVDSketch.jl.git",
)
