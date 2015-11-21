# Multivariate Asymmetric Laplace

## Generic multivariate Asymmetric Laplace distribution class

abstract AbstractMvAsymmetricLaplace <: ContinuousMultivariateDistribution

immutable GenericMvALDist{Cov<:AbstractPDMat} <: AbstractMvAsymmetricLaplace
    dim::Int
    zeromean::Bool
    μ::Vector{Float64}
    Σ::Cov

    function GenericMvALDist{Cov}(dim::Int, zmean::Bool, μ::Vector{Float64}, Σ::Cov)
      @compat new(dim, zmean, μ, Σ)
    end
end

function GenericMvALDist{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov, zmean::Bool)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMvALDist{Cov}(d, zmean, μ, Σ)
end

function GenericMvALDist{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMvALDist{Cov}(d, allzeros(μ), μ, Σ)
end

function GenericMvALDist{Cov<:AbstractPDMat}(Σ::Cov)
    d = dim(Σ)
    GenericMvALDist{Cov}(d, true, zeros(d), Σ)    
end

## Construction of multivariate normal with specific covariance type

typealias IsoALDist  GenericMvALDist{ScalMat}
typealias DiagALDist GenericMvALDist{PDiagMat}
typealias MvALDist GenericMvALDist{PDMat}

MvALDist(μ::Vector{Float64}, C::PDMat) = GenericMvALDist(μ, C)
MvALDist(C::PDMat) = GenericMvALDist(C)
MvALDist(μ::Vector{Float64}, Σ::Matrix{Float64}) = GenericMvALDist(μ, PDMat(Σ))
MvALDist(Σ::Matrix{Float64}) = GenericMvALDist(PDMat(Σ))

DiagALDist(μ::Vector{Float64}, C::PDiagMat) = GenericMvALDist(μ, C)
DiagALDist(C::PDiagMat) = GenericMvALDist(C)
DiagALDist(μ::Vector{Float64}, σ::Vector{Float64}) = GenericMvALDist(μ, PDiagMat(abs2(σ)))

IsoALDist(μ::Vector{Float64}, C::ScalMat) = GenericMvALDist(μ, C)
IsoALDist(C::ScalMat) = GenericMvALDist(C)
@compat IsoALDist(μ::Vector{Float64}, σ::Real) = GenericMvALDist(μ, ScalMat(length(μ), abs2(Float64(σ))))
@compat IsoALDist(d::Int, σ::Real) = GenericMvALDist(df, ScalMat(d, abs2(Float64(σ))))

## convenient function to construct distributions of proper type based on arguments

mvaldist(μ::Vector{Float64}, C::AbstractPDMat) = GenericMvALDist(μ, C)
mvaldist(C::AbstractPDMat) = GenericMvALDist(C)

@compat mvaldist(μ::Vector{Float64}, σ::Real) = IsoALDist(μ, Float64(σ))
mvaldist(d::Int, σ::Float64) = IsoALDist(d, σ)
mvaldist(μ::Vector{Float64}, σ::Vector{Float64}) = DiagALDist(μ, σ)
mvaldist(μ::Vector{Float64}, Σ::Matrix{Float64}) = MvALDist(μ, Σ)
mvaldist(Σ::Matrix{Float64}) = MvALDist(df, Σ)

# Basic statistics

length(d::GenericMvALDist) = d.dim

mean(d::GenericMvALDist) = d.μ 
mode(d::GenericMvALDist) = d.μ
modes(d::GenericMvALDist) = [mode(d)]

var(d::GenericMvALDist) = d.df>2 ? (d.df/(d.df-2))*diag(d.Σ) : Float64[NaN for i = 1:d.dim]
scale(d::GenericMvALDist) = full(d.Σ)
cov(d::GenericMvALDist) = d.df>2 ? (d.df/(d.df-2))*full(d.Σ) : NaN*ones(d.dim, d.dim)
invscale(d::GenericMvALDist) = full(inv(d.Σ))
invcov(d::GenericMvALDist) = d.df>2 ? ((d.df-2)/d.df)*full(inv(d.Σ)) : NaN*ones(d.dim, d.dim)
logdet_cov(d::GenericMvALDist) = d.df>2 ? logdet((d.df/(d.df-2))*d.Σ) : NaN

# For calculations see "The Laplace Distributionn and Generalizations", S. Kotz & T. Kozubowski and K. Podgorski
#function entropy(d::GenericMvALDist)
#    hdf, hdim = 0.5*d.df, 0.5*d.dim
#    shdfhdim = hdf+hdim
#    0.5*logdet(d.Σ)+hdim*log(d.df*pi)+lbeta(hdim, hdf)-lgamma(hdim)+shdfhdim*(digamma(shdfhdim)-digamma(hdf))
#end

# evaluation (for GenericMvALDist)

#function sqmahal{T<:Real}(d::GenericMvALDist, x::DenseVector{T}) 
#    z::Vector{Float64} = d.zeromean ? x : x - d.μ
#    invquad(d.Σ, z) 
#end
#
#function sqmahal!{T<:Real}(r::DenseArray, d::GenericMvALDist, x::DenseMatrix{T})
#    z::Matrix{Float64} = d.zeromean ? x : x .- d.μ
#    invquad!(r, d.Σ, z)
#end
#
#sqmahal{T<:Real}(d::AbstractMvAsymmetricLaplace, x::DenseMatrix{T}) = sqmahal!(Array(Float64, size(x, 2)), d, x)


function mvtdist_consts(d::AbstractMvAsymmetricLaplace)
    hdf = 0.5 * d.df
    hdim = 0.5 * d.dim
    shdfhdim = hdf + hdim
    v = lgamma(shdfhdim) - lgamma(hdf) - hdim*log(d.df) - hdim*log(pi) - 0.5*logdet(d.Σ)
    return (shdfhdim, v)
end

function _logpdf{T<:Real}(d::AbstractMvAsymmetricLaplace, x::DenseVector{T})
    shdfhdim, v = mvtdist_consts(d)
    v - shdfhdim * log1p(sqmahal(d, x) / d.df)
end

function _logpdf!{T<:Real}(r::DenseArray, d::AbstractMvAsymmetricLaplace, x::DenseMatrix{T})
    sqmahal!(r, d, x)
    shdfhdim, v = mvtdist_consts(d)
    for i = 1:size(x, 2)
        r[i] = v - shdfhdim * log1p(r[i] / d.df)
    end
    return r
end

_pdf!{T<:Real}(r::DenseArray, d::AbstractMvNormal, x::DenseMatrix{T}) = exp!(_logpdf!(r, d, x))

function gradlogpdf{T<:Real}(d::GenericMvALDist, x::DenseVector{T})
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    prz = invscale(d)*z
    -((d.df + d.dim) / (d.df + dot(z, prz))) * prz
end

# Sampling (for GenericMvALDist)

function _rand!{T<:Real}(d::GenericMvALDist, x::DenseVector{T})
    expdist = Exponential(1)
    mvgaussdist = MvNormal(zeros(length(d.μ)),d.Σ)
    w = rand(expdist)
    sqrtw = sqrt(w)
	y = rand(mvgaussdist)
    meanvec = sqrtw .* d.μ
    dispvec = w .* y
	broadcast!(+, x, meanvec, y)
    x  
end

function _rand!{T<:Real}(d::GenericMvALDist,  x::DenseMatrix{T})
	cols = size(x,2)
	for ii = 1:cols
		x[:,ii] = rand(d)
	end
	x
end	

#function _rand!{T<:Real}(d::GenericMvALDist,  x::DenseMatrix{T})
#    cols = size(x,2)
#    print(cols)
#    expdist = Exponential(1)
#    mvgaussdist = MvNormal(zeros(length(d.μ)),d.Σ)
#    print(mvgaussdist)
#    w = rand(expdist,cols)
#    print(w)
#    print("w")
#    sqrtw = broadcast(sqrt,w)
#	y = rand(mvgaussdist,cols)
#    print(y)
#    print("y")
#    meanvec = sqrtw .* d.μ
#    print(meanvec)
#    print("meanvec")
#    dispvec = w .* y
#    print(dispvec)
#    print("dispvec")
#    broadcast(+,x,meanvec,dispvec) 
#    print(x)
#    print("x")
#    x
#    
#end
#
