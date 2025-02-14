# 
# This module is greatly inspired by the implementation of the HLL algorithm in the Julia library:
# 
# Below is the header from this file
# From Flajolet, Philippe; Fusy, Éric; Gandouet, Olivier; Meunier, Frédéric (2007)
# DOI: 10.1.1.76.4286
# With algorithm improvements by Google (https://ai.google/research/pubs/pub40671)

# Principle:
# When observing N distinct uniformly distributed integers, the expected maximal
# number of leading zeros in the integers is log(2, N), with large variation.
# To cut variation, we keep 2^P counters, each keeping track of N/2^P
# observations. The estimated N for each counter is averaged using harmonic mean.
# Last, corrections for systematic bias are added, one multiplicative and one
# additive factor.
# To make the observations uniformly distributed integers, we hash them.

# We made sugnificant changes to the original implementation:
# - We use a BitVector instead of a UInt8 for the counters
# - We implemented additional operators to support set operations, like union (union), intersection(intersect), difference(diff), 
#   and equality (isequal). Now they work the same way as they work for sets
# - we added a function to convert the BitVector to a UInt64 (dump) and vice versa (restore)
# - We added a function to calculate the delta between two HLL sets (delta)
# - We added a function to calculate SHA1 of the counts as a string
# - We also renamed some of the original operators to be more consistent with HyperLogLog terminology
"""
MIT License

Copyright (c) 2023: Jakob Nybo Nissen.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

https://github.com/jakobnissen/Probably.jl/blob/master/src/hyperloglog/hyperloglog.jl

I borrowed a lot from this project, but also made a lot of changes, 
so, for all errors do not blame the original author but me.
"""

module HllSets

    include("constants.jl")
    using SHA, DataFrames, CSV, Arrow, Tables, LinearAlgebra, JSON3, SparseArrays

    export HllSet, add!, count, union, intersect, diff, isequal, isempty, id, delta, getbin, getzeros, maxidx, match, cosine, dump, restore

    struct HllSet{P}
        counts::Vector{UInt32}
        function HllSet{P}() where {P}
            isa(P, Integer) || throw(ArgumentError("P must be integer"))
            (P < 4 || P > 18) && throw(ArgumentError("P must be between 4 and 18"))
            new(fill(UInt32(0), 2^P))
        end
    end

    function HllSet(p::Int=10)
        return HllSet{p}()
    end

    # Overload the show function to print the HllSet
    #--------------------------------------------------
    Base.show(io::IO, x::HllSet{P}) where {P} = println(io, "HllSet{$(P)}()")

    Base.sizeof(::Type{HllSet{P}}) where {P} = 1 << P
    Base.sizeof(x::HllSet{P}) where {P} = sizeof(typeof(x))

    # ====================================================
    # Functions to calculate bin number and number of trailing zeros
    #--------------------------------------------------
    """
        The `getbin` function is part of the `HllSet` struct and it's used to get the binary representation 
        of an integer `x` based on the parameter `P` of the `HllSet`.

        Here's a step-by-step explanation of how it works:

        1. The function takes two arguments: an instance of `HllSet` and an integer `x`.

        2. It calculates the number of bits to shift right (`>>>`) by subtracting `P + 1` 
        from `8 * sizeof(UInt)`. This is done to get the `P + 1` most significant bits of `x`.

        3. The `x` is then shifted right by the calculated number of bits plus 1.

        4. The shifted `x` is converted to a hexadecimal string and the "0x" prefix 
        (which indicates a hexadecimal number in Julia) is removed.

        5. Finally, the hexadecimal string is parsed back into an integer and returned.

        This function essentially extracts the `P + 1` most significant bits from `x` and returns them as an integer. 
        The `+ 1` is there to compensate for the fact that `BitVector` in Julia is of size 64.
    """
    function getbin(hll::HllSet{P}, x::Int) where {P}
        return getbin(x, P=P)        
    end

    function getbin(x::Int; P::Int=10) 
        # Increasing P by 1 to compensate BitVector size that is of size 64
        x = x >>> (8 * sizeof(UInt) - (P + 1)) + 1
        str = replace(string(x, base = 16), "0x" => "")
        return parse(Int, str, base = 16)
    end

    function getzeros(hll::HllSet{P}, x::Int) where {P}
        return getzeros(x, P=P)
    end

    function getzeros(x::Int; P::Int=10)
        or_mask = ((UInt(1) << P) - 1) << (8 * sizeof(UInt) - P)
        return trailing_zeros(x | or_mask) + 1
    end

    # Function to add an element to the HllSet
    #--------------------------------------------------
    function add!(hll::HllSet{P}, x::Any; seed::Int = 0) where {P}
        # println("seed = ", seed, "; P = ", P, "; x = ", x)
        h = u_hash(x; seed=seed)
        # println("hash = ", h)
        bin = getbin(hll, h)
        idx = getzeros(hll, h)
        if idx <= 32
            hll.counts[bin] |= (1 << (idx - 1))
        end
    end

    function add!(hll::HllSet{P}, values::Union{Set, Vector}; seed::Int = 0) where {P}
        for value in values
            add!(hll, value, seed=seed)
        end
    end

    # Overload the Set operations for the HllSet
    #--------------------------------------------------
    function Base.union!(dest::HllSet{P}, src::HllSet{P}) where {P}
        length(dest.counts) == length(src.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        @inbounds for i in 1:length(dest.counts)
            dest.counts[i] = dest.counts[i] .| src.counts[i]
        end
        return dest
    end

    function Base.copy!(dest::HllSet{P}, src::HllSet{P}) where {P}
        length(dest.counts) == length(src.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        @inbounds for i in 1:length(dest.counts)
            dest.counts[i] = src.counts[i]
        end
        return dest
    end

    function Base.copy!(src::HllSet{P}) where {P}
        # length(dest.counts) == length(src.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        dest = HllSet{P}()
        @inbounds for i in 1:length(src.counts)
            dest.counts[i] = src.counts[i]
        end
        return dest
    end

    function Base.union(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = x.counts[i] .| y.counts[i]
        end
        return z
    end

    function Base.intersect(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = x.counts[i] .& y.counts[i]
        end
        return z
    end

    function set_xor(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = xor.(x.counts[i], (y.counts[i]))
        end
        return z
    end

    function set_comp(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        z = HllSet{P}()
        @inbounds for i in 1:length(x.counts)
            z.counts[i] = .~y.counts[i] .& x.counts[i]
        end
        return z
    end

    """
        This convenience methods that semantically reflect the purpose of using set_comp function
        in case of comparing two states of the same set now (current) and as it was before (previous).
        - set_added - returns the elements that are in the current set but not in the previous
        - set_deleted - returns the elements that are in the previous set but not in the current
    """
    function set_added(current::HllSet{P}, previous::HllSet{P}) where {P} 
        length(previous.counts) == length(current.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        added = HllSet{P}()
        @inbounds for i in 1:length(previous.counts)
            added.counts[i] = .~previous.counts[i] .& current.counts[i]
        end
        return added
    end

    function set_deleted(current::HllSet{P}, previous::HllSet{P}) where {P} 
        length(previous.counts) == length(current.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        added = HllSet{P}()
        @inbounds for i in 1:length(previous.counts)
            added.counts[i] = .~current.counts[i] .& previous.counts[i]
        end
        return added
    end

    """
        The `Base.diff` function in Julia is part of the `HllSet` struct and it's used to calculate the difference between two `HllSet` instances.

        Here's a step-by-step explanation of how it works:

        1. The function takes two arguments: two instances of `HllSet`.

        2. It checks if the lengths of `x.counts` and `y.counts` are equal. If they are not, it throws an `ArgumentError`.

        3. It initializes three new `HllSet` instances: `n`, `d`, and `r`.

        4. It calculates the set complement of `hll_1` and `hll_2` (i.e., elements that are in `hll_1` but not in `hll_2`) 
        and assigns it to `n`.

        5. It calculates the set complement of `hll_2` and `hll_1` (i.e., elements that are in `hll_2` but not in `hll_1`) 
        and assigns it to `d`.

        6. It calculates the intersection of `hll_1` and `hll_2` (i.e., elements that are in both `hll_1` and `hll_2`) 
        and assigns it to `r`.

        7. It returns a tuple with three fields: `DEL` (for deleted elements), `RET` (for retained elements), 
        and `NEW` (for new elements). Each field is an `HllSet` representing the corresponding set of elements.

        This function essentially calculates the difference between two `HllSet`s in terms of deleted, retained, and new elements.

    """
    function Base.diff(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
        length(hll_1.counts) == length(hll_2.counts) || throw(ArgumentError("HllSet{P} must have same size"))

        n = HllSet{P}()
        d = HllSet{P}()
        r = HllSet{P}()

        d = set_comp(hll_1, hll_2)
        n = set_comp(hll_2, hll_1)
        r = intersect(hll_1, hll_2)

        return (DEL = d, RET = r, NEW = n)
    end

    function Base.isequal(x::HllSet{P}, y::HllSet{P}) where {P} 
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        @inbounds for i in 1:length(x.counts)
            x.counts[i] == y.counts[i] || return false
        end
        return true
    end

    Base.isempty(x::HllSet{P}) where {P} = all(all, x.counts)   

    # Functions that support calculations of HllSets cardinality
    #--------------------------------------------------
    α(x::HllSet{P}) where {P} =
        if P == 4
            return 0.673
        elseif P == 5
            return 0.697
        elseif P == 6
            return 0.709
        else
            return 0.7213 / (1 + 1.079 / sizeof(x))
        end 
    
    function bias(::HllSet{P}, biased_estimate) where {P}
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have bias estimates for P ∈ 4:18")
        end
        rawarray = @inbounds RAW_ARRAYS[P - 3]
        biasarray = @inbounds BIAS_ARRAYS[P - 3]
        firstindex = searchsortedfirst(rawarray, biased_estimate)
        # Raw count large, no need for bias correction
        if firstindex == length(rawarray) + 1
            return 0.0
            # Raw count too small, cannot be corrected. Maybe raise error?
        elseif firstindex == 1
            return @inbounds biasarray[1]
            # Else linearly approximate the right value for bias
        else
            x1, x2 = @inbounds rawarray[firstindex - 1], @inbounds rawarray[firstindex]
            y1, y2 = @inbounds biasarray[firstindex - 1], @inbounds biasarray[firstindex]
            delta = @fastmath (biased_estimate - x1) / (x2 - x1) # relative distance of raw from x1
            return y1 + delta * (y2 - y1)
        end
    end

    function maxidx(x::UInt32)        
        total_bits = sizeof(x) * 8
        leading_zeros_count = leading_zeros(x)
        return total_bits - leading_zeros_count
    end

    # Function to calculate the cardinality of the HllSet
    #--------------------------------------------------
    function Base.count(x::HllSet{P}) where {P}
        # Harmonic mean estimates cardinality per bin. There are 2^P bins
        harmonic_mean = sizeof(x) / sum(1 / 1 << maxidx(i) for i in x.counts)
        biased_estimate = α(x) * sizeof(x) * harmonic_mean
        return round(Int, biased_estimate - bias(x, biased_estimate))
    end

    """
        The `id` function in Julia is part of the `HllSet` struct and it's used to generate 
            a unique identifier for an instance of `HllSet`.

        Here's a step-by-step explanation of how it works:

        1. The function takes one argument: an instance of `HllSet`.

        2. It initializes an empty byte array `bytearray`.

        3. It then iterates over each bit vector in `x.counts`. For each bit vector, it reinterprets 
            the bit vector as an array of `UInt8` (unsigned 8-bit integers) and appends it to `bytearray`. 
            This effectively converts the vector of bit vectors into a byte array.

        4. It calculates the SHA1 hash of the byte array. SHA1 is a cryptographic hash function that 
            produces a 160-bit (20-byte) hash value. It's commonly used to check the integrity of data.

        5. It converts the hash value into a hexadecimal string using the `SHA.bytes2hex` function and returns it.

        This function essentially generates a unique identifier for an `HllSet` based on the contents 
            of its `counts` field. The identifier is a SHA1 hash, so 
            it's highly unlikely that two different `HllSet`s will have the same identifier.
    """
    function id(x::HllSet{P}) where {P}
        if x == nothing
            return nothing
        end
        # Convert the Vector{UInt32} to a byte array
        bytearray = reinterpret(UInt8, x.counts)

        # Calculate the SHA1 hash
        hash_value = SHA.sha1(bytearray)
        return SHA.bytes2hex(hash_value)
    end

    function sha1(x::HllSet{P}) where {P}
        # Convert the Vector{BitVector} to a byte array
        bytearray = UInt8[]
        for bv in x.counts
            append!(bytearray, reinterpret(UInt8, bv))
        end
        # Calculate the SHA1 hash
        hash_value = SHA.sha1(bytearray)
        return SHA.bytes2hex(hash_value)
    end

    # Function to calculate the Jaccard similarity between two HllSets
    #--------------------------------------------------
    """
        The `Base.match` function in Julia is part of the `HllSet` struct and it's used to calculate 
            the percentage of matching elements between two `HllSet` instances.

        Here's a step-by-step explanation of how it works:

        1. The function takes two arguments: two instances of `HllSet`.

        2. It checks if the lengths of `x.counts` and `y.counts` are equal. If they are not, it throws an `ArgumentError`.

        3. It calculates the union of `x` and `y` and counts the number of elements in the result. This is stored in `count_u`.

        4. It calculates the intersection of `x` and `y` and counts the number of elements in the result. This is stored in `count_i`.

        5. It calculates the ratio of `count_i` to `count_u`, multiplies it by 100 to get a percentage, 
        rounds it to the nearest integer, and returns it.

        This function essentially calculates the Jaccard index (the size of the intersection 
            divided by the size of the union) of the two `HllSet`s and returns it as a percentage. 
            The Jaccard index is a measure of the similarity between two sets. It's 100% if the sets 
            are identical and 0% if they have no elements in common.
    """
    function Base.match(x::HllSet{P}, y::HllSet{P}) where {P}
        length(x.counts) == length(y.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        
        count_u = count(union(x, y))
        count_i = count(intersect(x, y))
        return round(Int64, ((count_i / count_u) * 100))
    end

    """
        The `cosine` function in Julia is part of the `HllSet` struct and it's used to calculate 
            the cosine similarity between two `HllSet` instances.

        Here's a step-by-step explanation of how it works:

        1. The function takes two arguments: two instances of `HllSet`.

        2. It checks if the lengths of `x.counts` and `y.counts` are equal. If they are not, it throws an `ArgumentError`.

        3. It assigns the `counts` field of `hll_1` and `hll_2` to `v1` and `v2` respectively.

        4. It calculates the dot product of `v1` and `v2` and divides it by the product of the norms of `v1` and `v2`.

        5. It returns the result of the division.

        This function essentially calculates the cosine similarity between the `counts` fields of two `HllSet`s. 
            The cosine similarity is a measure of the cosine of the angle between two vectors. 
            It's a measure of how similar the vectors are. If the vectors are identical, the cosine similarity is 1. 
            If the vectors are orthogonal (i.e., they have an angle of 90 degrees between them), the cosine similarity is 0.
    """

    function cosine(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
        length(hll_1.counts) == length(hll_2.counts) || throw(ArgumentError("HllSet{P} must have same size"))

        v1 = hll_1.counts
        v2 = hll_2.counts
        if norm(v1) == 0 || norm(v2) == 0
            return 0.0
        end
        return dot(v1, v2) / (norm(v1) * norm(v2))
    end

    # dump  function: 
    #   Convert the reversed BitVector to a UInt64
    #   we reversed the BitVector to make integer smaller
    #--------------------------------------------------
    function Base.dump(x::HllSet{P}) where {P}
        # Base.depwarn("dump(hll::HllSet{P}) is deprecated, use getcounts(x::Int; P::Int=10) instead.", :getbin)
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have dump for P ∈ 4:18")
        end
        
        return x.counts
    end

    # restore function
    #   Assumes that integers in vector are generated from reversed bit-vectors 
    #--------------------------------------------------
    function restore!(z::HllSet{P}, x::Vector{UInt32}) where {P} 
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have restore for P ∈ 4:18")
        end
        if length(x) != length(z.counts)
            error("The length of the vector must be equal to the length of the HllSet")
        end        
        # z.counts = x
        @inbounds for i in 1:length(x)
            z.counts[i] = x[i]
        end
        return z
    end

    function restore!(z::HllSet{P}, x::String) where {P}
        # For safety - this is also enforced in the HLL constructor
        if P < 4 || P > 18
            error("We only have restore for P ∈ 4:18")
        end
        dataset = JSON3.read(x, Vector{UInt32})
        
        @inbounds for i in 1:length(x)
            z.counts[i] = x[i]
        end
        return z
    end     

    function u_hash(x; seed::Int=0) 
        if seed == 0
            abs_hash = abs(hash(x))
        else
            abs_hash = abs(hash(hash(x) + seed))
        end         
        return Int(abs_hash % typemax(Int64))
    end
end