module Util
    using SHA
    using DataFrames
    using EasyConfig
    using JSON3
    using Serialization

    export sha1, ints_to_bits, bits_to_ints, l_hash, print_props, struct_to_df, 
            sha1_union, sha1_intersect, sha1_comp, sha1_xor

    function vector_to_string(vec::Vector{UInt64})::String
        # Convert the vector to a byte array
        byte_array = reinterpret(UInt8, vec)
        # Convert the byte array to a string
        return String(byte_array)
    end

    function string_to_vector(str::String)::Vector{UInt64}
        # Convert the string back to a byte array
        byte_array = Vector{UInt8}(str)
        # Convert the byte array back to a Vector{UInt64}
        return reinterpret(UInt64, byte_array)
    end
    
    function to_blob(data::Vector{UInt64})
        byte_array = ""
        for i in 1:length(data)
            byte_array *= string(Int32(data[i]), base=16, pad=4)
        end        
        
        return byte_array
    end

    function parse_uint8_array(str::AbstractString) :: Vector{UInt8}
        # Split the string on commas
        parts = split(str, ",")
        # Parse each part as UInt8 and collect into a Vector{UInt8}
        uint8_array = [parse(UInt8, part) for part in parts]
        return uint8_array
    end

    function struct_to_dict(s::T) where T
        # Extract field names and values from the struct
        field_names = fieldnames(T)
        field_values = [getfield(s, f) for f in field_names]

        # Create a dictionary from the struct fields
        dict = Dict{Symbol, Any}()
        for (name, value) in zip(field_names, field_values)
            # Check if the value is a struct (non-primitive type) and convert to JSON string if so
            if isstructtype(typeof(value)) || typeof(value) <: AbstractArray
                value = JSON3.write(value)
            end
            dict[Symbol(name)] = value
        end

        return dict
    end

    function struct_to_df(s::T) where T
        # Extract field names and values from the struct
        field_names = fieldnames(T)
        field_values = [getfield(s, f) for f in field_names]

        # Create a DataFrame with a single row from the struct fields
        df = DataFrame()
        for (name, value) in zip(field_names, field_values)
            # Check if the value is a struct (non-primitive type) and convert to JSON string if so
            if isstructtype(typeof(value)) || typeof(value) <: AbstractArray
                value = JSON3.write(value)
            end
            df[!, Symbol(name)] = [value]
        end

        return df
    end

    function dict_to_struct(dict::Dict{AbstractString,AbstractString}, T::Type)
        # Convert keys from AbstractString to Symbol
        symbol_dict = Dict(Symbol(key) => value for (key, value) in dict)
        
        # Extract field names for the struct type T
        field_names = fieldnames(T)
        # Prepare a tuple to hold the field values extracted from the dictionary
        field_values = ()
        
        for name in field_names
            # Extract the value from the dictionary
            value = get(symbol_dict, name, nothing)
            
            # If the value is a JSON string (assuming complex types were stored as JSON), parse it
            # Note: This assumes you have a way to determine which fields need parsing
            # You might need a more sophisticated approach for real-world usage
            if isa(value, String) && occursin(r"^(\{|\[).*(\}|\])$", value) # rudimentary check for JSON strings
                value = JSON3.read(value, Any) # You might need to specify the exact type instead of Any
            end
            
            # Append the value to the tuple of field values
            field_values = (field_values..., value)
        end
        
        # Construct and return the struct instance
        return T(field_values...)
    end

    # SHA1 hash function for Vector{UInt64}
    # This set of function is no more valid.
    # Use sha1_union, sha1_intersect, sha1_comp, sha1_xor
    #--------------------------------------------------
    # function sha1(x::Vector{UInt64})
    #     # Create a SHA1 hash object
    #     h = SHA1()
    #     # Update the hash object with the input
    #     for i in 1:length(x)
    #         update!(h, reinterpret(UInt8, x[i]))
    #     end
    #     # Return the hash
    #     return digest(h)
    # end 

    # function sha1(x::Vector{String})
    #     # Create a SHA1 hash object
    #     h = SHA1()
    #     # Update the hash object with the input
    #     for i in 1:length(x)
    #         update!(h, x[i])
    #     end
    #     # Return the hash
    #     return digest(h)
    # end

    function remove_random_bits(bitvectors::Vector{BitVector}, N::Int)
        # Filter out empty BitVectors
        non_empty_bitvectors = filter(bv -> !isempty(bv), bitvectors)
        
        total_bits = sum(count(b -> b, bv) for bv in non_empty_bitvectors)
        if N > total_bits
            error("N is greater than the total number of true bits in the BitVectors")
        end

        for _ in 1:N
            # Randomly select a non-empty BitVector
            bv_index = rand(1:length(non_empty_bitvectors))
            bv = non_empty_bitvectors[bv_index]

            # Get indices with true values
            true_indices = findall(bv)

            # Randomly select one of the true indices
            if !isempty(true_indices)
                bit_index = rand(true_indices)

                # Set the selected bit to false
                bv[bit_index] = false
            end
        end

        return bitvectors
    end

    # Support for calculating sha1 for union and intersection of strings
    #--------------------------------------------------
    function char_to_bin(c::Char)
        return string(UInt8(c), base=2)
    end

    function string_to_bin(str)
        return join([char_to_bin(c) for c in str])
    end

    function bin_to_string(bin_str)
        return join([Char(parse(UInt8, bin_str[i:min(i+7, end)] , base=2)) for i in 1:8:length(bin_str)])
    end

    function sha1_union(strings::Array{String, 1})
        bin_strings = [string_to_bin(str) for str in strings]
        bin_union = bin_strings[1]
        for i in 2:length(bin_strings)
            bin_union = string(parse(BigInt, "0b" * bin_union) | parse(BigInt, "0b" * bin_strings[i]), base=2)
        end
        str_union = bin_to_string(bin_union)
        new_sha1_hash = bytes2hex(SHA.sha1(str_union))

        return new_sha1_hash
    end

    function sha1_intersect(strings::Array{String, 1})
        bin_strings = [string_to_bin(str) for str in strings]
        bin_intersect = bin_strings[1]
        for i in 2:length(bin_strings)
            bin_intersect = string(parse(BigInt, "0b" * bin_intersect) & parse(BigInt, "0b" * bin_strings[i]), base=2)
        end
        str_intersect = bin_to_string(bin_intersect)
        new_sha1_hash = bytes2hex(SHA.sha1(str_intersect))

        return new_sha1_hash
    end

    function sha1_comp(sha_1::String, sha_2::String)  
        bin_1 = string_to_bin(sha_1)
        bin_2 = string_to_bin(sha_2)
        bin_comp = string(parse(BigInt, "0b" * bin_1) & ~parse(BigInt, "0b" * bin_2), base=2)
        str_comp = bin_to_string(bin_comp)
        new_sha1_hash = bytes2hex(SHA.sha1(str_comp))
        
        return new_sha1_hash
    end

    function sha1_xor(sha_1::String, sha_2::String)  
        bin_1 = string_to_bin(sha_1)
        bin_2 = string_to_bin(sha_2)
        bin_xor = string(xor(parse(BigInt, "0b" * bin_1), parse(BigInt, "0b" * bin_2)), base=2)
        str_xor = bin_to_string(bin_xor)
        new_sha1_hash = bytes2hex(SHA.sha1(str_xor))
        
        return new_sha1_hash
    end

    function sha1_union(strings::Set)
        arr = collect(strings)
        return sha1_union(arr)
    end

    function sha1_intersect(strings::Set)
        arr = collect(strings)
        return sha1_intersect(arr)
    end

    # Graph Utils functions
    #-----------------------------------------------------------------------------#
    
    function print_props(io::IO, o::Union{Config, Dict})
        for (i,(k,v)) in enumerate(pairs(o))
            if i < 5
                print(io, k, '=', repr(v))
                i == length(o) || print(io, ", ")
            end
        end
        length(o) > 5 && print(io, "…")
    end

end # module