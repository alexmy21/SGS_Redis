#!lua name=e_algebra

-- This file is part of efl.lua and released under the MIT license.
-- See the LICENSE file for more information.
-- ======================================================================
-- e_algebra.lua
-- ======================================================================

-- Operations on HyperLogLog sets
-- ======================================================================

local function op(KEYS, ARGV)
    local op = ARGV[1]
    local hllset1_key = KEYS[1]
    local hllset2_key = KEYS[2]

    local result_key = "temp:hllset_" .. op .. "_result"

    redis.call("BITOP", op, result_key, hllset1_key, hllset2_key)

    local result_content = redis.call("GET", result_key)
    local sha256 = redis.sha256hex(result_content)

    redis.call("SET", sha256, result_content)
    redis.call("DEL", result_key)
    
    return sha256
end

local function union(KEYS, ARGV)
    return op(KEYS, {"OR"})
end

local function intersection(KEYS, ARGV)
    return op(KEYS, {"AND"})
end

local function difference(KEYS, ARGV)
    return op(KEYS, {"XOR"})
end

redis.register_function('op',           op)
redis.register_function('union',        union)
redis.register_function('intersection', intersection)
redis.register_function('difference',   difference)


