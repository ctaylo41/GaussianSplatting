//
//  sort.metal (gpu_sort.metal)
//  GaussianSplatting
//
//  Fixed GPU Radix Sort - Robust implementation with proper synchronization
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant uint RADIX_BITS = 8;
constant uint RADIX_SIZE = 256;  // 2^8
constant uint THREADGROUP_SIZE = 256;

// ============================================================================
// 32-BIT RADIX SORT (for simple depth sorting in viewer)
// ============================================================================

// Convert float to sortable uint (handles negative floats correctly)
inline uint floatToSortable(float f) {
    uint u = as_type<uint>(f);
    // Flip all bits if negative, else flip just sign bit
    uint mask = -int(u >> 31) | 0x80000000;
    return u ^ mask;
}

// Compute depth and create key-value pairs
// NOTE: positions are simd_float3 from C++ which is 16 bytes (float4 aligned)
kernel void computeDepths(
    device const float4* positions [[buffer(0)]],  // simd_float3 is 16-byte aligned like float4
    constant float3& cameraPos [[buffer(1)]],
    device uint* keys [[buffer(2)]],
    device uint* values [[buffer(3)]],
    constant uint& numElements [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;
    
    float3 pos = positions[id].xyz;  // Extract xyz from float4
    
    // Skip invalid positions
    if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z) ||
        isinf(pos.x) || isinf(pos.y) || isinf(pos.z)) {
        keys[id] = 0xFFFFFFFF;  // Put invalid at end
        values[id] = id;
        return;
    }
    
    float3 diff = pos - cameraPos;
    float depth = dot(diff, diff);  // squared distance (avoids sqrt)
    
    // Handle edge cases
    if (isnan(depth) || isinf(depth) || depth < 0.0f) {
        keys[id] = 0xFFFFFFFF;  // Put invalid at end
        values[id] = id;
        return;
    }
    
    // Convert to sortable uint, invert for back-to-front (far first)
    keys[id] = ~floatToSortable(depth);
    values[id] = id;  // Original index
}

// ============================================================================
// HISTOGRAM KERNEL - Thread-safe digit counting
// CRITICAL: Uses per-threadgroup local histogram then atomically merges to global
// ============================================================================

kernel void histogram32(
    device const uint* keys [[buffer(0)]],
    device atomic_uint* globalHistogram [[buffer(1)]],
    constant uint& bitOffset [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // Local histogram for this threadgroup
    threadgroup uint localHist[RADIX_SIZE];
    
    // CRITICAL: Initialize local histogram - ALL threads must participate
    // Each thread initializes one or more entries
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        localHist[i] = 0;
    }
    
    // CRITICAL: Barrier to ensure all initialization is complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Count digits - only threads with valid elements participate
    if (id < numElements) {
        uint key = keys[id];
        uint digit = (key >> bitOffset) & 0xFF;
        
        // Atomic increment of local histogram
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[digit], 1, memory_order_relaxed);
    }
    
    // CRITICAL: Barrier to ensure all counting is complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Add local histogram to global - each thread handles its entries
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        uint count = localHist[i];
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[i], count, memory_order_relaxed);
        }
    }
}

// ============================================================================
// PREFIX SUM KERNEL - Blelloch scan for 256 elements
// ============================================================================

kernel void prefixSum256(
    device uint* histogram [[buffer(0)]],
    device uint* prefixSums [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint temp[RADIX_SIZE * 2];
    
    // Load into shared memory
    temp[tid] = (tid < RADIX_SIZE) ? histogram[tid] : 0;
    temp[tid + RADIX_SIZE] = 0;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = RADIX_SIZE >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear last element for exclusive scan
    if (tid == 0) {
        temp[RADIX_SIZE - 1] = 0;
    }
    
    // Down-sweep phase
    for (uint d = 1; d < RADIX_SIZE; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write results
    if (tid < RADIX_SIZE) {
        prefixSums[tid] = temp[tid];
    }
}

// ============================================================================
// SCATTER32 SIMPLE - O(n²) but guaranteed correct
// Each thread counts how many elements before it have the same digit
// ============================================================================

kernel void scatter32Simple(
    device const uint* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device uint* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* prefixSums [[buffer(4)]],
    constant uint& bitOffset [[buffer(5)]],
    constant uint& numElements [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;
    
    uint key = keysIn[id];
    uint value = valuesIn[id];
    uint digit = (key >> bitOffset) & 0xFF;
    
    // Count elements before me with same digit (O(n) per element = O(n²) total)
    uint localRank = 0;
    for (uint i = 0; i < id; i++) {
        uint otherDigit = (keysIn[i] >> bitOffset) & 0xFF;
        if (otherDigit == digit) {
            localRank++;
        }
    }
    
    // Calculate write position
    uint basePos = prefixSums[digit];
    uint writePos = basePos + localRank;
    
    // Bounds check before writing
    if (writePos < numElements) {
        keysOut[writePos] = key;
        valuesOut[writePos] = value;
    }
}

// ============================================================================
// CLEAR HISTOGRAM
// ============================================================================

kernel void clearHistogram(
    device atomic_uint* histogram [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    if (id < RADIX_SIZE) {
        atomic_store_explicit(&histogram[id], 0, memory_order_relaxed);
    }
}

// ============================================================================
// 64-BIT RADIX SORT (for tile + depth compound keys)
// ============================================================================

kernel void histogram64(
    device const ulong* keys [[buffer(0)]],
    device atomic_uint* globalHistogram [[buffer(1)]],
    constant uint& bitOffset [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint localHist[RADIX_SIZE];
    
    // Initialize local histogram
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        localHist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Count digits
    if (id < numElements) {
        ulong key = keys[id];
        uint digit = (key >> bitOffset) & 0xFF;
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Add to global histogram
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        uint count = localHist[i];
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[i], count, memory_order_relaxed);
        }
    }
}

kernel void scatter64Simple(
    device const ulong* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device ulong* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* prefixSums [[buffer(4)]],
    constant uint& bitOffset [[buffer(5)]],
    constant uint& numElements [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;
    
    ulong key = keysIn[id];
    uint value = valuesIn[id];
    uint digit = (key >> bitOffset) & 0xFF;
    
    uint localRank = 0;
    for (uint i = 0; i < id; i++) {
        uint otherDigit = (keysIn[i] >> bitOffset) & 0xFF;
        if (otherDigit == digit) {
            localRank++;
        }
    }
    
    uint writePos = prefixSums[digit] + localRank;
    if (writePos < numElements) {
        keysOut[writePos] = key;
        valuesOut[writePos] = value;
    }
}

// ============================================================================
// OPTIMIZED SCATTER - Uses threadgroup-local sorting then global merge
// ============================================================================

kernel void scatterOptimized32(
    device const uint* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device uint* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* globalPrefixSums [[buffer(4)]],
    device atomic_uint* digitCounters [[buffer(5)]],
    constant uint& bitOffset [[buffer(6)]],
    constant uint& numElements [[buffer(7)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    // Shared memory for local sorting
    threadgroup uint localKeys[THREADGROUP_SIZE];
    threadgroup uint localValues[THREADGROUP_SIZE];
    threadgroup uint localDigits[THREADGROUP_SIZE];
    threadgroup uint localHist[RADIX_SIZE];
    threadgroup uint localOffsets[RADIX_SIZE];
    threadgroup uint globalBaseOffsets[RADIX_SIZE];
    
    // Initialize histogram
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        localHist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load data and count digits locally
    bool valid = id < numElements;
    uint key = 0, value = 0, digit = 0;
    
    if (valid) {
        key = keysIn[id];
        value = valuesIn[id];
        digit = (key >> bitOffset) & 0xFF;
        
        localKeys[tid] = key;
        localValues[tid] = value;
        localDigits[tid] = digit;
        
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute local prefix sum and atomically get global offset for each digit
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        // Local prefix sum
        uint sum = 0;
        for (uint j = 0; j < i; j++) {
            sum += localHist[j];
        }
        localOffsets[i] = sum;
        
        // Get global offset for this block's digits
        uint count = localHist[i];
        if (count > 0) {
            uint base = atomic_fetch_add_explicit(&digitCounters[i], count, memory_order_relaxed);
            globalBaseOffsets[i] = globalPrefixSums[i] + base;
        } else {
            globalBaseOffsets[i] = globalPrefixSums[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scatter locally within threadgroup
    threadgroup uint localSortedKeys[THREADGROUP_SIZE];
    threadgroup uint localSortedValues[THREADGROUP_SIZE];
    
    if (valid) {
        uint localPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit], 1, memory_order_relaxed);
        if (localPos < THREADGROUP_SIZE) {
            localSortedKeys[localPos] = key;
            localSortedValues[localPos] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory
    if (valid && tid < THREADGROUP_SIZE) {
        uint myKey = localSortedKeys[tid];
        uint myVal = localSortedValues[tid];
        uint sortedDigit = (myKey >> bitOffset) & 0xFF;
        
        // Count how many of this digit came before this position
        uint posInDigit = 0;
        for (uint i = 0; i < tid; i++) {
            if (((localSortedKeys[i] >> bitOffset) & 0xFF) == sortedDigit) {
                posInDigit++;
            }
        }
        
        uint globalPos = globalBaseOffsets[sortedDigit] + posInDigit;
        if (globalPos < numElements) {
            keysOut[globalPos] = myKey;
            valuesOut[globalPos] = myVal;
        }
    }
}

kernel void scatter64Optimized(
    device const ulong* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device ulong* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* globalPrefixSums [[buffer(4)]],
    device atomic_uint* digitCounters [[buffer(5)]],
    constant uint& bitOffset [[buffer(6)]],
    constant uint& numElements [[buffer(7)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup ulong localKeys[THREADGROUP_SIZE];
    threadgroup uint localValues[THREADGROUP_SIZE];
    threadgroup uint localHist[RADIX_SIZE];
    threadgroup uint localOffsets[RADIX_SIZE];
    threadgroup uint globalBaseOffsets[RADIX_SIZE];
    
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        localHist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    bool valid = id < numElements;
    ulong key = 0;
    uint value = 0, digit = 0;
    
    if (valid) {
        key = keysIn[id];
        value = valuesIn[id];
        digit = (key >> bitOffset) & 0xFF;
        
        localKeys[tid] = key;
        localValues[tid] = value;
        
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        uint sum = 0;
        for (uint j = 0; j < i; j++) {
            sum += localHist[j];
        }
        localOffsets[i] = sum;
        
        uint count = localHist[i];
        if (count > 0) {
            uint base = atomic_fetch_add_explicit(&digitCounters[i], count, memory_order_relaxed);
            globalBaseOffsets[i] = globalPrefixSums[i] + base;
        } else {
            globalBaseOffsets[i] = globalPrefixSums[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup ulong localSortedKeys[THREADGROUP_SIZE];
    threadgroup uint localSortedValues[THREADGROUP_SIZE];
    
    if (valid) {
        uint localPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit], 1, memory_order_relaxed);
        if (localPos < THREADGROUP_SIZE) {
            localSortedKeys[localPos] = key;
            localSortedValues[localPos] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (valid && tid < THREADGROUP_SIZE) {
        uint sortedDigit = (localSortedKeys[tid] >> bitOffset) & 0xFF;
        
        uint posInDigit = 0;
        for (uint i = 0; i < tid; i++) {
            if (((localSortedKeys[i] >> bitOffset) & 0xFF) == sortedDigit) {
                posInDigit++;
            }
        }
        
        uint globalPos = globalBaseOffsets[sortedDigit] + posInDigit;
        if (globalPos < numElements) {
            keysOut[globalPos] = localSortedKeys[tid];
            valuesOut[globalPos] = localSortedValues[tid];
        }
    }
}

// ============================================================================
// TILE SORTING KERNELS - Generate keys and build ranges
// ============================================================================

// MUST MATCH tiled_shaders.metal ProjectedGaussian struct EXACTLY
struct ProjectedGaussianForSort {
    float2 screenPos;       // 8 bytes, offset 0
    packed_float3 conic;    // 12 bytes, offset 8 - Inverse 2D covariance
    float depth;            // 4 bytes, offset 20
    float opacity;          // 4 bytes, offset 24 - AFTER sigmoid (for rendering efficiency)
    packed_float3 color;    // 12 bytes, offset 28
    float radius;           // 4 bytes, offset 40
    uint tileMinX;          // 4 bytes, offset 44
    uint tileMinY;          // 4 bytes, offset 48
    uint tileMaxX;          // 4 bytes, offset 52
    uint tileMaxY;          // 4 bytes, offset 56
    float _pad1;            // 4 bytes, offset 60 - explicit padding for float2 alignment
    float2 viewPos_xy;      // 8 bytes, offset 64
    packed_float3 cov2D;    // 12 bytes, offset 72 - (a, b, c) - the 2D covariance BEFORE inversion
    float _pad2;            // 4 bytes, offset 84 - padding to make struct 88 bytes (multiple of 8)
};  // Total: 88 bytes

// Count how many tile-Gaussian pairs each Gaussian generates
kernel void countTilePairs(
    device const ProjectedGaussianForSort* projected [[buffer(0)]],
    device atomic_uint* totalPairs [[buffer(1)]],
    device uint* pairCounts [[buffer(2)]],
    constant uint& numGaussians [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numGaussians) {
        if (id < numGaussians + 1024) pairCounts[id] = 0;  // Clear padding
        return;
    }
    
    ProjectedGaussianForSort g = projected[id];
    
    // Skip invalid Gaussians
    if (g.radius <= 0 || g.tileMinX > g.tileMaxX || g.tileMinY > g.tileMaxY) {
        pairCounts[id] = 0;
        return;
    }
    
    uint numTilesX = g.tileMaxX - g.tileMinX + 1;
    uint numTilesY = g.tileMaxY - g.tileMinY + 1;
    uint count = numTilesX * numTilesY;
    
    pairCounts[id] = count;
    atomic_fetch_add_explicit(totalPairs, count, memory_order_relaxed);
}

// Generate tile-depth keys and Gaussian indices
kernel void generateTileKeys(
    device const ProjectedGaussianForSort* projected [[buffer(0)]],
    device const uint* pairOffsets [[buffer(1)]],
    device ulong* keys [[buffer(2)]],
    device uint* values [[buffer(3)]],
    constant uint& numGaussians [[buffer(4)]],
    constant uint& numTilesX [[buffer(5)]],
    constant uint& maxTilesPerGaussian [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numGaussians) return;
    
    ProjectedGaussianForSort g = projected[id];
    
    if (g.radius <= 0 || g.tileMinX > g.tileMaxX || g.tileMinY > g.tileMaxY) return;
    
    uint offset = pairOffsets[id];
    uint depthKey = as_type<uint>(g.depth);
    
    uint idx = 0;
    for (uint ty = g.tileMinY; ty <= g.tileMaxY; ty++) {
        for (uint tx = g.tileMinX; tx <= g.tileMaxX; tx++) {
            if (idx >= maxTilesPerGaussian) return;
            
            uint tileIdx = ty * numTilesX + tx;
            ulong key = ((ulong)tileIdx << 32) | depthKey;
            
            keys[offset + idx] = key;
            values[offset + idx] = id;
            idx++;
        }
    }
}

// Build tile ranges from sorted keys
kernel void buildTileRanges(
    device const ulong* sortedKeys [[buffer(0)]],
    device uint2* tileRanges [[buffer(1)]],
    constant uint& numPairs [[buffer(2)]],
    constant uint& numTiles [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numTiles) return;
    
    // Binary search for first key with this tile
    uint targetTile = id;
    uint start = 0;
    uint count = 0;
    
    // Find start
    uint lo = 0, hi = numPairs;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        uint keyTile = uint(sortedKeys[mid] >> 32);
        if (keyTile < targetTile) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    start = lo;
    
    // Count entries for this tile
    for (uint i = start; i < numPairs; i++) {
        uint keyTile = uint(sortedKeys[i] >> 32);
        if (keyTile != targetTile) break;
        count++;
    }
    
    tileRanges[id] = uint2(start, count);
}
