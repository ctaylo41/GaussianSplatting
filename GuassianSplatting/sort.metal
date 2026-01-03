//
//  gpu_sort.metal
//  GaussianSplatting
//
//  Fixed GPU Radix Sort - Deterministic scatter with proper local ranking
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
    float3 diff = pos - cameraPos;
    float depth = dot(diff, diff);  // squared distance (avoids sqrt)
    
    // Convert to sortable uint, invert for back-to-front (far first)
    keys[id] = ~floatToSortable(depth);
    values[id] = id;  // Original index
}

// ============================================================================
// HISTOGRAM KERNEL - Thread-safe digit counting
// ============================================================================

kernel void histogram32(
    device const uint* keys [[buffer(0)]],
    device atomic_uint* globalHistogram [[buffer(1)]],
    constant uint& bitOffset [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    // Local histogram for this threadgroup
    threadgroup atomic_uint localHist[RADIX_SIZE];
    
    // Initialize local histogram
    if (tid < RADIX_SIZE) {
        atomic_store_explicit(&localHist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Count digits
    if (id < numElements) {
        uint key = keys[id];
        uint digit = (key >> bitOffset) & 0xFF;
        atomic_fetch_add_explicit(&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Add local histogram to global
    if (tid < RADIX_SIZE) {
        uint count = atomic_load_explicit(&localHist[tid], memory_order_relaxed);
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[tid], count, memory_order_relaxed);
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
// RANK KERNEL - Compute each element's position within its digit bucket
// This is the KEY FIX for deterministic scatter
// ============================================================================

kernel void computeRanks32(
    device const uint* keys [[buffer(0)]],
    device uint* ranks [[buffer(1)]],
    constant uint& bitOffset [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    // Each threadgroup processes a contiguous block of elements
    // We compute the local rank (position among same-digit elements that come before)
    
    threadgroup uint localCounts[RADIX_SIZE];
    threadgroup uint blockStart;
    
    // Initialize counts
    if (tid < RADIX_SIZE) {
        localCounts[tid] = 0;
    }
    if (tid == 0) {
        blockStart = tgid * tgSize;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (id >= numElements) return;
    
    uint key = keys[id];
    uint myDigit = (key >> bitOffset) & 0xFF;
    
    // Count how many elements BEFORE me in this block have the same digit
    // We do this sequentially within the threadgroup for correctness
    // (This is O(n) per block, which is fine for small blocks)
    
    uint myRank = 0;
    uint start = blockStart;
    uint end = min(start + tgSize, numElements);
    
    for (uint i = start; i < id; i++) {
        uint otherKey = keys[i];
        uint otherDigit = (otherKey >> bitOffset) & 0xFF;
        if (otherDigit == myDigit) {
            myRank++;
        }
    }
    
    ranks[id] = myRank;
}

// ============================================================================
// SCATTER WITH GLOBAL RANKS - Uses prefix sum + local ranks
// ============================================================================

kernel void scatter32WithRanks(
    device const uint* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device uint* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* prefixSums [[buffer(4)]],
    device const uint* globalRanks [[buffer(5)]],
    constant uint& bitOffset [[buffer(6)]],
    constant uint& numElements [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;
    
    uint key = keysIn[id];
    uint value = valuesIn[id];
    uint digit = (key >> bitOffset) & 0xFF;
    
    // Count how many elements before me have the same digit
    uint localRank = 0;
    for (uint i = 0; i < id; i++) {
        uint otherDigit = (keysIn[i] >> bitOffset) & 0xFF;
        if (otherDigit == digit) {
            localRank++;
        }
    }
    
    // Write position = prefix sum for this digit + local rank
    uint writePos = prefixSums[digit] + localRank;
    
    keysOut[writePos] = key;
    valuesOut[writePos] = value;
}

// ============================================================================
// OPTIMIZED SCATTER - Uses threadgroup-local sorting then global merge
// This is MUCH faster than the O(n²) version above
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
    if (tid < RADIX_SIZE) {
        localHist[tid] = 0;
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
    if (tid < RADIX_SIZE) {
        // Local prefix sum
        uint sum = 0;
        for (uint i = 0; i < tid; i++) {
            sum += localHist[i];
        }
        localOffsets[tid] = sum;
        
        // Get global offset for this block's digits
        uint count = localHist[tid];
        if (count > 0) {
            uint base = atomic_fetch_add_explicit(&digitCounters[tid], count, memory_order_relaxed);
            globalBaseOffsets[tid] = globalPrefixSums[tid] + base;
        } else {
            globalBaseOffsets[tid] = globalPrefixSums[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scatter locally within threadgroup
    threadgroup uint localSortedKeys[THREADGROUP_SIZE];
    threadgroup uint localSortedValues[THREADGROUP_SIZE];
    
    if (valid) {
        uint localPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit], 1, memory_order_relaxed);
        localSortedKeys[localPos] = key;
        localSortedValues[localPos] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory
    if (valid) {
        // Find which digit bucket this thread's sorted position belongs to
        uint sortedDigit = (localSortedKeys[tid] >> bitOffset) & 0xFF;
        
        // Count how many of this digit came before this position
        uint posInDigit = 0;
        for (uint i = 0; i < tid; i++) {
            if (((localSortedKeys[i] >> bitOffset) & 0xFF) == sortedDigit) {
                posInDigit++;
            }
        }
        
        uint globalPos = globalBaseOffsets[sortedDigit] + posInDigit;
        keysOut[globalPos] = localSortedKeys[tid];
        valuesOut[globalPos] = localSortedValues[tid];
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
    threadgroup atomic_uint localHist[RADIX_SIZE];
    
    if (tid < RADIX_SIZE) {
        atomic_store_explicit(&localHist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (id < numElements) {
        ulong key = keys[id];
        uint digit = (key >> bitOffset) & 0xFF;
        atomic_fetch_add_explicit(&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < RADIX_SIZE) {
        uint count = atomic_load_explicit(&localHist[tid], memory_order_relaxed);
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[tid], count, memory_order_relaxed);
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
    
    if (tid < RADIX_SIZE) {
        localHist[tid] = 0;
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
    
    if (tid < RADIX_SIZE) {
        uint sum = 0;
        for (uint i = 0; i < tid; i++) {
            sum += localHist[i];
        }
        localOffsets[tid] = sum;
        
        uint count = localHist[tid];
        if (count > 0) {
            uint base = atomic_fetch_add_explicit(&digitCounters[tid], count, memory_order_relaxed);
            globalBaseOffsets[tid] = globalPrefixSums[tid] + base;
        } else {
            globalBaseOffsets[tid] = globalPrefixSums[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup ulong localSortedKeys[THREADGROUP_SIZE];
    threadgroup uint localSortedValues[THREADGROUP_SIZE];
    
    if (valid) {
        uint localPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit], 1, memory_order_relaxed);
        localSortedKeys[localPos] = key;
        localSortedValues[localPos] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (valid) {
        uint sortedDigit = (localSortedKeys[tid] >> bitOffset) & 0xFF;
        
        uint posInDigit = 0;
        for (uint i = 0; i < tid; i++) {
            if (((localSortedKeys[i] >> bitOffset) & 0xFF) == sortedDigit) {
                posInDigit++;
            }
        }
        
        uint globalPos = globalBaseOffsets[sortedDigit] + posInDigit;
        keysOut[globalPos] = localSortedKeys[tid];
        valuesOut[globalPos] = localSortedValues[tid];
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

// Count how many tile-pairs each Gaussian generates
kernel void countTilePairs(
    device const ProjectedGaussianForSort* projected [[buffer(0)]],
    device atomic_uint* tileCounts [[buffer(1)]],
    device atomic_uint* totalPairs [[buffer(2)]],
    constant uint& numGaussians [[buffer(3)]],
    constant uint& numTilesX [[buffer(4)]],
    constant uint& maxTilesPerGaussian [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numGaussians) return;
    
    ProjectedGaussianForSort p = projected[id];
    
    if (p.radius <= 0.0f) return;
    if (p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) return;
    
    uint tilesX = p.tileMaxX - p.tileMinX + 1;
    uint tilesY = p.tileMaxY - p.tileMinY + 1;
    uint numTiles = tilesX * tilesY;
    
    // Skip Gaussians covering too many tiles
    if (numTiles > maxTilesPerGaussian) return;
    
    // Count for each tile
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint tileIdx = ty * numTilesX + tx;
            atomic_fetch_add_explicit(&tileCounts[tileIdx], 1, memory_order_relaxed);
        }
    }
    
    atomic_fetch_add_explicit(totalPairs, numTiles, memory_order_relaxed);
}

// Generate 64-bit keys: (tile_id << 32) | depth_bits
kernel void generateTileKeys(
    device const ProjectedGaussianForSort* projected [[buffer(0)]],
    device const uint* tileOffsets [[buffer(1)]],
    device atomic_uint* tileWriteIdx [[buffer(2)]],
    device ulong* keys [[buffer(3)]],
    device uint* values [[buffer(4)]],
    constant uint& numGaussians [[buffer(5)]],
    constant uint& numTilesX [[buffer(6)]],
    constant uint& maxTilesPerGaussian [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numGaussians) return;
    
    ProjectedGaussianForSort p = projected[id];
    
    if (p.radius <= 0.0f) return;
    if (p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) return;
    
    uint tilesX = p.tileMaxX - p.tileMinX + 1;
    uint tilesY = p.tileMaxY - p.tileMinY + 1;
    if (tilesX * tilesY > maxTilesPerGaussian) return;
    
    // Convert depth to sortable bits (front-to-back: smaller depth = smaller key)
    uint depthBits = as_type<uint>(p.depth);
    depthBits = (depthBits & 0x80000000) ? ~depthBits : (depthBits | 0x80000000);
    
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint tileIdx = ty * numTilesX + tx;
            
            // Get write position for this tile
            uint writeIdx = atomic_fetch_add_explicit(&tileWriteIdx[tileIdx], 1, memory_order_relaxed);
            uint globalIdx = tileOffsets[tileIdx] + writeIdx;
            
            // Create compound key: tile in high bits, depth in low bits
            ulong key = (ulong(tileIdx) << 32) | ulong(depthBits);
            
            keys[globalIdx] = key;
            values[globalIdx] = id;  // Gaussian index
        }
    }
}

// Build tile ranges from sorted keys
kernel void buildTileRanges(
    device const ulong* sortedKeys [[buffer(0)]],
    device uint2* tileRanges [[buffer(1)]],  // .x = start, .y = count
    constant uint& numPairs [[buffer(2)]],
    constant uint& numTiles [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numTiles) return;
    
    // Binary search for tile boundaries
    uint targetTile = id;
    
    // Find first occurrence of this tile
    uint left = 0, right = numPairs;
    while (left < right) {
        uint mid = (left + right) / 2;
        uint keyTile = uint(sortedKeys[mid] >> 32);
        if (keyTile < targetTile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint start = left;
    
    // Find first occurrence of next tile
    right = numPairs;
    while (left < right) {
        uint mid = (left + right) / 2;
        uint keyTile = uint(sortedKeys[mid] >> 32);
        if (keyTile <= targetTile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint end = left;
    
    tileRanges[id] = uint2(start, end - start);
}

// Alternative: linear scan for tile ranges (simpler, works for small tile counts)
kernel void buildTileRangesLinear(
    device const ulong* sortedKeys [[buffer(0)]],
    device uint2* tileRanges [[buffer(1)]],
    constant uint& numPairs [[buffer(2)]],
    constant uint& numTiles [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    // Single threadgroup processes all ranges
    // Good for small number of tiles
    
    threadgroup uint rangeStarts[4096];  // Adjust size as needed
    threadgroup uint rangeCounts[4096];
    
    uint tilesPerThread = (numTiles + 255) / 256;
    uint startTile = tid * tilesPerThread;
    uint endTile = min(startTile + tilesPerThread, numTiles);
    
    // Initialize
    for (uint t = startTile; t < endTile; t++) {
        rangeStarts[t] = 0;
        rangeCounts[t] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scan through sorted keys to find tile boundaries
    uint pairsPerThread = (numPairs + 255) / 256;
    uint startPair = tid * pairsPerThread;
    uint endPair = min(startPair + pairsPerThread, numPairs);
    
    uint prevTile = UINT_MAX;
    for (uint i = startPair; i < endPair; i++) {
        uint tile = uint(sortedKeys[i] >> 32);
        if (tile < numTiles) {
            if (tile != prevTile) {
                // Start of new tile range
                atomic_fetch_min_explicit((threadgroup atomic_uint*)&rangeStarts[tile], i, memory_order_relaxed);
            }
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&rangeCounts[tile], 1, memory_order_relaxed);
            prevTile = tile;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write results
    for (uint t = startTile; t < endTile; t++) {
        tileRanges[t] = uint2(rangeStarts[t], rangeCounts[t]);
    }
}

// ============================================================================
// SIMPLE FALLBACK: O(n²) scatter that's guaranteed correct (for debugging)
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
    
    uint writePos = prefixSums[digit] + localRank;
    keysOut[writePos] = key;
    valuesOut[writePos] = value;
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
    keysOut[writePos] = key;
    valuesOut[writePos] = value;
}

