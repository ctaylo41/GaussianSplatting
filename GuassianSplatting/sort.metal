//
//  sort.metal
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-31.
//
// GPU Radix Sort Implementation for Gaussian Splatting
//
// FIXED BUGS (Jan 10, 2026):
// 1. scatter32Simple: Now uses separate digitCounters buffer to avoid corrupting
//    prefixSums across multiple passes
// 2. Added computeLocalRanks64 kernel for scatter64Simple to compute position
//    within digit buckets
// 3. Fixed scatterOptimized32 to use localDigits array and proper local position
//    calculation instead of O(n²) loop through localSortedKeys

#include <metal_stdlib>
using namespace metal;

// Constants
constant uint RADIX_BITS = 8;
constant uint RADIX_SIZE = 256;
constant uint THREADGROUP_SIZE = 256;

// 32-BIT Radix Sort for depth sorting in viewer

// Convert float to sortable uint to handle negative floats correctly
inline uint floatToSortable(float f) {
    uint u = as_type<uint>(f);
    // Flip all bits if negative, else flip just sign bit
    uint mask = -int(u >> 31) | 0x80000000;
    return u ^ mask;
}

// Compute depth and create key-value pairs
kernel void computeDepths(
    device const float4* positions [[buffer(0)]],
    constant float3& cameraPos [[buffer(1)]],
    device uint* keys [[buffer(2)]],
    device uint* values [[buffer(3)]],
    constant uint& numElements [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;

    // Extract xyz from float4
    float3 pos = positions[id].xyz;  
    
    // Skip invalid positions
    if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z) ||
        isinf(pos.x) || isinf(pos.y) || isinf(pos.z)) {
        keys[id] = 0xFFFFFFFF;
        values[id] = id;
        return;
    }
    
    // Compute squared distance from camera
    float3 diff = pos - cameraPos;
    // squared distance avoids sqrt for efficiency
    float depth = dot(diff, diff);  
    
    // Handle edge cases
    if (isnan(depth) || isinf(depth) || depth < 0.0f) {
        keys[id] = 0xFFFFFFFF;  // Put invalid at end
        values[id] = id;
        return;
    }
    
    // Convert to sortable uint, invert for back-to-front far first
    keys[id] = ~floatToSortable(depth);
    values[id] = id;
}

// Histogram Kernel Thread-safe digit counting
// Uses per-threadgroup local histogram then atomically merges to global

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
    
    // Initialize local histogram all threads must participate
    // Each thread initializes one or more entries
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        localHist[i] = 0;
    }
    
    // Barrier to ensure all initialization is complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Count digits only threads with valid elements participate
    if (id < numElements) {
        uint key = keys[id];
        uint digit = (key >> bitOffset) & 0xFF;
        
        // Atomic increment of local histogram
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[digit], 1, memory_order_relaxed);
    }
    
    // Barrier to ensure all counting is complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Add local histogram to global each thread handles its entries
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        uint count = localHist[i];
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[i], count, memory_order_relaxed);
        }
    }
}

// Prefix sum Kernel Blelloch scan for 256 elements
kernel void prefixSum256(
    device uint* histogram [[buffer(0)]],
    device uint* prefixSums [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint temp[RADIX_SIZE * 2];
    
    // Load into shared memory
    temp[tid] = (tid < RADIX_SIZE) ? histogram[tid] : 0;
    temp[tid + RADIX_SIZE] = 0;
    
    // Synchronize to make sure all loads are done
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Up-sweep reduce phase
    uint offset = 1;
    for (uint d = RADIX_SIZE >> 1; d > 0; d >>= 1) {
        // Synchronize before each step
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
        // Synchronize before each step
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

// Scatter32 Simple - Fixed to not corrupt prefix sums
// Uses digitCounters for atomic increments, prefixSums for base offsets

kernel void scatter32Simple(
    device const uint* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device uint* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* prefixSums [[buffer(4)]],
    device atomic_uint* digitCounters [[buffer(5)]],
    constant uint& bitOffset [[buffer(6)]],
    constant uint& numElements [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;
    
    // Extract key, value, and digit
    uint key = keysIn[id];
    uint value = valuesIn[id];
    uint digit = (key >> bitOffset) & 0xFF;
    
    // Get base position from prefix sum and increment counter atomically
    // This preserves prefixSums for next pass
    uint offset = atomic_fetch_add_explicit(&digitCounters[digit], 1, memory_order_relaxed);
    uint writePos = prefixSums[digit] + offset;
    
    // Bounds check before writing
    if (writePos < numElements) {
        keysOut[writePos] = key;
        valuesOut[writePos] = value;
    }
}

// Clear Histogram Kernel
kernel void clearHistogram(
    device atomic_uint* histogram [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    if (id < RADIX_SIZE) {
        atomic_store_explicit(&histogram[id], 0, memory_order_relaxed);
    }
}

// REMOVED: computeLocalRanks64 - replaced with atomic scatter
// The O(n²) loop was the performance bottleneck

// 64-BIT RADIX SORT for tile + depth compound keys

kernel void histogram64(
    device const ulong* keys [[buffer(0)]],
    device atomic_uint* globalHistogram [[buffer(1)]],
    constant uint& bitOffset [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    // Simple version each thread directly increments global histogram
    // Avoids any threadgroup-level issues with partial threadgroups
    if (id < numElements) {
        ulong key = keys[id];
        // Extract digit
        uint digit = (key >> bitOffset) & 0xFF;
        atomic_fetch_add_explicit(&globalHistogram[digit], 1, memory_order_relaxed);
    }
}

// Atomic-based Scatter64 - O(1) per element instead of O(n²)
// Uses atomic counters to compute local rank without loops
kernel void scatter64WithAtomicRank(
    device const ulong* keysIn [[buffer(0)]],
    device const uint* valuesIn [[buffer(1)]],
    device ulong* keysOut [[buffer(2)]],
    device uint* valuesOut [[buffer(3)]],
    device const uint* prefixSums [[buffer(4)]],
    device atomic_uint* digitCounters [[buffer(5)]],
    constant uint& bitOffset [[buffer(6)]],
    constant uint& numElements [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numElements) return;
    
    ulong key = keysIn[id];
    uint value = valuesIn[id];
    uint digit = (key >> bitOffset) & 0xFF;
    
    // Atomically get local rank (O(1) per element!)
    // atomic_fetch_add returns the OLD value, which is exactly the count
    // of same-digit elements that came before
    uint localRank = atomic_fetch_add_explicit(&digitCounters[digit], 1, memory_order_relaxed);
    
    // Final position = prefix sum + local rank
    uint writePos = prefixSums[digit] + localRank;
    
    if (writePos < numElements) {
        keysOut[writePos] = key;
        valuesOut[writePos] = value;
    }
}

// Optimized Scatter32 Using Threadgroup Memory and Atomic Counters
// Reduces global atomic contention by using local threadgroup memory
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
    
    // Load keys and values
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
    
    // Place elements in local sorted arrays
    if (valid) {
        uint localPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit], 1, memory_order_relaxed);
        if (localPos < THREADGROUP_SIZE) {
            localSortedKeys[localPos] = key;
            localSortedValues[localPos] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory - use the sorted local arrays
    // The localSortedKeys/Values were already placed in correct order by atomic localOffsets
    if (tid < tgSize && tid < numElements) {
        // Check if this position has valid data
        uint localCount = 0;
        for (uint i = 0; i < RADIX_SIZE; i++) {
            localCount += localHist[i];
        }
        
        if (tid < localCount) {
            keysOut[tgid * THREADGROUP_SIZE + tid] = localSortedKeys[tid];
            valuesOut[tgid * THREADGROUP_SIZE + tid] = localSortedValues[tid];
        }
    }
}

// Optimized Scatter64 Using Threadgroup Memory and Atomic Counters
// Reduces global atomic contention by using local threadgroup memory
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
    
    // Initialize histogram
    for (uint i = tid; i < RADIX_SIZE; i += THREADGROUP_SIZE) {
        localHist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load data and count digits locally
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
    
    // Compute local prefix sum and atomically get global offset for each digit
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
    
    // Scatter locally within threadgroup
    threadgroup ulong localSortedKeys[THREADGROUP_SIZE];
    threadgroup uint localSortedValues[THREADGROUP_SIZE];
    
    // Place elements in local sorted arrays
    if (valid) {
        uint localPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit], 1, memory_order_relaxed);
        if (localPos < THREADGROUP_SIZE) {
            localSortedKeys[localPos] = key;
            localSortedValues[localPos] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory - use sorted local arrays directly
    // The localSortedKeys/Values were already placed in correct order by atomic localOffsets
    if (tid < tgSize && tid < numElements) {
        // Check if this position has valid data
        uint localCount = 0;
        for (uint i = 0; i < RADIX_SIZE; i++) {
            localCount += localHist[i];
        }
        
        if (tid < localCount) {
            keysOut[tgid * THREADGROUP_SIZE + tid] = localSortedKeys[tid];
            valuesOut[tgid * THREADGROUP_SIZE + tid] = localSortedValues[tid];
        }
    }
}

// Tile sorting kernels generate keys and build ranges

// Projected Gaussian struct for sorting
struct ProjectedGaussianForSort {
    float2 screenPos;       
    packed_float3 conic;    
    float depth;           
    float opacity;          
    packed_float3 color;    
    float radius;           
    uint tileMinX;          
    uint tileMinY;          
    uint tileMaxX;          
    uint tileMaxY;          
    float _pad1;            
    float2 viewPos_xy;      
    packed_float3 cov2D;    
    float _pad2;            
};  

// Count how many tile-Gaussian pairs each Gaussian generates
kernel void countTilePairs(
    device const ProjectedGaussianForSort* projected [[buffer(0)]],
    device atomic_uint* totalPairs [[buffer(1)]],
    device uint* pairCounts [[buffer(2)]],
    constant uint& numGaussians [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numGaussians) {
        if (id < numGaussians + 1024) pairCounts[id] = 0;
        return;
    }
    
    ProjectedGaussianForSort g = projected[id];
    
    // Skip invalid Gaussians
    if (g.radius <= 0 || g.tileMinX > g.tileMaxX || g.tileMinY > g.tileMaxY) {
        pairCounts[id] = 0;
        return;
    }
    
    // Compute number of tiles covered
    uint numTilesX = g.tileMaxX - g.tileMinX + 1;
    uint numTilesY = g.tileMaxY - g.tileMinY + 1;
    uint count = numTilesX * numTilesY;
    // Store count
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
    
    // Skip invalid Gaussians
    if (g.radius <= 0 || g.tileMinX > g.tileMaxX || g.tileMinY > g.tileMaxY) return;
    
    // Generate keys for each tile covered
    uint offset = pairOffsets[id];
    uint depthKey = as_type<uint>(g.depth);
    
    // Iterate over tiles
    uint idx = 0;
    for (uint ty = g.tileMinY; ty <= g.tileMaxY; ty++) {
        for (uint tx = g.tileMinX; tx <= g.tileMaxX; tx++) {
            if (idx >= maxTilesPerGaussian) return;
            
            // Compose 64-bit key upper 32 bits tile index, lower 32 bits depth
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
        // Binary search step
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
