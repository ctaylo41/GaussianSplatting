//
//  sort.metal
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#include <metal_stdlib>
using namespace metal;

// convert float to sortable uint and handles negative float correctly
inline uint floatToSortable(float f) {
    uint u = as_type<uint>(f);
    // mask based on sign bit
    uint mask = -int(u >> 31) | 0x80000000;
    return u ^ mask;
}

kernel void computeDepth(constant float3* positions [[buffer(0)]],
                          constant float3& cameraPos [[buffer(1)]],
                          device uint* keys [[buffer(2)]],
                          device uint* values [[buffer(3)]],
                          uint id [[thread_position_in_grid]])
{
    float3 pos = positions[id];
    float3 diff = pos - cameraPos;
    //sqaured distance
    float depth = dot(diff,diff);
    
    //converted to sortable uint and negative for back to front order
    // inverted for descending sort
    keys[id] = ~floatToSortable(depth);
    //original index
    values[id] = id;
}
// count digits in each threadgroup then atomically add to global histogram
kernel void histogram(device const uint* keys [[buffer(0)]],
                      device atomic_uint* globalHistogram [[buffer(1)]],
                      constant uint& bitOffset [[buffer(2)]],
                      constant uint& numElements [[buffer(3)]],
                      uint id [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tgSize [[threads_per_threadgroup]])
{
    // local histogram for threadgroup
    threadgroup atomic_uint localHist[256];
    
    // init local histogram
    if (tid < 256) {
        atomic_store_explicit(&localHist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // count digits in this threadgroup's range
    if (id < numElements) {
        uint key = keys[id];
        uint digit = (key >> bitOffset) & 0xFF;
        atomic_fetch_add_explicit(&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // add local histogram to global
    if (tid < 256) {
        uint count = atomic_load_explicit(&localHist[tid], memory_order_relaxed);
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[tid], count, memory_order_relaxed);
        }
    }
}

// exclusive prefix sum on histogram (Blelloch scan algorithm)
kernel void prefixSum(device uint* histogram [[buffer(0)]],
                      device uint* prefixSums [[buffer(1)]],
                      uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint temp[256];
    
    // Load into shared memory
    temp[tid] = histogram[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Up-sweep (reduce) phase
    for (uint stride = 1; stride < 256; stride *= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 256) {
            temp[idx] += temp[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Clear the last element (for exclusive scan)
    if (tid == 0) {
        temp[255] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Down-sweep phase
    for (uint stride = 128; stride >= 1; stride /= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 256) {
            uint left = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results
    prefixSums[tid] = temp[tid];
}

kernel void scatter(device const uint* keysIn [[buffer(0)]],
                    device const uint* valuesIn [[buffer(1)]],
                    device uint* keysOut [[buffer(2)]],
                    device uint* valuesOut [[buffer(3)]],
                    device atomic_uint* digitOffsets [[buffer(4)]],
                    constant uint& bitOffset [[buffer(5)]],
                    constant uint& numElements [[buffer(6)]],
                    uint id [[thread_position_in_grid]])
{
    if (id>=numElements) return;
    
    uint key = keysIn[id];
    uint value = valuesIn[id];
    uint digit = (key >> bitOffset) & 0xFF;
    
    uint writePos = atomic_fetch_add_explicit(&digitOffsets[digit], 1, memory_order_relaxed);
    
    keysOut[writePos] = key;
    valuesOut[writePos] = value;
}
