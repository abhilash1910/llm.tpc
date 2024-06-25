#include "kernel_config.h"

void main(tensor out, tensor data, size_t count){
          
          
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    VECTOR x; // Assuming VECTOR represents the appropriate SIMD type or intrinsic
    float accumulator = 0.0f;
    
    // Process each index within the specified depth range
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;

        // Guard against out-of-bounds access
        if (idx >= count) {
            continue;
        }

        // Load data and accumulate squares
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            float value;
            v_ld_tnsr_partial_i(value, data, ifmCoords);  // Load value from tensor using v_ld_tnsr_partial_i
            accumulator += value * value;
        }
    }

    // Perform reduction across all threads
    float block_sum = v_reduce_sum(accumulator);  // Assuming blockReduceSum is a hypothetical function

    // Use a single thread to perform the atomic addition to out
    if (get_thread_id() == 0) {  // Assuming get_thread_id returns the thread index
        add(out, block_sum);  // Assuming atomicAdd is a hypothetical function for atomic operations
    }
}