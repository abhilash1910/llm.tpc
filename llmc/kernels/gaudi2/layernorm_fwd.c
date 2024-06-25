#include "kernel_config.h"

void main(tensor out, tensor mean, tensor rstd, tensor inp,
          tensor weight, tensor bias, int N, int C){
          
          
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    VECTOR x; // Assuming VECTOR represents the appropriate SIMD type or intrinsic

    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;

        // Guard against out-of-bounds access
        if (idx >= N) {
            continue;
        }

        // Seek to the input position inp[idx,:]
         for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            float sum_sq_diff = 0.0f;

            // Calculate mean and variance
            for (int i = 0; i < VECTOR_SIZE; ++i) {
                // Load input data from inp tensor
                float x = v_ld_tnsr_partial_i(inp, ifmCoords, i);

                sum += x;
                sum_sq_diff += (x * x);
            }
            float m = sum / VECTOR_SIZE;
            float v = (sum_sq_diff / VECTOR_SIZE) - (m * m);
            float s = 1.0f / sqrt(v + 1e-5f);
             // Loop over vector elements again for normalization and scaling
            for (int i = 0; i < VECTOR_SIZE; ++i) {
                // Load input data from inp tensor
                float x = v_ld_tnsr_partial_i(inp, ifmCoords, c * VECTOR_SIZE + i);

                // Normalize and scale using weight and bias tensors
                float n = s * (x - m);
                float o = n * v_ld_tnsr_partial_i(weight, ifmCoords, c * VECTOR_SIZE + i) +
                          v_ld_tnsr_partial_i(bias, ifmCoords, c * VECTOR_SIZE + i);

                // Store results to out tensor
                st_tnsr_i_v(out, ifmCoords, c * VECTOR_SIZE + i, o);
            }

            // Cache the mean and rstd for the backward pass later
            st_tnsr_i_v(mean, ifmCoords, c / VECTOR_SIZE, m);
            st_tnsr_i_v(rstd, ifmCoords, c / VECTOR_SIZE, s);
        }
     }    
          
 }