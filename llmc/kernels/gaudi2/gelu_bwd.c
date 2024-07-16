#include "kernel_config.h"
#define M_PI 3.141
#define GELU_SCALING_FACTOR v_rsqrt_f32(2.0f / M_PI)
#define VECTOR float64
#define VECTOR_SIZE 64

void main(tensor d_in_out, tensor inp) {
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    VECTOR packed_dinp;
    VECTOR packed_inp;
    VECTOR packed_dout;

#pragma loop_taken
    for (int d = depthStart; d < depthEnd; d += depthStep) {
        ifmCoords[depth] = d;

        
        // Perform element-wise operations
        for (int k = 0; k < VECTOR_SIZE; ++k) {
            // Load input tensors
            v_f32_ld_tnsr_b(ifmCoords, inp);
            v_f32_ld_tnsr_b(ifmCoords, d_in_out);
            float64 x = packed_inp;
            float64 cube = 0.044715f * x * x * x;
            float64 tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float64 tanh_out = v_f32_calc_fp_special_b(tanh_arg, 0, SW_TANH, 1);
            float64 coshf_out = v_f32_calc_fp_special_b(tanh_out, 0, SW_TANH, 1); //missing cosh in instrinsics
            float64 sech_out = 1.0f / (coshf_out * coshf_out);
            float64 local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp = (tanh_out * packed_dout);
        }

        // Store the result
        v_f32_st_tnsr(ifmCoords, d_in_out, packed_dinp);
    }
}