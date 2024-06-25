#include "kernel_config.h"
#define GELU_SCALING_FACTOR v_rsqrt_f32(2.0f / M_PI)

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

        // Load input tensors
        packed_inp = v_ld_tnsr_i(ifmCoords, inp);
        packed_dout = v_ld_tnsr_i(ifmCoords, d_in_out);

        // Perform element-wise operations
        for (int k = 0; k < VECTOR_SIZE; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (half)(local_grad * (float)packed_dout[k]);
        }

        // Store the result
        st_tnsr_i_v(ifmCoords, d_in_out, packed_dinp);
    }
}