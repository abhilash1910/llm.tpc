#include "kernel_config.h"
#define VECTOR_SIZE 64
#define VECTOR float

void main(tensor out_dinp, tensor out_dweight, tensor out_dbias,
                         tensor dout, tensor inp, tensor weight,
                         int B, int T, int C, int OC) {
    const int depth = 0;
    const int depthStep = VECTOR_SIZE;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;
    int5 coords = {0, 0, 0, 0, 0};
    int d;

    // Backward into inp first, parallelize over B,T
    #pragma loop taken
    for (int b = indexSpaceStart[depth]; b < indexSpaceEnd[depth]; ++b) {
        for (int t = 0; t < T; ++t) {
            float accum[VECTOR_SIZE] = {0.0f};

            for (int oc = 0; oc < OC; oc += VECTOR_SIZE) {
                coords[0] = b; coords[1] = t; coords[2] = oc;
                v_f32_ld_tnsr_b(coords, dout);
                v_f32_ld_tnsr_b(coords, out_dinp);

                for (int k = 0; k < VECTOR_SIZE; ++k) {
                    for (int c = 0; c < C; ++c) {
                        coords[3] = c;
                        v_f32_ld_tnsr_partial_b(coords , d, dout, k);
                        coords[4] = c + k;
                        accum[k] += d * coords[4];
                        v_f32_ld_tnsr_partial_b(coords, accum, weight, k);
                    }
                }

                for (int k = 0; k < VECTOR_SIZE; ++k) {
                    coords[3] = k;
                    v_f32_st_tnsr_partial(coords, accum, out_dinp, k, 0);
                }
            }
        }
    }
    
    // Backward into weight/bias, parallelize over output channels OC
    #pragma loop taken
    for (int oc = indexSpaceStart[depth]; oc < indexSpaceEnd[depth]; ++oc) {
        float64 sum = 0.0f;

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                float64 accum[VECTOR_SIZE] = {0.0f};

                for (int c = 0; c < C; c += VECTOR_SIZE) {
                    coords[0] = b; coords[1] = t; coords[2] = oc; coords[3] = c;
                    float64 d = v_f32_ld_tnsr_b(coords, dout);
                    sum += d;

                    for (int k = 0; k < VECTOR_SIZE; ++k) {
                        coords[4] = k;
                        accum[k] += d * coords[4];
                        v_f32_ld_tnsr_b(coords, inp);
                    }
                }

                for (int k = 0; k < VECTOR_SIZE; ++k) {
                    coords[3] = k;
                    v_f32_st_tnsr(coords, out_dweight, accum[k]);
                }
            }
        }

        if (!out_dbias ) {
            int5 coords_ = {oc, 0, 0, 0, 0}; 
            v_f32_st_tnsr(coords_, out_dbias, sum);
        }
    }
    
}
