#include "kernel_config.h"
#define VECTOR_SIZE 64

void main(tensor out, tensor inp, tensor weight, tensor bias, int B, int T, int C, int OC) {
    int BT = B * T;
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;

        int bt = ifmCoords[0];
        int oc = ifmCoords[1];
        
        if (bt < BT && oc < OC) {
            float val = 0.0f;
            if (!bias) {
                v_f32_ld_tnsr_partial_b(ifmCoords, &val, bias, oc);
            }

            for (int k = 0; k < VECTOR_SIZE; k++) {
                float wrow[1024];
                //float *wrow = (float *)malloc(C * sizeof(float));
                
                v_f32_ld_tnsr_partial_b(ifmCoords,wrow, weight, oc * C + k);
                
                float inp_bt[1024];
                //float *inp_bt = (float*)malloc(C * sizeof(float));
                v_f32_ld_tnsr_partial_b(ifmCoords, inp_bt, inp,  bt * C + k);

                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
            }

            v_f32_ld_tnsr_partial_b(ifmCoords, out, bt * OC + oc, val);
        }
    }
}
