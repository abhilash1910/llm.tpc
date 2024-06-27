#include "kernel_config.h"

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
            if (bias != NULL) {
                v_ld_tnsr_partial_i(&val, bias, ifmCoords, oc);
            }

            for (int k = 0; k < VECTOR_SIZE; k++) {
                float wrow[C];
                v_ld_tnsr_partial_i(wrow, weight, ifmCoords, oc * C + k);

                float inp_bt[C];
                v_ld_tnsr_partial_i(inp_bt, inp, ifmCoords, bt * C + k);

                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
            }

            v_st_tnsr_partial_i(out, ifmCoords, bt * OC + oc, val);
        }
    }
}
