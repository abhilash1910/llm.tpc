#include "kernel_config.h"

void main(tensor out, tensor inp, tensor wte, tensor wpe, int B, int T, int C) {
    const int depth = 0;
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    // Process each index within the specified depth range
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;

        int linear_idx = (ifmCoords[depth]) * x128::size;
        int N = B * T * C;
        if (linear_idx >= N) { continue; }

        int bt = linear_idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = linear_idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;

        VECTOR  packed_out, wte128, wpe128;
  
        
        for (int k = 0; k < VECTOR_SIZE; ++k) {
             wte128 = v_ld_tnsr_partial_i(wte_ix, ifmCoords, i);
             wpe128 = v_ld_tnsr_partial_i(wpe_tc, ifmCoords, i);
             packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
        }

        st_tnsr_i_v(ifmCoords, out_btc, packed_out);
    }
}