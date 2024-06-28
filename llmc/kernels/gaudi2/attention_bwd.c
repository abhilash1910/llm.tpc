#include "kernel_config.h"

// Bare back backward implementation - needs refinement with graph
void main(tensor dinp, tensor dpreatt, tensor datt,
          tensor dout, tensor inp, tensor att,
          int B, int T, int C, int NH,
          const int block_size) {

    const int depth = 0;
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;

        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0 / sqrtf(hs);

        int total_threads = B * NH * T;
        int num_blocks = ceil_div(total_threads, block_size);

        for (int tid = 0; tid < total_threads; tid++) {
            int h = tid % NH;
            int t = (tid / NH) % T;
            int b = tid / (NH * T);

            float* att_bth = att + b * NH * T * T + h * T * T + t * T;
            float* datt_bth = datt + b * NH * T * T + h * T * T + t * T;
            float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
            float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
            float* query_t = inp + b * T * C3 + t * C3 + h * hs;

            // Backward pass 4, through the value accumulation
            float* dout_bth = dout + b * T * C + t * C + h * hs;
            for (int t2 = 0; t2 < T; t2++) {
                float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                float datt_bth2 = v_ld_tnsr_i(t2, datt_bth);
                float att_bth2 = v_ld_tensor(t2, att_bth);
                for (int i = 0; i < hs; i++) {
                    float value_t2i = v_ld_tnsr_i(i, value_t2);
                    float doubt_bthi = v_ld_tnsr_i(i, dout_bth);
                    float dvalue_t2i = v_ld_tnsr_i(i, dvalue_t2);
                    datt_bth2 += value_t2i * dout_bthi;
                    dvalue_t2i += att_bth2 * dout_bthi;
                    st_tnsr_i_v(i, dvalue_t2, dvalue_t2i);
                    
                }
                st_tnsr_i_v(i, datt_bth, datt_bth);
            }

            // Backward pass 2 & 3, the softmax
            for (int t2 = 0; t2 <= t; t2++) {
                float att_bth2 = v_ld_tensor(t2, att_bth);
                float datt_bth2 = v_ld_tensor(t2, datt_bth);
                for (int t3 = 0; t3 <= t; t3++) {
                    float att_bth3 = v_ld_tensor(t3, att_bth);
                    float dpreatt_bth3 = v_ld_tensor(t3, dpreatt_bth);
                    float indicator = t2 == t3 ? 1.0f : 0.0f;
                    float local_derivative = att_bth2 * (indicator - att_bth3);
                    dpreatt_bth3 += scale * local_derivative * datt_bth2;
                    st_tnsr_i_v(i, dpreatt_bth, dpreatt_bth3);
                }
            }

            // Backward pass 1, the query @ key matmul
            for (int t2 = 0; t2 <= t; t2++) {
                float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                float dpreatt_bth2 = v_ld_tensor(t2, dpreatt_bth);
                for (int i = 0; i < hs; i++) {
                    float dquery_ti = v_ld_tensor(i, dquery_t);
                    float query_ti = v_ld_tensor(i, query_t);
                    float dkey_t2i = v_ld_tensor(i, dkey_t2);
                    float key_t2i = v_ld_tensor(i, key_t2);
                    dquery_ti += key_t2i * dpreatt_bth2;
                    dkey_t2i += query_ti * dpreatt_bth2;
                    st_tnsr_i_v(i, dquery_t, dquery_ti);
                    st_tnsr_i_v(i, key_t2, key_t2i);
                }
            }
        }
    }
}
