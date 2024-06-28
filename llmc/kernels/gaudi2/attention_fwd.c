#include "kernel_config.h"

void attention_query_key_kernel(tensor preatt, tensor inp, int B, int T, int C, int NH) {
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    #pragma loop taken
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        int total_threads = B * NH * T * T;

        if (idx < total_threads) {
            int t2 = idx % T;
            int t = (idx / T) % T;
            if (t2 > t) {
                // autoregressive mask
                st_tnsr_i_v(idx, preatt, -INFINITY);
                continue;
            }
            int h = (idx / (T * T)) % NH;
            int b = idx / (NH * T * T);

            int C3 = C * 3;
            int hs = C / NH; // head size

            int5 query_t_coords = {b * T * C3 + t * C3 + h * hs, 0, 0, 0, 0};
            int5 key_t2_coords = {b * T * C3 + t2 * C3 + h * hs + C, 0, 0, 0, 0}; // +C because it's key

            float val = 0.0f;
            for (int i = 0; i < hs; i++) {
                float query_t_val = v_ld_tnsr_i(query_t_coords + i, inp);
                float key_t2_val = v_ld_tnsr_i(key_t2_coords + i, inp);
                val += query_t_val * key_t2_val;
            }
            val *= 1.0 / sqrtf(hs);

            st_tnsr_i_v(idx, preatt, val);
        }
    }
}


void attention_softmax_kernel1(tensor att, tensor preatt,
                                         int B, int T, int NH) {
                                         
                                         
     const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    #pragma loop taken
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        int total_threads = B * NH * T * T;

        if (idx < total_threads) {
            int h = idx % NH;
            int t = (idx / NH) % T;
            int b = idx / (NH * T);
            
            int5 preatt_bth = {preatt + b * NH * T * T + h * T * T +  t * T};
            int5 att_bth = {att + b * NH * T * T + h * T * T + t * T};
            
            float maxval = -10000.0f;
            
            for (int t2 = 0; t2 <= t; t2++) {
                float preatt_val = v_ld_tnsr_i(t2, preatt_bth);
                if (preatt_bth > maxval) {
                    maxval = preatt_bth;
                }
            }
      
            // calculate the exp and keep track of sum
            float expsum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                float preatt_val = v_ld_tnsr_i(t2, preatt_bth);
                float expv = exp(preatt_val - maxval);
                expsum += expv;
                st_tnsr_i_v(t2, att_bth, expv);
                
            }
            float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

            // normalize to get the softmax
            for (int t2 = 0; t2 < T; t2++) {
                if (t2 <= t) {
                    float att_val = v_ld_tnsr_i(t2, att_bth);
                    att_val *= expsum_inv;
                    st_tnsr_i_v(t2, att_bth, att_val);
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    st_tnsr_i_v(t2, att_bth, 0.0f);
                }
          }
    }                                    
                                         
}


void attention_value_kernel(tensor out, tensor att, tensor inp,
                            int B, int T, int C, int NH){
                            
                            
                            
      const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    #pragma loop taken
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        int total_threads = B * NH * T * T;

        if (idx < total_threads) {
            int h = idx % NH;
            int t = (idx / NH) % T;
            int b = idx / (NH * T);
            
            int C3 = C*3;
            int hs = C / NH; // head size
            
            int5 out_bth = {out + b * T * C + t * C +  h * hs};
            int5 att_bth = {att + b * NH * T * T + h * T * T + t * T};
            
            for (int t2 = 0; t2 <= t; t2++) {
                float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                float att_btht2 = v_ld_tnsr_i(t2, att_bth);
                for (int i = 0; i < hs; i++) {
                    float out_val = v_ld_tnsr_i(i, out_bth);
                    out_val += att_btht2 * v_ld_tnsr_i(i, value_t2);
                    st_tnsr_i_v(i, out_bth, out_val);
                 }
            }
         }   
     }                                                           
                            
}


void main( tensor out, tensor preatt, tensor att,
           tensor inp,
           int B, int T, int C, int NH,
           const int block_size){


    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

     #pragma loop taken
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        
        ifmCoords[depth] = idx;
        
        attention_query_key_kernel1(preatt, inp, B, T, C, NH);
        // softmax and value accumulation
        attention_softmax_kernel1(att, preatt, B, T, NH);
        attention_value_kernel1(out, att, inp, B, T, C, NH);

  }

}