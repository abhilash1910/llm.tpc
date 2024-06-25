#include "kernel_config.h"
const int VECTOR_SIZE = 128; 
const int WARP_SIZE= 128;

void wpe_backward(tensor dwpe, tensor dout, tensor inp, int B, int T, int C, unsigned int seed) {
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

     VECTOR packed_dout, packed_dwpe;
    // Process each index within the specified depth range
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;

        int linear_idx = idx * VECTOR_SIZE;
        if (linear_idx >= T * C) { continue; }

        int t = linear_idx / C;
        int c = linear_idx % C;
        float accum[VECTOR_SIZE] = {0.0f};

        for (int b = 0; b < B; b++) {
            packed_dout = v_ld_tnsr_partial_i(dout, ifmCoords, b * T * C + t * C + c);
            for (int k = 0; k < VECTOR_SIZE; k++) {
                accum[k] += (float)packed_dout[k];
            }
        }

        floatX* dwpe_tc = &dwpe[t * C + c];
        
        for (int k = 0; k < VECTOR_SIZE; k++) {
            packed_dwpe[k] = v_ld_tnsr_partial_i(dwpe_tc, ifmCoords, k);
        }
        // Disable stochastic rounding (To do)
        //for (unsigned int k = 0; k < x128::size; k++) {
            // We use stochastic rounding to go from FP32 to BF16 but the seed should be deterministic
        //    stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k], seed + k);
        //}

        for (int k = 0; k < x128::size; k++) {
            st_tnsr_i_v(ifmCoords, dwpe_tc + k, packed_dwpe[k]);
        }
    }
}


void wte_backward_kernel(tensor dwte, tensor bucket_info, tensor workload_indices, tensor dout, tensor inp,
                         unsigned int seed, int B, int T, int C) {
    const int depth = 0;
    
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // Depth step and start/end indexes for the TPC parallelism
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    // Loop over the buckets
    for (int idx = depthStart; idx < depthEnd; idx += depthStep) {
        ifmCoords[depth] = idx;
        
        int bucket = idx / (depthEnd - depthStart);
        int lane_id = idx % WARP_SIZE;
        int c_per_warp = WARP_SIZE * VECTOR_SIZE;

        int bucket_start_idx = v_ld_tnsr_partial_i(bucket_info, ifmCoords, bucket * 4 + 0);
        int bucket_size = v_ld_tnsr_partial_i(bucket_info, ifmCoords, bucket * 4 + 1);
        int bucket_ix = v_ld_tnsr_partial_i(bucket_info, ifmCoords, bucket * 4 + 2);
        int bucket_c = v_ld_tnsr_partial_i(bucket_info, ifmCoords, bucket * 4 + 3);
        int c = bucket_c * c_per_warp + (lane_id * VECTOR_SIZE);

        // Skip if channels exceed C or if no items to process in this warp, not mandatory
        if (c >= C || (idx / VECTOR_SIZE) >= bucket_size) { continue; }

        float accum[VECTOR_SIZE] = {0.0f};

        // Accumulate gradients from dout
        for (int item = idx / VECTOR_SIZE; item < bucket_size; item += depthEnd / VECTOR_SIZE) {
            int bt = v_ld_tnsr_partial_i(workload_indices, ifmCoords, bucket_start_idx + item);

            for (int k = 0; k < VECTOR_SIZE; k++) {
                float dout_value = v_ld_tnsr_partial_i(dout, ifmCoords, bt * C + c + k);
                accum[k] += dout_value;
            }
        }

        // Load dwte
        floatX* dwte_ix = &dwte[bucket_ix * C + c];
        VECTOR packed_in_out = v_ld_tnsr_i( ifmCoords, dwte_ix);

        // Accumulate shared memory into warp 0's registers
        for (int k = 0; k < VECTOR_SIZE; k++) {
            accum[k] += v_ld_tnsr_partial_i(accum_shared, ifmCoords, lane_id + k * depthEnd);
        }

        // Stochastic rounding and storing the result back to dwte
        //for (unsigned int k = 0; k < VECTOR_SIZE; k++) {
        //    stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + k);
       // }
       
        st_tnsr_i_v(dwte_ix, ifmCoords, packed_in_out);
    }
}

// to do use sorted gpu copying algo
void encoder_backward_naive(tensor dwte, tensor dwpe, tensor dout, tensor inp,
          int B, int T, int C){
   
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

        // Calculate the channel position
        int c = idx;
        if (c >= C) {
            continue;
        }

        int BT = B * T;

        // Loop over each B*T element
        for (int i = 0; i < BT; i++) {
            int t = i % T;
            int ix = v_ld_tnsr_partial_i(inp, ifmCoords, i);
            
            // Load dout
            float dout_btc = v_ld_tnsr_partial_i(dout, ifmCoords, i * C + c);
            
            // Update dwte
            float dwte_value = v_ld_tnsr_partial_i(dwte, ifmCoords, ix * C + c);
            dwte_value += dout_btc;
            st_tnsr_i_v(dwte, ifmCoords, ix * C + c, dwte_value);
            
            // Update dwpe
            float dwpe_value = v_ld_tnsr_partial_i(dwpe, ifmCoords, t * C + c);
            dwpe_value += dout_btc;
            st_tnsr_i_v(dwpe, ifmCoords, t * C + c, dwpe_value);
        }
    }
}


void main(tensor dwte, tensor dwpe, tensor dout, tensor inp,
          int B, int T, int C){
          
          
    encoder_backward_naive(dwte, dwpe, dout, inp, B, T, C);      
          
}
















}