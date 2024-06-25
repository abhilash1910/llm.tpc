#include "kernel_config.h"
#define GELU_SCALING_FACTOR v_rsqrt_f32(2.0f / M_PI)
#define floatX float
struct SoftmaxParams{
 float Scale;
 float Offset;

};

SoftmaxParams softmax_fw_blockwide(tensor inp, int V, int P){

  const int depth = 0;

  const int5 indexSpaceStart = get_index_space_offset();
  const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

  int5 ifmCoords = {0, 0, 0, 0, 0};

  // DEPTH
  const int depthStep = VECTOR_SIZE;
  const int depthStart = indexSpaceStart[depth] * depthStep;
  const int depthEnd = indexSpaceEnd[depth] * depthStep;

  
  float thread_maxval = -INFINITY;
  float thread_sumval = 0.0f;

  // Get the input pointer for this thread
  int idx = depthStart;
  VECTOR x;
  VECTOR packed_x;

  // Special-case loop to handle the unaligned elements at the end of the array
  for (int i = (V + VECTOR_SIZE - 1) / VECTOR_SIZE + idx - VECTOR_SIZE; (i + 1) * VECTOR_SIZE > V; i -= VECTOR_SIZE) {
      for (int k = 0; k < VECTOR_SIZE; ++k) {
          if (i * VECTOR_SIZE + k >= V) {
              break; // bounds checking against real V (rather than padded P)
          }
          x[k] = v_ld_tnsr_i(inp, i * VECTOR_SIZE + k);
          float v = (float)x[k];
          float old_maxval = thread_maxval;
          thread_maxval = s_f32_max(thread_maxval, v);
          thread_sumval *= exp(old_maxval - thread_maxval);
          thread_sumval += exp(v - thread_maxval);
      }
  }

  // Main loop for the bulk of the iterations (no bounds checking required!)
  for (int i = idx; i >= 0; i -= VECTOR_SIZE) {
      packed_x = v_ld_tnsr_i(inp, i);
      for (int k = 0; k < VECTOR_SIZE; ++k) {
          float v = (float)packed_x[k];
          float old_maxval = thread_maxval;
          thread_maxval = s_f32_max(thread_maxval, v);
          thread_sumval *= exp(old_maxval - thread_maxval);
          thread_sumval += exp(v - thread_maxval);
      }
  }

  // Block Max Reduction -> Maths -> Block Sum Reduction
  float block_maxval = v_reduce_max(thread_maxval);
  thread_sumval *= exp(thread_maxval - block_maxval);
  float block_sumval = v_reduce_sum(thread_sumval);

  // Store the softmax parameters
  float inverse_sumval = 1.0f / block_sumval;
  return SoftmaxParams{inverse_sumval, block_maxval};
  
}


void main(tensor logits, tensor losses, tensor probs, tensor dloss,
          tensor targets, int B, int T, int V, int P, bool WRITE_LOGITS,
          bool WRITE_PROBS) {
    const int depth = 0;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;
    
    for(int idx = depthStart; idx < depthEnd; idx++){
    
    
      int ix = v_ld_tnsr_i({idx}, targets);
      
      SoftmaxParams sp = softmax_fw_blockwide(idx, V, P);
      
      if (indesSpaceStart[depth] == 0){
      
         float prob = exp((float)v_ld_tnsr_i({idx, ix}, logits) - sp.offset) * sp.scale;
         st_tnsr_i_v({idx}, losses, (floatX)(-log(prob)));
        }

        // Calculate the gradients directly and support writing probs for inference and debugging
        int remaining_elements = V;
        int i = indexSpaceStart[depth];
        while (remaining_elements > 0) {
            int vec_size = min(remaining_elements, VECTOR_SIZE);  // VECTOR_SIZE is hardware specific
            floatX logits_vec[VECTOR_SIZE], probs_vec[VECTOR_SIZE];
            v_ld_tnsr_partial_i({idx, i}, logits, vec_size, logits_vec);

            for (int k = 0; k < vec_size; ++k) {
                float prob = exp((float)logits_vec[k] - sp.offset) * sp.scale;
                probs_vec[k] = (floatX)prob;
                float indicator = (i + k == ix) ? 1.0f : 0.0f;
                logits_vec[k] = (floatX)((prob - indicator) * dloss);
            }

            // Reduce cache persistence for the overwritten logits
            if (WRITE_LOGITS) {
                v_st_tnsr_partial_i({idx, i}, logits, vec_size, logits_vec);
            }
            if (WRITE_PROBS) {
                v_st_tnsr_partial_i({idx, i}, probs, vec_size, probs_vec);
            }

            i += vec_size;
            remaining_elements -= vec_size;
        }
      
    int unaligned_start = V & ~(VECTOR_SIZE - 1); // round down to multiple of x128::size
    for (int i = indexSpaceStart[depth] + unaligned_start ; i < V; i++) {
        float prob = exp((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WRITE_LOGITS){
            st_tnsr_i_v({idx, i}, logits , (floatX)dlogit);
        }
        if (WRITE_PROBS) {
            st_tnsr_i_v({idx, i}, probs, (floatX)prob);
        }
      }
    
    
    }

}