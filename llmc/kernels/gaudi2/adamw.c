#include "kernel_config.h"

void main(tensor params_memory, tensor grads_memory, tensor m_memory, tensor v_memory, 
          tensor params_out, tensor m_out, tensor v_out,
          tensor master_params_memory, 
          float beta1, float beta2, float epsilon, float learning_rate, 
          float beta1_correction, float beta2_correction, float weight_decay, 
          float grad_scale, unsigned int seed) 
{
  const int depth = 0;
  
  const int5 indexSpaceStart = get_index_space_offset();
  const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

  int5 coords = {0, 0, 0, 0};

  // DEPTH
  const int depthStep = VECTOR_SIZE;
  const int depthStart = indexSpaceStart[depth] * depthStep;
  const int depthEnd = indexSpaceEnd[depth] * depthStep;

  
  VECTOR p, g, m_t, v_t;
  VECTOR m_t_out, v_t_out, p_out;
  VECTOR one = 1.0;

#pragma loop_taken
  for (int d = depthStart; d < depthEnd; d += depthStep) {
            coords[depth] = d;
            // Load values from tensors
            p = v_ld_tnsr_i(coords, params_memory);
            g = v_ld_tnsr_i(coords, grads_memory);
            m_t = v_ld_tnsr_i(coords, m_memory);
            v_t = v_ld_tnsr_i(coords, v_memory);
            // Scale gradient
            g *= grad_scale;

            // Update the first moment (momentum)
            m_t_out = v_rlerp_f32(g, m_t, beta1);
            
            // Update the second moment (RMSprop)
            v_t_out = v_rlerp_f32(g * g, v_t, beta2);

            // Compute bias-corrected moment estimates
            VECTOR m_hat = m_t_out / beta1_correction;
            VECTOR v_hat = v_t_out / beta2_correction;

            // Fetch the old value of this parameter as a float, from either source
            VECTOR old_param = (master_params_memory != NULL) ? v_ld_tnsr_i(coords, master_params_memory) : p;

            // Update this parameter
            p_out = old_param - (learning_rate * (m_hat / (v_rsqrt_f32(v_hat) + epsilon) + weight_decay * old_param));

            // Stochastic rounding
            //unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x, seed);
            //stochastic_rounding(p_out, &p, random);

            // Store updated values back to tensors
            st_tnsr_i_v(coords, params_out, p);
            st_tnsr_i_v(coords, m_out, m_t_out);
            st_tnsr_i_v(coords, v_out, v_t_out);

            // Write the full, float version of the param into our master copy, if we maintain one
            if (master_params_memory != NULL) {
                st_tnsr_i_v(coords, master_params_memory, p_out);
            }
          }
}
