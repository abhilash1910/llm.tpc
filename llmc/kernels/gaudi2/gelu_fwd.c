#include "kernel_config.h"
#define GELU_SCALING_FACTOR v_rsqrt_f32(2.0f / M_PI)

void main(tensor out, const tensor in){


  const int depth = 0;
  
  const int5 indexSpaceStart = get_index_space_offset();
  const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;
  
  int5 ifmCoords = {0, 0, 0, 0, 0};
  
  const int depthStep = VECTOR_SIZE;
  const int depthStart = indexSpaceStart[depth] * depthStep;
  const int depthEnd = indexSpaceEnd[depth] * depthStep;
  
  VECTOR packed_inp, packed_out;
  
  #pragma loop taken
  for(int d = depthStart; d< depthEnd; d++){
  
    packed_inp = v_ld_tnsr_i(ifmCoords, in);
    
    #pragma loop taken
    for(int k =0; k < VECTOR_SIZE ; k++){
    
      float xi = (float)packed_inp[k];
      float cube = 0.044715f * xi * xi * xi;
      packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)))); 
    
    }
    
  st_tnsr_i_v(ifmCoords, output, packed_out);
  
  }


}