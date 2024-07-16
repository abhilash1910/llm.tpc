#include "kernel_config.h"
#define M_PI 3.141
#define GELU_SCALING_FACTOR v_rsqrt_f32(2.0f / M_PI)
#define VECTOR float64
#define VECTOR_SIZE 64
#define floatX float64

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
  
    
    
    #pragma loop taken
    for(int k =0; k < VECTOR_SIZE ; k++){
    
      packed_inp = v_f32_ld_tnsr_b(ifmCoords, in);
      float64 xi = packed_inp;
      float64 cube = 0.044715f * xi * xi * xi;
      packed_out = (floatX)(0.5f * xi * (1.0f + v_f32_calc_fp_special_b(GELU_SCALING_FACTOR * (xi + cube), 0, SW_TANH, 1))); 
    
    }
    
  v_f32_st_tnsr(ifmCoords, out, packed_out);
  
  }


}