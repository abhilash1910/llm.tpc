
#include <vector>
#include <cstring>
#include <iostream>
#include "gelu_fwd.hpp"
#include "common_utils.hpp"

extern unsigned char _binary___global_norm_o_start;
extern unsigned char _binary___global_norm_o_end;

 tpc_lib_api::GlueCodeReturn global_norm::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_global_norm");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn global_norm::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    const int c_unrollCount = 4;
    global_norm* call_params = static_cast<global_norm*>(in_defs->nodeParams.nodeParams);
    
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
    {
        in_defs->inputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    
    // validate input and output data type
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM];
    uint64_t* indexSpaceSizes         = in_defs->inputTensors[0].geometry.maxSizes;
    outputSizes[0] = indexSpaceSizes[0];
    //uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    //memcpy(outputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    //round up to elementsInVec and divide by elementsInVec.
    //unsigned depthIndex = (outputSizes[0] + 63) / 64;
    unsigned depthIndex = VECTOR_SIZE;
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = depthIndex;
	  unsigned elementsInVec = 64;
    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

    for(unsigned int i = 0;i < in_defs->inputTensorNr; i++) {
            //out_defs->inputTensorAccessPattern[ii].allRequired = true;
            out_defs->inputTensorAccessPattern[i].mapping[0].indexSpaceDim      = 0;
            out_defs->inputTensorAccessPattern[i].mapping[0].a        = 0;
            out_defs->inputTensorAccessPattern[i].mapping[0].start_b  = 0;
            out_defs->inputTensorAccessPattern[i].mapping[0].end_b    = elementsInVec - 1;
        }
     
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

    
    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    
    out_defs->kernel.paramsNr =1;
    out_defs->kernel.scalarParams[0] = static_cast<uint32_t>(call_params->count);
    

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___global_norm_o_end - &_binary___global_norm_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___global_norm_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
