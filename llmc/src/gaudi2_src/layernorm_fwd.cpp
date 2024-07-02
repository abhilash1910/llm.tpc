
#include <vector>
#include <cstring>
#include <iostream>
#include "gelu_fwd.hpp"
#include "common_utils.hpp"

extern unsigned char _binary___gelu_fwd_o_start;
extern unsigned char _binary___gelu_fwd_o_end;

 tpc_lib_api::GlueCodeReturn gelu_fwd::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_gelu_fwd");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn AddF32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    const int c_unrollCount = 4;

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
    
    //memcpy(outputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    //round up to elementsInVec and divide by elementsInVec.
    //unsigned depthIndex = (outputSizes[0] + 63) / 64;
    unsigned depthIndex = VECTOR_SIZE;
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = depthIndex;
	
    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

     for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        for (unsigned j = 0; j < out_defs->indexSpaceRank; j++)
        {
            out_defs->inputTensorAccessPattern[i].mapping[j].indexSpaceDim     = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].a = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].start_b = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].end_b   = 0;
        }
    }
    for (unsigned int i = 0; i < out_defs->indexSpaceRank; i++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[i].indexSpaceDim     = i;
        out_defs->outputTensorAccessPattern[0].mapping[i].a = 0;
        out_defs->outputTensorAccessPattern[0].mapping[i].start_b = 0;
        out_defs->outputTensorAccessPattern[0].mapping[i].end_b   = 0;
    }
    
    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr =0;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___gelu_fwd_o_end - &_binary___gelu_fwd_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___gelu_fwd_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}