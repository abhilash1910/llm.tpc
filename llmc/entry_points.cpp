
#include "attention_fwd.hpp"
#include "layer_norm.hpp"
#include "attention_bwd.hpp"
#include "matmul_fwd.hpp"
#include "matmul_bwd.hpp"
#include "global_norm.hpp"
#include "gelu_fwd.hpp"
#include "gelu_bwd.hpp"
#include "fused_classifier_fwd.hpp"
#include "adamw.hpp"

#include "entry_points.hpp"
#include <stdio.h>

extern "C"
{

tpc_lib_api::GlueCodeReturn GetKernelGuids( _IN_    tpc_lib_api::DeviceId        deviceId,
                                            _INOUT_ uint32_t*       kernelCount,
                                            _OUT_   tpc_lib_api::GuidInfo*       guids)
{
    printf("GetKernelGuids called, deviceId=%d\n", deviceId);
    if (deviceId == tpc_lib_api::DEVICE_ID_GAUDI2)
    {
        if (guids != nullptr )
        {
           layernorm_fwd layernorm_fwdInstance;
           layernorm_fwdInstance.GetKernelName(guids[GAUDI_KERNEL_LAYER_NORM_FWD].name);
           attention_fwd attention_fwdInstance;
           attention_fwdInstance.GetKernelName(guids[GAUDI_KERNEL_ATTENTION_FWD].name);
           attention_bwd attention_bwdInstance;
           attention_bwdInstance.GetKernelName(guids[GAUDI_KERNEL_ATTENTION_BWD].name);
           matmul_fwd matmul_fwdInstance;
           matmul_fwdInstance.GetKernelName(guids[GAUDI_KERNEL_MATMUL_FWD].name);
           matmul_bwd matmul_bwdInstance;
           matmul_bwdInstance.GetKernelName(guids[GAUDI_KERNEL_MATMUL_BWD].name);
           global_norm global_normInstance;
           global_normInstance.GetKernelName(guids[GAUDI_KERNEL_GLOBALNORM].name);
           fused_classifier_fwd fused_classifier_fwdInstance;
           fused_classifier_fwdInstance.GetKernelName(guids[GAUDI_KERNEL_FUSED_CLASSIFIER_FWD].name);
           gelu_fwd gelu_fwdInstance;
           gelu_fwdInstance.GetKernelName(guids[GAUDI_KERNEL_GELU_FWD].name);
           gelu_bwd gelu_bwdInstance;
           gelu_bwdInstance.GetKernelName(guids[GAUDI_KERNEL_GELU_BWD].name);
           
           
        }

    }
   
    else
    {
        if (kernelCount != nullptr)
        {
            // currently the library support 0 kernels.
            *kernelCount = 0;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}


tpc_lib_api::GlueCodeReturn
InstantiateTpcKernel(_IN_  tpc_lib_api::HabanaKernelParams* params,
             _OUT_ tpc_lib_api::HabanaKernelInstantiation* instance)
{
    char kernelName [tpc_lib_api::MAX_NODE_NAME];

    ///////---Gaudi---
    ///////////////////////////////
    
    layernorm_fwd layernorm_fwdInstance;
    layernorm_fwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return layernorm_fwdInstance.GetGcDefinitions(params, instance);
    }
    attention_fwd attention_fwdInstance;
    attention_fwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return attention_fwdInstance.GetGcDefinitions(params, instance);
    }
    attention_bwd attention_bwdInstance;
    attention_bwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return attention_fwdInstance.GetGcDefinitions(params,instance);
    }
    matmul_fwd matmul_fwdInstance;
    matmul_fwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return matmul_fwdInstance.GetGcDefinitions(params,instance);
    }
    matmul_bwd matmul_bwdInstance;
    matmul_bwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return matmul_bwdInstance.GetGcDefinitions(params,instance);
    }
    gelu_fwd gelu_fwdInstance;
    gelu_fwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return gelu_fwdInstance.GetGcDefinitions(params,instance);
    }
    gelu_bwd gelu_bwdInstance;
    gelu_bwdInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return gelu_bwdInstance.GetGcDefinitions(params,instance);
    }
    global_norm global_normInstance;
    global_normInstance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return global_normInstance.GetGcDefinitions(params,instance);
    }
    

    printf("InstantiateTpcKernel: NOT FOUND, name=%s\n", kernelName);
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId,  tpc_lib_api::ShapeInferenceParams* inputParams,  tpc_lib_api::ShapeInferenceOutput* outputData)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

} // extern "C"