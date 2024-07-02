#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class encoder_bwd {
public:
    
    struct KernelParams {
        int B;
        int T;
        int C;
        };
        
    encoder_bwd() = default;
    encoder_bwd(const encoder_bwd&) = delete;
    encoder_bwd& operator=(const encoder_bwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "encoder_bwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};