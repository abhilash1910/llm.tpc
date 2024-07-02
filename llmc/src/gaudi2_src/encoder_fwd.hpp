#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class encoder_fwd {
public:
    
    struct KernelParams {
        int B;
        int T;
        int C;
        };
        
    encoder_fwd() = default;
    encoder_fwd(const encoder_fwd&) = delete;
    encoder_fwd& operator=(const encoder_fwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "encoder_fwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};