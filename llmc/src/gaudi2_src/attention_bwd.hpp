#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class attention_bwd {
public:
    
    struct KernelParams {
        int B;
        int T;
        int C;
        int NH;
        const int block_size;
        };
        
    attention_bwd() = default;
    attention_bwd(const attention_bwd&) = delete;
    attention_bwd& operator=(const attention_bwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "attention_bwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};