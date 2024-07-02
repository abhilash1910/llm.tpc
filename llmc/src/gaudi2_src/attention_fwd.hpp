#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class attention_fwd {
public:
    
    struct KernelParams {
        int B;
        int T;
        int C;
        int NH;
        const int block_size;
        };
        
    attention_fwd() = default;
    attention_fwd(const attention_fwd&) = delete;
    attention_fwd& operator=(const attention_fwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "attention_fwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};