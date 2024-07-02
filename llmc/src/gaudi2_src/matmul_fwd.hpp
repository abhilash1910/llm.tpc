#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class matmul_fwd {
public:
    
    struct KernelParams {
        int B;
        int T;
        int C;
        int OC;
        };
        
    matmul_fwd() = default;
    matmul_fwd(const matmul_fwd&) = delete;
    matmul_fwd& operator=(const matmul_fwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "matmul_fwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};