#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class matmul_bwd {
public:
    
    struct KernelParams {
        int B;
        int T;
        int C;
        int OC;
        };
        
    matmul_bwd() = default;
    matmul_bwd(const matmul_bwd&) = delete;
    matmul_bwd& operator=(const matmul_bwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "matmul_bwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};