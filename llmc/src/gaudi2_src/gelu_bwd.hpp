#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class gelu_fwd {
public:
    

    gelu_bwd() = default;
    gelu_bwd(const gelu_fwd&) = delete;
    gelu_bwd& operator=(const gelu_bwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "gelu_bwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};