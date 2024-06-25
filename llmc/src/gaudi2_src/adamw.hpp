#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class Adamw {
public:
    // NB! Should be aligned with same structure in glue code
    struct KernelParams {
        float beta1;
        float beta2;
        float epsilon;
        float learning_rate;
        float beta1_correction;
        float beta2_correction;
        float weight_decay;
        float grad_scale;
        unsigned int seed;
        };

    Adamw() = default;
    Adamw(const Adamw&) = delete;
    Adamw& operator=(const Adamw&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "adamw");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};