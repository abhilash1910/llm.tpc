#pragma once

#include <cstring>
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class fused_classifier_fwd {
public:
    // NB! Should be aligned with same structure in glue code
    struct KernelParams {
        int B;
        int T;
        int V;
        int P;
        bool WRITE_LOGITS;
        bool WRITE_PROBS;
        };

    fused_classifier_fwd() = default;
    fused_classifier_fwd(const fused_classifier_fwd&) = delete;
    fused_classifier_fwd& operator=(const fused_classifier_fwd&) = delete;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName [tpc_lib_api::MAX_NODE_NAME]) {
        strcpy(kernelName, "fused_classifier_fwd");
        return tpc_lib_api::GLUE_SUCCESS;
    }

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);

};