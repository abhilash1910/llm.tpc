#ifndef _ENTRY_POINTS_HPP_
#define _ENTRY_POINTS_HPP_

extern "C"
{

typedef enum
{
    GAUDI_KERNEL_LAYERNORM_FWD,
    GAUDI_KERNEL_ATTENTION_FWD,
    GAUDI_KERNEL_ATTENTION_BWD,
    GAUDI_KERNEL_MATMUL_FWD,
    GAUDI_KERNEL_MATMUL_BWD,
    GAUDI_KERNEL_GLOBALNORM,
    GAUDI_KERNEL_FUSED_CLASSIFIER_FWD,
    GAUDI_KERNEL_ADAMW,
    GAUDI_KERNEL_GELU_FWD,
    GAUDI_KERNEL_GELU_BWD

} Gaudi_Kernel_Name_e;


/*
 ***************************************************************************************************
 *   @brief This function returns exported kernel names
 *
 *   @param deviceId    [in] The type of device E.g. dali/gaudi etc.* 
 *   @param kernelCount [in/out] The number of strings in 'names' argument.
 *                      If the list is too short, the library will return the
 *                      required list length.
 *   @param guids       [out]  List of structure to be filled with kernel guids.
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
tpc_lib_api::GlueCodeReturn GetKernelGuids( _IN_    tpc_lib_api::DeviceId        deviceId,
                                            _INOUT_ uint32_t*                    kernelCount,
                                            _OUT_   tpc_lib_api::GuidInfo*       guids);

/*
 ***************************************************************************************************
 *   @brief This kernel library main entry point, it returns all necessary
 *          information about a kernel to execute on device.
 *
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
tpc_lib_api::GlueCodeReturn
InstantiateTpcKernel(_IN_  tpc_lib_api::HabanaKernelParams* params,
             _OUT_ tpc_lib_api::HabanaKernelInstantiation*instance);

tpc_lib_api::GlueCodeReturn 
GetShapeInference(_IN_ tpc_lib_api::DeviceId deviceId,  _IN_ tpc_lib_api::ShapeInferenceParams* inputParams,  _OUT_ tpc_lib_api::ShapeInferenceOutput* outputData);

} // extern "C"
#endif