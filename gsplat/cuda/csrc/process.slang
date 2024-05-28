[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void project_gaussians_backward( DiffTensorView input, DiffTensorView v_mean3d, DiffTensorView v_scale, DiffTensorView v_quat, DiffTensorView output)
{

    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= input.size(0))
        return;

    bwd_diff(project_gaussians_backward)(input, v_mean3d, v_scale, v_quat, output);
}


[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void rasterize_backward( DiffTensorView input, DiffTensorView v_xy, DiffTensorView v_conic, DiffTensorView v_colors, DiffTensorView v_opacity, DiffTensorView output)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= input.size(0))
        return;

    bwd_diff(rasterize_backward)(input, v_xy, v_conic, v_colors, v_opacity, output);
}