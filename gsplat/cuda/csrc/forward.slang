// forward.slang

// Define a constant buffer to hold size information
cbuffer SizeBuffer {
    uint size;
};

// Define input and output buffers
RWStructuredBuffer<float> input : register(u0);
RWStructuredBuffer<float> output : register(u1);

// Define the compute shader entry point
[Differentiable]
void forwardKernel(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Define an entry point for setting up and dispatching the shader
void forward(uint size, RWStructuredBuffer<float> input, RWStructuredBuffer<float> output) {
    // Bind the size buffer
    SizeBuffer.size = size;

    // Bind the input and output buffers
    this.input = input;
    this.output = output;

    // Calculate dispatch parameters
    uint threadsPerGroup = 256;
    uint groups = (size + threadsPerGroup - 1) / threadsPerGroup;

    // Dispatch the compute shader
    [numthreads(threadsPerGroup, 1, 1)]
    forwardKernel(dispatchThreadID);
}
