vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

// Original code by Michele Morrone me@michelemorrone.eu / brutpitt@gmail.com
// https://github.com/BrutPitt/glslSmartDeNoise/blob/master/Shaders/frag.glsl
// This software is distributed under the terms of the BSD 2-Clause license

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D image_in;
layout(set = 0, binding = 1, rgba8ui) uniform writeonly restrict uimage2D image_out;

#define INV_SQRT_OF_2PI 0.39894228040143267793994605993439  // 1.0/SQRT_OF_2PI
#define INV_PI          0.31830988618379067153776752674503

layout(push_constant) uniform Parameters {
    uint Width;
    uint Height;
    float sigma;
    float kSigma;
    float threshold;
} params;

void main() {
    vec2 size = vec2(textureSize(image_in, 0));
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / size; //wSize in original code
    float radius = round(params.kSigma*params.sigma);
    float radQ = radius * radius;

    float invSigmaQx2 = .5 / (params.sigma * params.sigma);      // 1.0 / (sigma^2 * 2.0)
    float invSigmaQx2PI = INV_PI * invSigmaQx2;    // // 1/(2 * PI * sigma^2)

    float invThresholdSqx2 = .5 / (params.threshold * params.threshold);     // 1.0 / (params.sigma^2 * 2.0)
    float invThresholdSqrt2PI = INV_SQRT_OF_2PI / params.threshold;   // 1.0 / (sqrt(2*PI) * params.sigma)

    const vec3 centrPx = texture(image_in, uv).rgb / 255.0;

    float zBuff = 0.0;
    vec3 aBuff = vec3(0.0);

    vec2 d;
    for (d.x=-radius; d.x <= radius; d.x++) {
        float pt = sqrt(radQ-d.x*d.x);       // pt = yRadius: have circular trend
        for (d.y=-pt; d.y <= pt; d.y++) {
            float blurFactor = exp( -dot(d , d) * invSigmaQx2 ) * invSigmaQx2PI;

            vec3 walkPx = texture(image_in,uv+d/size).rgb / 255.0;

            vec3 dC = walkPx-centrPx;
            float deltaFactor = exp( -dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;

            zBuff += deltaFactor;
            aBuff += deltaFactor*walkPx;
        }
    }
    uvec4 value = uvec4(255.0*centrPx, 255);
    //uvec4 value = uvec4(160, 160, 128, 128);
    //uvec4 value = uvec4(round(255.0 * aBuff/zBuff), 0);
    imageStore(image_out, ivec2(gl_GlobalInvocationID.xy), min(value, 255));
}"
}