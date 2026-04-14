#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#define M 2048
#define K 64
#define N 2048



__global__ void matmul(const float* __restrict__ matrixA,
                       const float* __restrict__ matrixB,
                       float*       __restrict__ matrixC)
{
    __shared__ __half smenA[32 * 16];   // 32 rows × K-slice(16) cols
    __shared__ __half smenB[16 * 16];   // K-slice(16) rows × 16 cols

    const int lane        = threadIdx.x % 32;
    const int warpID      = threadIdx.x / 32;
    const int warp_row    = warpID / 2;          // 0 or 1
    const int warp_col    = warpID % 2;          // 0 or 1

    const int blockRowID  = blockIdx.y * 32;
    const int blockColID  = blockIdx.x * 16;

    const int warpRowBase = warp_row * 16;
    const int warpColBase = warp_col * 8;

    // mma register layout helpers
    const int group        = lane / 4;    // 0–7: selects A-col and B-col
    const int tid_in_group = lane % 4;    // 0–3: selects A/B row pair

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int idx = 0; idx < K; idx += 16)
    {
        for (int i = threadIdx.x; i < 512; i += blockDim.x)
        {
            int r = i / 16;
            int c = i % 16;
            int global_row = blockRowID + r;
            int global_col = idx + c;
            smenA[r * 16 + c] = __float2half(matrixA[global_row * K + global_col]);
        }

        for (int i = threadIdx.x; i < 256; i += blockDim.x)
        {
            int r = i / 16;
            int c = i % 16;
            int global_row = idx       + r;
            int global_col = blockColID + c;     // FIX: full 16 cols
            smenB[r * 16 + c] = __float2half(matrixB[global_row * N + global_col]);
        }

        __syncthreads();

        const int a_row0 = warpRowBase + group;
        const int a_row1 = warpRowBase + group + 8;
        const int a_col0 = tid_in_group * 2;
        const int a_col1 = tid_in_group * 2 + 8;

        uint32_t a0 = *reinterpret_cast<const uint32_t*>(&smenA[a_row0 * 16 + a_col0]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(&smenA[a_row1 * 16 + a_col0]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(&smenA[a_row0 * 16 + a_col1]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(&smenA[a_row1 * 16 + a_col1]);


        const int b_col  = warpColBase + group;
        const int b_row0 = tid_in_group * 2;
        const int b_row1 = tid_in_group * 2 + 8;

        uint32_t b0, b1;
        {
            __half lo, hi;
            lo = smenB[ b_row0      * 16 + b_col];
            hi = smenB[(b_row0 + 1) * 16 + b_col];
            b0 = (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);

            lo = smenB[ b_row1      * 16 + b_col];
            hi = smenB[(b_row1 + 1) * 16 + b_col];
            b1 = (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
        }

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );

        __syncthreads();
    }

    const int out_row = blockRowID + warpRowBase + group;
    const int out_col = blockColID + warpColBase + tid_in_group * 2;

    matrixC[ out_row      * N + out_col]     = d0;
    matrixC[ out_row      * N + out_col + 1] = d1;
    matrixC[(out_row + 8) * N + out_col]     = d2;
    matrixC[(out_row + 8) * N + out_col + 1] = d3;
}

static void cpu_matmul(const float* A, const float* B, float* C, int Mm, int Kk, int Nn)
{
    for (int m = 0; m < Mm; m++)
        for (int n = 0; n < Nn; n++) {
            float acc = 0.f;
            for (int k = 0; k < Kk; k++)
                acc += __half2float(__float2half(A[m * Kk + k]))
                     * __half2float(__float2half(B[k * Nn + n]));
            C[m * Nn + n] = acc;
        }
}

static void fill_random(float* p, int n) {
    for (int i = 0; i < n; i++) p[i] = (float)(rand() % 8 - 4);
}

static int verify(const float* gpu, const float* cpu, int n) {
    int errs = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(gpu[i] - cpu[i]) > 0.1f) {
            if (errs < 8)
                printf("  MISMATCH [%d]: gpu=%.1f cpu=%.1f\n", i, gpu[i], cpu[i]);
            errs++;
        }
    }
    return errs;
}

int main(void)
{
    srand(42);


    float *hA = (float*)malloc(M * K * sizeof(float));
    float *hB = (float*)malloc(K * N * sizeof(float));
    float *hC = (float*)malloc(M * N * sizeof(float));
    float *rC = (float*)malloc(M * N * sizeof(float));

    fill_random(hA, M * K);
    fill_random(hB, K * N);
    cpu_matmul(hA, hB, rC, M, K, N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0,  M * N * sizeof(float));

    dim3 grid(N / 16, M / 32);   // one warp per 16×8 output tile
    dim3 block(128);

    matmul<<<grid, block>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    int errs = verify(hC, rC, M * N);
    printf("%s — %d / %d elements checked\n", errs ? "FAIL" : "PASS", M*N - errs, M*N);

    free(hA); free(hB); free(hC); free(rC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return errs != 0;
}
