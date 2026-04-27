#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <iostream>

#define SEQ_LEN 2048        
#define D_MODEL 2048
#define NUM_HEADS 8
#define HEAD_DIM 256

#define Br 64  // for query
#define Bc 32  // for key and value

#define SCALE (1.0f / sqrtf((float)HEAD_DIM))

__global__ void FlashAttention(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ out,
    int seq_len , int d_model)

{
    __align__(16) extern __shared__ __half smem[];

    __half* smenA    = smem;
    __half* smenB    = smenA    + 16  * 16;
    __half* qshared  = smenB    + 16  * 8;
    __half* kshared = qshared  + 8   * HEAD_DIM;
    __half* vshared  = kshared + 4   * HEAD_DIM; 
    float*  output   = (float*)(vshared + 32 * HEAD_DIM); // V
    float* m = (float*)(output + 64 * 32);  
    float* l = m + 64;       
    float* alpha_s = l + 64;   // 64 floats
    float* beta_s  = alpha_s + 64;                

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;
    const int tid     = threadIdx.x;

    const int lane  = tid % 32;
    const int group = lane / 4; 

    const long long base = (long long)batchid * NUM_HEADS * seq_len * d_model + 
                (long long)headid * seq_len * d_model;

    const half* Qptr = Q   + base;
    const half* Kptr = K   + base;
    const half* Vptr = V   + base;
          half* optr = out + base;
    
    // tile id -- 2048 * 256 / 64 * 32 -> 32 * 8 = 256 (for query)
    // total rows and total cols -- 256 / 32 - 8 cols and 2048 / 64 - 32 rows (for query)

    // total rows and total cols -- 256 / 32 - 8 cols and 2048 / 32 - 64 rows (for query)
    for(int i = tid ; i < 64 ; i += blockDim.x)
        m[i] = -FLT_MAX , l[i] = 0.f;

    __syncthreads();

    const int rowtileID = tileid;
    for(int rowid = 0 ; rowid < 64 ; rowid++)
    {
        for(int i = tid ; i < 64 * 32 ; i += blockDim.x)
            output[i] = 0.f;
        __syncthreads();

        for(int coltileID = 0 ; coltileID < 8 ; coltileID++)
        {
            // 64 rows, 4 float4s per row
            for(int i = tid; i < 64 * 4; i += blockDim.x)
            {
                int r     = i / 4;    // which row (0..63)
                int chunk = i % 4;    // which float4 within the row (0..3), covers cols 0,8,16,24

                *reinterpret_cast<float4*>(&qshared[r * 32 + chunk * 8]) =
                *reinterpret_cast<const float4*>(
                    &Qptr[rowtileID * 64 * 256 + coltileID * 32 + r * 256 + chunk * 8]);
            }


            for(int i = tid ; i < 32 * 32 ; i += blockDim.x)
                {
                    int r = i / 32;
                    int c = i % 32;

                    kshared[c * 32 + r] = Kptr[rowid * 32 * 256 + coltileID * 32 + r * 256 + c];
                }

            // we loaded Q and K.T , 64 * 32 and 32 * 32
            
            {
                const int totalrows = 4;
                for(int rowidxx = 0 ; rowidxx < totalrows ; rowidxx++)
                {
                    for(int coll = 0 ; coll < 4 ; coll++)
                    {
                        float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                        for(int cc = 0 ; cc < 2 ; cc++)
                        {
                            for(int i = tid ; i < 256 ; i += blockDim.x)
                                { 
                                    int r = i / 16;
                                    int c = i % 16;

                                    smenA[r * 16 + c] = qshared[rowidxx * 32 * 16 + cc * 16 + r * 32 + c];
                                }
                            
                            for(int i = tid ; i < 128 ; i += blockDim.x)
                                {
                                    int r = i / 8;
                                    int c = i % 8;
                                    
                                    smenB[r * 8 + c] = kshared[coll * 8 + cc * 16 * 32 + r * 32 + c];
                                }

                            __syncthreads();
                            const int col0 = (lane % 4) * 2;
                            const int col1 = col0 + 8;

                            uint32_t a_frag[4];
                            a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col0]);
                            a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                            a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col1]);
                            a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                            const int r0 = (lane % 4) * 2;
                            const int r1 = r0 + 8;

                            uint32_t b_frag[2];
                            b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));

                            b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                            __syncthreads();
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                "{%0,%1,%2,%3},"
                                "{%4,%5,%6,%7},"
                                "{%8,%9},"
                                "{%10,%11,%12,%13};"
                                    : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                                    : "r"(a_frag[0]), "r"(a_frag[1]),
                                    "r"(a_frag[2]), "r"(a_frag[3]),
                                    "r"(b_frag[0]), "r"(b_frag[1])
                                    ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                                );


                        }
                        const int r0 = group;
                        const int r1 = r0 + 8;
                        const int c0 = (lane % 4) * 2;
                        const int c1 = c0 + 1;


                        output[(rowidxx * 16 + r0) * 32 + coll * 8 + c0] += d1 * SCALE;
                        output[(rowidxx * 16 + r0) * 32 + coll * 8 + c1] += d2 * SCALE;
                        output[(rowidxx * 16 + r1) * 32 + coll * 8 + c0] += d3 * SCALE;
                        output[(rowidxx * 16 + r1) * 32 + coll * 8 + c1] += d4 * SCALE;

                    }
                }
            } // here one Q @ K.T 
            
        }
        //load V // we will load 32 * 256
        // it depends on rowid 

        // 8192 halfs / 8 per float4 = 1024 iterations
        for(int i = tid; i < 1024; i += blockDim.x)
        {
            *reinterpret_cast<float4*>(&vshared[i * 8]) =
            *reinterpret_cast<const float4*>(&Vptr[rowid * 32 * 256 + i * 8]);
        }
        __syncthreads();

        // start actual softmax
        // output is  64 * 32
        if(tid < 64)
        {
            float m_tile = -FLT_MAX;
            float l_tile = 0.f;
            
            for(int c = 0; c < 32; c++)
                m_tile = fmaxf(m_tile, output[tid * 32 + c]);

            for(int c = 0; c < 32; c++)
            {
                output[tid * 32 + c] = expf(output[tid * 32 + c] - m_tile);
                l_tile += output[tid * 32 + c];
            }


            float m_old = m[tid];
            float m_new = fmaxf(m_old, m_tile);
            alpha_s[tid] = expf(m_old - m_new);
            beta_s[tid]  = expf(m_tile - m_new);

            l[tid] = alpha_s[tid] * l[tid] + beta_s[tid] * l_tile;
            m[tid] = m_new;
         
        }
        __syncthreads();


        /*
        .... do scores * V here
        */
        const int itr = 256 / 8;    // = 32;
        const int rowitr = 64 / 16; // = 4;

        for(int iter = 0 ; iter < rowitr ; iter++)
        {
            for(int colitr = 0 ; colitr < itr ; colitr++)
            {
                float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;
                
                for(int cc = 0 ; cc < 2 ; cc++)
                {
                    for(int i = tid ; i < 256 ; i += blockDim.x)
                    {
                        int r = i / 16;
                        int c = i % 16;

                        smenA[r * 16 + c] = __float2half_rn(output[iter * 16 * 32 + cc * 16 + r * 32 + c]);
                    }

                    for(int i = tid ; i < 128 ; i += blockDim.x)
                    {
                        int r = i / 8;
                        int c = i % 8;

                        smenB[r * 8 + c] = vshared[(cc * 16 + r) * 256 + colitr * 8 + c];
                    }
                    __syncthreads();

                    // we got both smen perfectly
                    const int col0 = (lane % 4) * 2;
                    const int col1 = col0 + 8;

                    uint32_t a_frag[4];
                    a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col0]);
                    a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                    a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col1]);
                    a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                    const int r0 = (lane % 4) * 2;
                    const int r1 = r0 + 8;

                    uint32_t b_frag[2];
                    b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));

                    b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                    __syncthreads();
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},"
                        "{%4,%5,%6,%7},"
                        "{%8,%9},"
                        "{%10,%11,%12,%13};"
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]),
                            "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1])
                            ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                        );

                }

                    const int roww = rowtileID * 64 * 256 + iter * 16 * 256;
                    const int colbase = colitr * 8;


                    const int rr0 = group;
                    const int rr1 = rr0 + 8;
                    const int cc0 = (lane % 4) * 2;
                    const int cc1 = cc0 + 1;

                    int actual_row0 = iter * 16 + rr0;  
                    int actual_row1 = iter * 16 + rr1;  

                    float aa = __half2float(optr[roww + colbase + rr0 * 256 + cc0]);
                    float ab = __half2float(optr[roww + colbase + rr0 * 256 + cc1]);
                    float ac = __half2float(optr[roww + colbase + rr1 * 256 + cc0]);
                    float ad = __half2float(optr[roww + colbase + rr1 * 256 + cc1]);

                    aa = alpha_s[actual_row0] * aa + beta_s[actual_row0] * d1;
                    ab = alpha_s[actual_row0] * ab + beta_s[actual_row0] * d2;
                    ac = alpha_s[actual_row1] * ac + beta_s[actual_row1] * d3;
                    ad = alpha_s[actual_row1] * ad + beta_s[actual_row1] * d4;

                    optr[roww + colbase + rr0 * 256 + cc0] = __float2half(aa);
                    optr[roww + colbase + rr0 * 256 + cc1] = __float2half(ab);
                    optr[roww + colbase + rr1 * 256 + cc0] = __float2half(ac);
                    optr[roww + colbase + rr1 * 256 + cc1] = __float2half(ad);

                // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                //     threadIdx.x < 12 && iter == 0 && colitr == 0)
                // {
                //     int idx00 = roww + colbase + rr0 * 256 + cc0;
                //     int idx01 = roww + colbase + rr0 * 256 + cc1;
                //     int idx10 = roww + colbase + rr1 * 256 + cc0;
                //     int idx11 = roww + colbase + rr1 * 256 + cc1;

                //     printf("tid=%d -> (%d,%d,%d,%d)\n",
                //         threadIdx.x, idx00, idx01, idx10, idx11);
                // }

            }
        }
        //
    }
    // actual last scaling here
    
    for(int idx = tid; idx < 64 * 256; idx += blockDim.x)
    {
        int r = idx / 256;
        int c = idx % 256;
        float linv = __fdividef(1.0f, l[r]);
        int global_idx = rowtileID * 64 * 256 + r * 256 + c;
        optr[global_idx] = __float2half(__half2float(optr[global_idx]) * linv);
    }

}
