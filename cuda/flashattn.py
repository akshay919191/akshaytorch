#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// N = seq len , num_head , d_model , d = d_model / num_head

#define SEQ_LEN 2048
#define D_MODEL 2048
#define NUM_HEADS 16
#define HEAD_DIM 64

// M size on RTX 3050 is 64kb means 64000 we will go with 48kb just to not hit limit wihtout mistake
// bc M / 4d => 48000 / 4 * 64 -> 192 , but as we need to divide it further we will take lesser but power of 2 which is 128

// br is min of d and bc

#define Bc 32
#define Br 32
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
// divide query into Tr(8) (N / Br) number of blocks size Br * d 
// same for key and value but by Bc Tc(4) (N / Bc) number of blocks of size Bc * d

// via coding till 100 lines i realised getting what is br , bc , tr , tc will actually help a lot 

/*

here is the breakdown for them:
br = is rows of tile query
bc = is cols of tile key and val

tr = threads for each row in tile query
tc = threads for each col in tile key and value

*/

#define SCALE (1.0f / sqrtf((float)HEAD_DIM))

__global__ void FlashAttention(const float* __restrict__ query,
        const float* __restrict__ key, // makin restrict for making it more efficient and optimized
        const float* __restrict__ value,
        float* __restrict__ output,
        float* __restrict__ log,
        int N , int d)

    {
        // as working on we need atleast 3 Dim to just follow up
        const int batchid = blockIdx.x; // tells to be on which batch
        const int headid = blockIdx.y; // tells on which head group
        const int tileid = blockIdx.z; // tells on which tile in head
        const int tid = threadIdx.x; // thread id

        const int Tc = (N + Bc - 1) / Bc;
        
        const int q_row = tileid * Br + tid; // sequence position in row (note this only works on tile not actual query)
        const bool validROW = (q_row < N);

        // to make above positioning work we have to index , as we know GPU read data in contiguous memory , so we have to write strides for it
        const long long stride_bh = (long long)N * d * NUM_HEADS; // its a batch jump stride(batch stride)
        const long long base = (long long)batchid * (NUM_HEADS * N * d)
                     + (long long)headid  * (N * d); // this makes jump to correct head (not tile yet)

        // now we have pointed to head dim now to TILE
        // reminder whenever we make a pointer to a array or a 2D or 3D or any dim it just points towards the first digit means 0 batch , 0 head dim , 0 tile , 0 tid
        const float* Qptr = query + base; // this confirm that each thread we launch get its own and do not overlap , 
        // same for others 
        const float* Kptr = key + base;
        const float* Vptr = value + base;
        float*       Optr = output + base;


        float* Lptr = log + (long long)batchid * NUM_HEADS * N
                  + (long long)headid  * N; // have to take a look here

        // now SRAM memory allocation
        // we know from above query is loaded in Br and K n V are loadin in Bc
        extern __shared__ float smem[];
        float* qshared   = smem;
        float* kshared   = qshared + Br * HEAD_DIM;
        float* vshared   = kshared + Bc * HEAD_DIM;
        float* outshared = vshared + Bc * HEAD_DIM;


        // now we will load (listen this is not tiling so we will load once a element in to SRAM)
        if(validROW){
            for(int j = 0 ; j < d ; j++)
                qshared[tid * d + j] = Qptr[q_row * d + j]; // here each threads is loading row in to SRAM(head dim)
        } else {
            for(int j = 0 ; j < d ; j++)
                qshared[tid * d + j] = 0.0f;
        } 
        __syncthreads();
        // why didn't we added syncthreads here i know it will overwrite it but then what do rest of threads will do , free?

        // now we have loaded a row (whole block(br * HEAD_DIM)) now we need acc to calculate max , sum
        float m_i = -FLT_MAX; // each row max
        float l_i = 0.0f; // each row sum
        float out[HEAD_DIM] = {0} ; // output for one time

        // bow to this part , fuck this part , this is the real mf , now we have to loop through K and  V track sum , max and do softmax
    
        //  as we have loaded the Q now we are looping through KV 
        for(int i = 0 ; i < Tc ; i++){
            // load K
            // what we will do here , load a row -> save in to shared memory
            for(int row = tid ; row < Bc ; row += Br){ //  its a tile means , there are Br number of threads in parallel
                int index = i * Bc + row; // row index
                if(index < N){
                    for(int j = 0 ; j < d ; j++)
                        kshared[row * d + j] = Kptr[index * d + j];
                }else{
                    for(int j = 0 ; j < d ; j++) kshared[row * d + j] = 0.0f;
                }
            }

            // same for V

            for(int row = tid ; row < Bc ; row += Br){ 
                int index = i * Bc + row; // row index
                if(index < N){
                    for(int j = 0 ; j < d ; j++)
                        vshared[row * d + j] = Vptr[index * d + j];
                }else{
                    for(int j = 0 ; j < d ; j++) vshared[row * d + j] = 0.0f;
                }
            }

            __syncthreads();

            // dot product 
            // now we have to mult in col format and reduce it or just add it 
            for(int tile = 0 ; tile < Bc ; tile ++){
                float dot = 0.0f;
                for (int j = 0 ; j < d ; j++)
                    dot += qshared[tid * d + j] * kshared[tile * d + j];
                int kvrow = i * Bc + tile;
                outshared[tid * Bc + tile] = (kvrow < N) ? dot * SCALE : -FLT_MAX;
            }

            // now online softmax 
            // first find max
            float m_tilde = -FLT_MAX;
            for(int j = 0 ; j < Bc ; j++)
                m_tilde = fmaxf(m_tilde , outshared[tid * Bc + j]);

            float l_tilde = 0.0f;
            for(int j = 0 ; j < Bc ; j++)
            {
                outshared[tid * Bc + j] = expf(outshared[tid * Bc + j] - m_tilde);
                l_tilde += outshared[tid * Bc + j];
            }

            // don't stress over it ,its online softmax part(read it lateR)
            float m_new = fmaxf(m_i , m_tilde);
            float l_new = expf(m_i - m_new) * l_i + expf(m_tilde - m_new) * l_tilde;

            float alpha = expf(m_i - m_new);
            float beta = expf(m_tilde - m_new);

            for(int j = 0 ; j < d ; j++){
                float pv = 0.0f;
                for (int col = 0 ; col < Bc ; col++)
                    pv += outshared[tid * Bc + col] * vshared[col * d + j];
                out[j] = alpha * out[j] + beta * pv;
            }

            m_i = m_new;
            l_i = l_new;
            __syncthreads();

        }

        if(validROW){
            float l_inv = 1.0f / l_i;
            for(int j = 0 ; j < d ; j++)
                Optr[q_row * d + j] = out[j] * l_inv;

            Lptr[q_row] = m_i + logf(l_i);
        }
}

void launch_flash_attn(
    const float* d_Q,   // device pointers [B, H, N, d]
    const float* d_K,
    const float* d_V,
    float*       d_O,
    float*       d_L,   // [B, H, N]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
) {
    const int Tr = (seq_len + Br - 1) / Br;
    dim3 grid(batch_size, num_heads, Tr);
    dim3 block(Br);

    size_t smem = (size_t)(Br + Bc + Bc) * HEAD_DIM * sizeof(float)
                + (size_t) Br * Bc * sizeof(float);



    FlashAttention<<<grid, block, smem, stream>>>(d_Q, d_K, d_V, d_O, d_L,seq_len,head_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("LAUNCH FAILED: %s\n", cudaGetErrorString(err));
}

int main() {
    const int B = 4, H = NUM_HEADS, N = SEQ_LEN, d = HEAD_DIM;
    const long long elems = (long long)B * H * N * d;

    float* h_Q = (float*)malloc(elems * sizeof(float));
    float* h_K = (float*)malloc(elems * sizeof(float));
    float* h_V = (float*)malloc(elems * sizeof(float));
    float* h_O = (float*)malloc(elems * sizeof(float));
    float* h_L = (float*)malloc((long long)B * H * N * sizeof(float));

    srand(42);
    for (long long i = 0; i < elems; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    CHECK_CUDA(cudaMalloc(&d_Q, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_L, (long long)B * H * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, elems * sizeof(float), cudaMemcpyHostToDevice));


    for (int i = 0; i < 3; i++)
        launch_flash_attn(d_Q, d_K, d_V, d_O, d_L, B, H, N, d);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("maxGridSize = (%d, %d, %d)\n",
        prop.maxGridSize[0],
        prop.maxGridSize[1],
        prop.maxGridSize[2]);

    const int RUNS = 100;
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        launch_flash_attn(d_Q, d_K, d_V, d_O, d_L, B, H, N, d);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));   // ← wait inside the window

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, t0, t1));
    float avg_ms = total_ms / RUNS;


    double flops     = 4.0 * H * N * N * d;
    double tflops    = (flops / (avg_ms * 1e-3)) / 1e12;

    CHECK_CUDA(cudaMemcpy(h_O, d_O, elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_L, d_L, (long long)B * H * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (long long i = 0; i < elems; i++) {
        if (!isfinite(h_O[i])) { ok = false; break; }
    }
    printf("  Output check    : %s\n", ok ? "PASSED ✓" : "FAILED ✗");
    printf("─────────────────────────────────────────\n");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_L);
    return ok ? 0 : 1;
}
