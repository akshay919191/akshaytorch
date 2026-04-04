#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SEQ_LEN 2048
#define D_MODEL 2048
#define NUM_HEADS 16
#define HEAD_DIM 64


#define Br 32 // number of rows in query
#define Bc 32 // number of rows in key and value

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define SCALE 1.0f / sqrtf((float)HEAD_DIM)

__global__ void FlashAttention(const float* __restrict__ query,
                               const float* __restrict__ key,
                               const float* __restrict__ value,
                               float*       __restrict__ output,
                               float*       __restrict__ log,
                               int                       N,
                               int                       d)
    {
        // id's for fucking data
        const int batchid = blockIdx.x ;
        const int headid  = blockIdx.y ;
        const int tileid  = blockIdx.z ;
        const int tid     = threadIdx.x;

        // making the fucking base to get to the point where we will be 
        const long long base = (long long)batchid * NUM_HEADS * N * d // jump to correct batch 
                          + (long long)headid * N * d ;// jump to correct head 

        // point the the fucking location of data on which we are performing calculations
        const float* Qptr = query  + base;
        const float* Kptr = key    + base;
        const float* Vptr = value  + base;
        float*     outptr = output + base;

        // now we at the fucking location now we can start , and stop u fuckin nigga , how the fuck you can start the main motive of flash attn was fast calc for which you'll need SRAM , for which u mf didn't allocated space yet , gop make it nigga
        // we can do same as fused attn

        extern __shared__ float smen[];
        float* qshared   = smen  ;
        float* kshared   = qshared + Br * HEAD_DIM ;// we used fucking this number because we are loading Br number of columns in one time for HEAD_DIM times means fucking row you fucking cumass , if you are thinking why we write it here go get fucked yourself , its ptr , so it will start from smen + Br * headdim , so this - smen is qshared size you mf because ptr points towards the first point
        float* vshared   = kshared + Bc * HEAD_DIM ;
        float* outshared = vshared + Bc * HEAD_DIM ;


        const int q_row = tileid * Br + tid;
        const bool validROW = (q_row < N);  

        // now you are pointed toward the data , have fasted space available on your potato system , go start you bitch
        
        // here is one problem we don't know the number of blocks we need to load for Key and value according to paper : https://arxiv.org/pdf/2205.14135
        // we need Tc

        const int Tc = (N + Bc - 1) / Bc; // number of kv tiles

        // according to fucking paper i mentioned they told to load the full row iteratively
        // you can use fucking tiling here , i will make it after make this version work
        if(validROW){
            for(int i = 0  ; i < d ; i++) qshared[tid * d + i] = Qptr[q_row * d + i];
        } else{
            for(int i = 0  ; i < d ; i++) qshared[tid * d + i] = 0.0f;
        }

        // loaded the fucking query one row

        // if you fucking know we are using online softmax , we need to update max and sum
        float m_i = -FLT_MAX;
        float l_i = 0.0f;
        float o[HEAD_DIM] = {0}; // i fucking don't know this if i found i will fucking write it ,. now i knwo you mf , because the output at last is one row into one col which is head_dim size 

        // now load the fucking K and V here
        for(int kv_tile = 0 ; kv_tile < Tc ; kv_tile++){
            // each tile have dim of Bc * d for both key and value

            // we can't do the same as qshared , why? becuase we didn't declared a var like q_row so we need to loop through all tid and then all tid will loop through each element
            // be careful here don't do j += Bc , becuase our query tile is size of Br so we need ot jump for same size
            for(int rowID = tid ; rowID < Bc ; rowID += Br){
                int cur_IDX = rowID + kv_tile * Bc; //  we are seeing on which row + which tile on row
                //(cur_IDX < N){ // full size N , divided by size Bc in to Kv number of tiles so 
                for(int iter = 0 ; iter < d ; iter++){
                    if(cur_IDX < N) kshared[rowID * d + iter] = Kptr[cur_IDX * d + iter];
                    else kshared[rowID * d + iter] = 0.0f;
                }
            }

            // now do for Value
            for(int rowID = tid ; rowID < Bc ; rowID += Br){
                int cur_IDX = rowID + kv_tile * Bc;
                for(int iter = 0 ; iter < d ; iter++){
                    if(cur_IDX < N) vshared[rowID * d + iter] = Vptr[cur_IDX * d + iter];
                    else vshared[rowID * d + iter] = 0.0f;
                }
            }

            // now sync it so all threads are at same level
            __syncthreads();

            // now the dotproduct 
            // see because in attn its obvious that all dims are same so the loaded dims are same even for shared memory
            // so we gonna iterate over d terms because we loaded that much and sum it up and load it
            // be careful we will load q for one time multiply it by all K loaded and sum each time and save it

            for(int rowID = 0 ; rowID < Bc ; rowID++)
            {
                float dot = 0.0f;
                for(int iter = 0 ; iter < d ; iter++)
                    dot += qshared[tid * d + iter] * kshared[rowID * d + iter];
                // now we gonna store in outshared for upcoming mess
                int rowKV = kv_tile * Bc + rowID;
                outshared[tid * Bc + rowID] = (rowKV < N) ? dot * SCALE : -FLT_MAX;
            }

            // now we are going for softmax so we will need a var to store max 
            float m_tile = -FLT_MAX;
            // shape of final is Br * Bc
            for(int colID = 0 ; colID < Bc ; colID++)
                m_tile = fmaxf(m_tile , outshared[tid * Bc + colID]);

            // per row 
            float l_tile = 0.0f;
            for(int colID = 0 ; colID < Bc ; colID++)
                {
                    outshared[tid * Bc + colID] = expf(m_tile - outshared[tid * Bc + colID]);
                    l_tile += outshared[tid * Bc + colID];
                }

            float m_new = fmaxf(m_tile , m_i);
            float l_new = expf(m_i - m_new) * l_i + expf(m_tile - m_new) * l_tile;

            float alpha = expf(m_i - m_new) , beta = expf(m_tile - m_new);

            for(int col = 0 ; col < d ; col++)
            {
                float pv = 0.0f;
                for(int colID = 0 ; colID < Bc ; colID++)
                    pv += outshared[tid * Bc + colID] * vshared[colID * d + col]; // indexing might look unfamiliar but outshared is size of Br * Bc so col is bounded to Bc and we are loading col in vshared so
                o[col] = alpha * o[col] + beta * pv;
            }

            m_i = m_new;
            l_i = l_new;
            __syncthreads();

        }

        if(validROW){
            float l_inv = 1.0f / (float)l_i;
            for(int col = 0 ; col < d ; col++)
                outptr[q_row * d + col] = o[col] * l_inv;
        }   

    }

