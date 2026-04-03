#include <iostream>
#include <cuda_runtime.h>
#define TILE 32

__global__ void add(float* a , float* b , float* c , int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x; // assigning threads kind of (giving each them specific id)
    if (i < n){ // confirming if number of threads 
        c[i] = a[i] + b[i];
    }
}

int main(){
    int n = 1000;
    size_t size = n * sizeof(float); // Total number of bytes needed for n floats

    // Allocate host (CPU) memory
    float *h_a = new float[n]; //first number 
    float *h_b = new float[n]; // second number
    float *h_c = new float[n]; // answer

    // Initialize input data
    for(int i = 0 ; i < n ; ++i){ 
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    float *d_a , *d_b , *d_c ;// device memory(real gpu memory)
    cudaMalloc(&d_a , size);
    cudaMalloc(&d_b , size);
    cudaMalloc(&d_c , size);

    //move to gpu(from cpu we defined)
    cudaMemcpy(d_a , h_a , size ,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , h_b , size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_c , h_c , size , cudaMemcpyHostToDevice);

    int threadperblock = 256;
    int blockpergrid = (n + threadperblock - 1) / threadperblock;

    add<<<blockpergrid , threadperblock>>>(d_a , d_b , d_c , n); // computing on GPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // copy result back


    for(int i = 0 ; i < 20 ; i++){
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}



__global__ void matADD(int* A , int* B , int* C , int M , int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < M && col < N){
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

int main(){

    int M = 4;       // number of rows
    int N = 5;  
    int n = M * N ;   // number of columns
    size_t size = n * sizeof(int);

    int* h_a = new int[n];
    int* h_b = new int[n];
    int* h_c = new int[n];

    for (int i = 0; i < M; ++i) {      
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;      
            h_a[idx] = i * 1 + j * 2; 
            h_b[idx] = i * 3 + j * 4;
        }
    }

    
    int *a , *b , *c;
    cudaMalloc(&a , size);
    cudaMalloc(&b , size);
    cudaMalloc(&c , size);

    cudaMemcpy(a , h_a , size , cudaMemcpyHostToDevice);
    cudaMemcpy(b , h_b , size , cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N + 15)/16, (M + 15)/16);

    matADD<<<blocksPerGrid , threadsPerBlock>>>(a , b , c , M , N);
    cudaMemcpy(h_c , c , size , cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            printf("%d ", h_a[idx]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            printf("%d ", h_b[idx]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            printf("%d ", h_c[idx]);
        }
        printf("\n");
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}


__global__ void ReLU(float* a , float* b , int M , int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N){
        if (a[row * N + col] > 0){
        b[row * N + col] = a[row * N + col];
        }else{
            b[row * N + col] = 0;
        }
    }
}


int main(){
    int M = 3;
    int N = 4;

    int n = M * N;
    size_t size = n * sizeof(float);

    float* h_a = new float[n];
    float* h_b = new float[n];

    float h_input[3][4] = {
        { -1.5f,  0.0f,  2.3f, -4.1f },
        {  3.2f, -0.7f, -8.0f,  5.5f },
        { -2.2f,  9.1f,  0.0f, -6.3f }
    };

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i * N + j] = h_input[i][j];
        }
    }

    float *a , *b;
    cudaMalloc(&a , size);
    cudaMalloc(&b , size);

    cudaMemcpy(a , h_a , size , cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N + 15) / 16 , (M + 15) / 16);

    ReLU<<<blocksPerGrid , threadsPerBlock>>>(a , b , M , N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_b , b , size , cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < M ; i++){
        for(int j = 0 ; j < N ; j++){
            printf("%6f " , h_a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    for(int i = 0 ; i < M ; i++){
        for(int j = 0 ; j < N ; j++){
            printf("%6f " , h_b[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    delete[] h_a;
    delete[] h_b;

    cudaFree(a);
    cudaFree(b);

    return 0;

}

__global__ void GeLU(float *a , float * b , int M , int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < M && col < N){
        float term = a[row * N + col];
        b[row * N + col] = 0.5f * term * (1.0f + tanhf(0.79788456f * (term + 0.044715f * term * term * term)));
    }
}

int main(){
    int M = 3;
    int N = 4;

    int n = M * N;
    size_t size = n * sizeof(float);

    float* h_a = new float[n];
    float* h_b = new float[n];

    float h_input[3][4] = {
        { -1.5f,  0.0f,  2.3f, -4.1f },
        {  3.2f, -0.7f, -8.0f,  5.5f },
        { -2.2f,  9.1f,  0.0f, -6.3f }
    };

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i * N + j] = h_input[i][j];
        }
    }

    float *a , *b;
    cudaMalloc(&a , size);
    cudaMalloc(&b , size);

    cudaMemcpy(a , h_a , size , cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16 , 16);
    dim3 blocksPerGrid((N + 15) / 16 , (M + 15) / 16);

    GeLU<<<blocksPerGrid , threadsPerBlock>>>(a , b , M , N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("erorr is %s: " , cudaGetErrorString(err));
    }

    cudaMemcpy(h_b , b , size , cudaMemcpyDeviceToHost);


    for(int i = 0 ; i < M ; i++){
        for(int j = 0 ; j < N ; j++){
            printf("%6f " , h_a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    for(int i = 0 ; i < M ; i++){
        for(int j = 0 ; j < N ; j++){
            printf("%6f " , h_b[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    delete[] h_a;
    delete[] h_b;

    cudaFree(a);
    cudaFree(b);

    return 0;

}


__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int rowStride = blockDim.y * gridDim.y;
    int colStride = blockDim.x * gridDim.x;

    for (int i = row; i < M; i += rowStride) {
        for (int j = col; j < N; j += colStride) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 3;
    int N = 4;
    int K = 4; 

    float h_a[3 * 4] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    float h_b[4 * 4] = {
        1, 0, 2, 1,
        0, 1, 0, 2,
        1, 1, 1, 0,
        2, 0, 1, 1
    };
    float h_c[3 * 4] = {0};

    float *a, *b, *c;
    cudaMalloc(&a, M * K * sizeof(float));
    cudaMalloc(&b, K * N * sizeof(float));
    cudaMalloc(&c, M * N * sizeof(float));

    cudaMemcpy(a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, M, N, K);

    cudaDeviceSynchronize();
    cudaMemcpy(h_c, c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

   
    printf("A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) printf("%6.1f ", h_a[i * K + j]);
        printf("\n");
    }

   
    printf("\nB:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) printf("%6.1f ", h_b[i * N + j]);
        printf("\n");
    }

    
    printf("\nC = A x B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%6.1f ", h_c[i * N + j]);
        printf("\n");
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

__global__ void softmax(float* A, float* B, int M, int N){
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M){
        float sum = 0.0f;

        for (int c = 0; c < N; c++){
            sum += expf(A[r * N + c]);
        }

        for (int c = 0; c < N; c++){
            B[r * N + c] = expf(A[r * N + c]) / sum;
        }
    }
}

int main(){
    int M = 3;
    int N = 4;

    int n = M * N;
    size_t size = n * sizeof(float);

    float h_a[3 * 4] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    float h_b[3 * 4] = {0};

    float *a , *b;
    cudaMalloc(&a , size);
    cudaMalloc(&b , size);

    cudaMemcpy(a , h_a , size , cudaMemcpyHostToDevice);

    int threadsPerBlock = 8;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    softmax<<<blocksPerGrid , threadsPerBlock>>>(a , b , M , N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_b , b , size , cudaMemcpyDeviceToHost);

    printf("A:\n");
    for (int i = 0; i < M * N; i++) {
        printf("%6f ", h_b[i]);
        
    }
    printf("\n");

    

    cudaFree(a);
    cudaFree(b);

    return 0;

}

__global__ void reduceSum(float* A, float* out, int N) {
    __shared__ float tile[256];
    int tid = threadIdx.x;

    tile[tid] = (tid < N) ? A[tid] : 0.0f;

    __syncthreads();

   
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tile[tid] += tile[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[0] = tile[0];
}

int main() {
    int N = 8;
    size_t size = N * sizeof(float);


    float h_A[8] = {1,2,3,4,5,6,7,8};
    float h_out = 0.0f;

    float *d_A, *d_out;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    
    reduceSum<<<1, 32>>>(d_A, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum = " << h_out << std::endl;

    cudaFree(d_A);
    cudaFree(d_out);

    return 0;
}



__global__ void tiledmatmul(float* A, float* B, float* out, int M, int N, int K) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    float sum = 0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row -->> tile size into block id + ty to points towards the first element of the tile, 
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    for(int i = 0 ; i < (K + TILE - 1) / TILE ; i++){ // why we used K here? -> K is the connecting dim of both matrix so it takes care of bounds
        // now we need to load a tile from A and from B
        // we have to take care of bounds because we make tiles the power of 2 but dims are not always so 
        if(row < M && (TILE * i + tx) < K) // row should be less than M , because we load according to it , and the indicing we will use should be in bound too
            // why we didn't bound tx and ty here? because these are threads ID we will launch kernel making sure it make sense
            tileA[ty][tx] = A[row * K + (TILE * i + tx)]; // A[row][K] we have to load this , row can be any in tile like we can go from 0th - 15th row so it depends but to get to next row as row is only indicing pointers towards it , we need to tell it should jump this much so row * K and in each row we are bound to tile size may be it can be more than the tile size , so we will multiply TILE * i + tx (as we are loading columns in row) so this makes sure we iterate over each but this can overflow so we need to bound this too
        else //  else we want it to be zero (you'll be wondering doesn't making it zero increase matrix size?? no it won't why? because for now we are not storing , and we are thinking in memory and check our conditioning in IF , so if it doesn't satisfy it , we are making it zero because at the end we are doing sum so it doesn't count at all)
            tileA[ty][tx] = 0.0;
            
        // now same for tile B
        if((TILE * i + ty) < K && col < N) // here indexing might be different , here we are doing B[k][col] notice on thing the stride for column is 1 (stride is the steps needed to jump over the same category , in row we have to jump over all columns so) so col will be as it is , what about k ? so we are jumping to next row in next iterat , so (TILE * i + ty) * N , why this because first we want to know which row number we are wanting to laod then we can just give stride to jump over it 

            tileB[ty][tx] = B[(TILE * i + ty) * N + col];
        else    
            tileB[ty][tx] = 0.0;

        // NOW before moving we want to load all tiles;so
        __syncthreads();

        for (int k = 0 ; k < TILE ; k++){
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    } 

    if(row < M && col < N)
        out[row * N + col] = sum;

}


int main() {
    int M = 64, K = 64, N = 64;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);


    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    
    for (int i = 0; i < M*K; i++) h_A[i] = i % 5;
    for (int i = 0; i < K*N; i++) h_B[i] = i % 3;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);    

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);

    tiledmatmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Result matrix (partial):\n";
    for (int i = 0; i < std::min(M, 5); i++) {
        for (int j = 0; j < std::min(N, 5); j++) {
            std::cout << h_C[i*N + j] << " ";
        }
        std::cout << "\n";
    }

    bool correct = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += h_A[i*K + k] * h_B[k*N + j];
            if (fabs(sum - h_C[i*N + j]) > 1e-3) {
                std::cout << "Mismatch at " << i << "," << j << ": GPU=" 
                          << h_C[i*N + j] << " CPU=" << sum << "\n";
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) std::cout << "Output is CORRECT!\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}


__global__ void tiledsoftmax(float* A , float* B , int M , int N){
    __shared__ float tileA[TILE];
    __shared__ float globalMAX;
    __shared__ float sum;

    int tx = threadIdx.x ; // its the thread id , thread id is max the tile size means 16
    
    /* now we need row indexes which are quite simple to find ,(disclaimer , row only points towards the first element)*/
    int row = blockIdx.x;  

    // we need a global max for each row
    if (tx == 0) {
        globalMAX = -FLT_MAX;    
        sum       = 0.0f;
    }
    __syncthreads(); 
    if(row >= M) return; //  safe exit 

    // now we need to iterate through all the tile for max and write in tileA (it can be overwritten so we can use it multiple time)

    for(int i = 0 ; i < (N + TILE - 1) / TILE ; i++){ // we got max for 1 block 
        int col = i * TILE + tx;
        float val = (col < N) ? A[row * N + col] : -FLT_MAX;
        tileA[threadIdx.x] = val;
        __syncthreads(); // 

        for(int stride = TILE / 2 ; stride > 0 ; stride /= 2){
            if (tx < stride)
                tileA[tx] = fmaxf(tileA[tx] , tileA[tx + stride]);

            __syncthreads();
        } // now its reduced to all over tileA[0] the max one 

        if(threadIdx.x == 0) // you can choose 0 or 0 - 15 any because all threads see value due to shared memory so to save compute we are using one thread to write into shared memeory instead of all threads do the same work overwriting for no reason
            globalMAX = max(globalMAX , tileA[0]); // now we can grab the max one 
        __syncthreads();
    }

    // now we will load each tile from a row , subtract the max , exponentialize each term , sum up that again do the fucking things then we are done
    

    for (int tileIdx = 0; tileIdx < (N + TILE - 1) / TILE; tileIdx++) {
        int col = tileIdx * TILE + tx;
        float val = (col < N) ? expf(A[row * N + col] - globalMAX) : 0.0f;
        tileA[tx] = val;
        __syncthreads();

        // parallel reduction to sum
        for (int stride = TILE / 2; stride > 0; stride /= 2) {
            if (tx < stride)
                tileA[tx] += tileA[tx + stride];
            __syncthreads();
        }

        if (tx == 0)
            sum += tileA[0];
        __syncthreads();
    }

    for (int i = 0 ; i < (N + TILE - 1) / TILE ; i++){
        int col = i * TILE + tx;
        if (col < N)
            B[row * N + col] = expf(A[row * N + col] - globalMAX) / sum;
    }

}

int main(){
    int M = 64 , N = 64;
    int n = M * N;

    size_t size = n * sizeof(float);
    float* h_a = (float*)malloc(size); // its c style works for cpp too , but u can also go for new float[n]
    float* h_b = (float*)malloc(size);

    for(int i = 0 ; i < n ; i++) h_a[i] = 2.0f;

    float *a , *b;
    cudaMalloc(&a , size);
    cudaMalloc(&b , size);

    cudaMemcpy(a , h_a , size , cudaMemcpyHostToDevice);

    int threadsPerBlock = TILE;
    int blocksPerGrid = M;

    tiledsoftmax<<<blocksPerGrid , threadsPerBlock>>>(a , b , M , N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_b , b , size , cudaMemcpyDeviceToHost);

    std::cout << "Result matrix (partial):\n";
    for (int i = 0; i < std::min(M, 5); i++) {
        for (int j = 0; j < std::min(N, 5); j++) {
            std::cout << h_b[i*N + j] << " ";
        }
        std::cout << "\n";
    }

    free(h_a);
    free(h_b);
    cudaFree(a);
    cudaFree(b);
}


__global__ void transpose(float *A , float *B , int M  , int N){
    __shared__ float tile[TILE][TILE + 1];

    int row = threadIdx.y + blockIdx.y * TILE;
    int col = threadIdx.x + blockIdx.x * TILE;

    if (row < M && col < N)
        tile[threadIdx.y][threadIdx.x] = A[row * N + col];
    
    __syncthreads();

    int outrow = threadIdx.y + blockIdx.x * TILE;
    int outcol = threadIdx.x + blockIdx.y * TILE;
    
    if (outrow < N && outcol < M)
        B[outrow * M + outcol] = tile[threadIdx.x][threadIdx.y];

}
int main(){
    int M = 4096 , N = 3084;
    size_t size = M * N * sizeof(float); 
    
    float *h_a = new float[M * N];
    float *h_out = new float[M * N];

    for (int i = 0 ; i < M ; i++){
        for (int j = 0 ; j < N ; j++){
            h_a[i * N + j] = i * 3.0f + j;
        }
    }

    float *a , *b;
    cudaMalloc(&a , size);
    cudaMalloc(&b , size);

    cudaMemcpy(a , h_a , size , cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N + 15) / 16 , (M + 15) / 16);




   
    transpose<<<blocksPerGrid, threadsPerBlock>>>(a, b, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out , b , size , cudaMemcpyDeviceToHost);

    // for (int i = 0 ; i < 4 ; i++){
    //     for (int j = 0 ; j < 3 ; j++){
    //         std::cout << h_a[i * N + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";

    // for (int i = 0; i < N; i++) {        // rows = N
    //     for (int j = 0; j < M; j++) {    // cols = M
    //         std::cout << h_out[i * M + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    delete[] h_a;
    delete[] h_out;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(a);
    cudaFree(b);
}


#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

#define TILE 32

// we will be doin (softmax(Query @ key.T)) @ value

__global__ void fusedattn(const float* __restrict__ Query 
    , const float* __restrict__ KEY 
    , const float* __restrict__ Value 
    , float* __restrict__ weight , int _seq , int _d_model){

    // we need to multiply these 2 without doin transpose 
    int tx = threadIdx.x;
    int row = blockIdx.x;

    // making free space like __shared 
    extern __shared__ float smen[];
    float* scores = smen;
    float* tileQ = smen + _seq;
    float* tileK = tileQ + TILE;

    // initialize the scores to be 0
    for (int i = tx ; i < _seq ; i += TILE) //  its using all threads to make zero you can go in one go too
        scores[i] = 0.0f;
    __syncthreads();

    // dot product
    for (int t = 0 ; t < (TILE + _d_model - 1) / TILE ; t++){
        int d = TILE * t + tx; // on which col(it covers whole row due to t)

        tileQ[tx] = (row < _seq && d < _d_model) ? Query[d + row * _d_model] : 0.0f;
        __syncthreads();

        for(int i = 0 ; i < _seq ; i++){
            float kval = (d < _d_model) ? KEY[i * _d_model + d] : 0.0f;
            tileK[tx] = tileQ[tx] * kval;
            __syncthreads();

            for(int stride = TILE / 2 ; stride > 0 ; stride /= 2){
                if(tx < stride) tileK[tx] += tileK[tx + stride];
                __syncthreads();
            }

            if(tx == 0) scores[i] += tileK[0]; //  per element indexing
            __syncthreads();
        }
    }   

    // scaling 
    float scale = 1.0f / sqrtf((float)_d_model);
    for(int j = tx ; j < _seq ; j += TILE)
        scores[j] *= scale;
    __syncthreads();


    __shared__ float smax;
    if(tx == 0) smax = -FLT_MAX;
    __syncthreads(); 
    tileK[tx] = -FLT_MAX;
    for(int k = tx ; k < _seq ; k += TILE)
        tileK[tx] = fmaxf(tileK[tx] , scores[k]); //  if we have tile size of 8 amd we have 12 elements , we are making sure that the frist element on tileK is max of all TILE's nth elements means tileK[0] = max of its own and scores which have all 12 scores means scores of 0 and 8th elements so it makes sure it stays the same size despite calulating the max
    __syncthreads();

    for(int stride = TILE / 2 ; stride > 0 ; stride >>= 1)
       {
        if(tx < stride) tileK[tx] = fmaxf(tileK[tx] , tileK[tx + stride]);
        __syncthreads();
    }

    if(tx == 0) smax = tileK[0];

    // now same way , no extra loading do num - max;
    for (int j = tx ; j < _seq ; j += TILE)
        scores[j] = expf(scores[j] - smax);
    __syncthreads();

    __shared__ float ssum;
    if (tx == 0) ssum = 0.0f;
    __syncthreads();

    tileK[tx] = 0.0f;
    for (int j = tx; j < _seq; j += TILE) //  loading to tileK as we have fixed size so just adding all elements in to TILE size
        tileK[tx] += scores[j];
    __syncthreads();

    for (int stride = TILE / 2; stride > 0; stride >>= 1)
    {
        if (tx < stride) tileK[tx] += tileK[tx + stride];
        __syncthreads();
    }
    if (tx == 0) ssum = tileK[0];    
    __syncthreads();

    for (int j = tx; j < _seq; j += TILE)
        scores[j] /= ssum;
    __syncthreads();

    for(int d = tx ; d < _d_model ; d += TILE){
        float acc = 0;
        for(int j = 0 ; j < _seq ; j++)
            acc += scores[j] * Value[j * _d_model + d];
        weight[row * _d_model + d] = acc;
    }
}



void launch_fusedattn(const float* Q, const float* K, const float* V,
                      float* out, int seq, int d_model)
{
    size_t smem = (seq + 2 * TILE) * sizeof(float);
    fusedattn<<<seq, TILE, smem>>>(Q, K, V, out, seq, d_model);
    cudaDeviceSynchronize();
}

int main()
{
    int seq     = 3;    
    int d_model = 4;   

    float* h_Q   = new float[seq * d_model];
    float* h_K   = new float[seq * d_model];
    float* h_V   = new float[seq * d_model];
    float* h_out = new float[seq * d_model];

    for (int i = 0; i < seq * d_model; i++)
    {
        h_Q[i] = 0.1f * i;
        h_K[i] = 0.2f * i;
        h_V[i] = 0.3f * i;
    }

    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q,   seq * d_model * sizeof(float));
    cudaMalloc(&d_K,   seq * d_model * sizeof(float));
    cudaMalloc(&d_V,   seq * d_model * sizeof(float));
    cudaMalloc(&d_out, seq * d_model * sizeof(float));

    cudaMemcpy(d_Q, h_Q, seq * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, seq * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, seq * d_model * sizeof(float), cudaMemcpyHostToDevice);

    // run the kernel
    launch_fusedattn(d_Q, d_K, d_V, d_out, seq, d_model);

    cudaMemcpy(h_out, d_out, seq * d_model * sizeof(float), cudaMemcpyDeviceToHost);



    std::cout << "Output (each row is out[token][d_model]):" << std::endl;
    for (int i = 0; i < seq; i++)
    {
        for (int j = 0; j < d_model; j++)
            std::cout << h_out[i * d_model + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_out;
    return 0;
}
