#include <iostream>
#include <cuda_runtime.h>

__global__ void add(float* a , float* b , float* c , int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x; // assigning threads kind of (giving each them specific id)
    if (i < n){ // confirming if number of threads 
        c[i] = a[i] +    b[i];
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

#define TILE 16

__global__ void tiledmatmul(float* A, float* B, float* out, int M, int N, int K) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
          
    float sum = 0.0f;

    for (int i = 0; i < (K + TILE - 1) / TILE; i++) {

        
        if (row < M && (i * TILE + tx) < K)
            tileA[ty][tx] = A[row * K + i * TILE + tx];
        else
            tileA[ty][tx] = 0.0f;

       
        if (col < N && (i * TILE + ty) < K)
            tileB[ty][tx] = B[(i * TILE + ty) * N + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();  

        for (int k = 0; k < TILE; k++)
            sum += tileA[ty][k] * tileB[k][tx];

        __syncthreads();  
    }

    if (row < M && col < N)
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
