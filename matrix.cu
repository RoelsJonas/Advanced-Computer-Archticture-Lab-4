#include <iostream>
#include <cstdint>      // Data types
#include <iostream>     // File operations

#define N 6     // Matrix height/width
#define BLOCK_SIZE  2

// https://imagetostl.com/view-ppm-online

using namespace std;

__global__ void calculateMatrix(int * matrix1, int * matrix2 , int* out) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x);
    int row = index / N;
    int column = index % N;
    out[index] = 0;
    for(int i = 0; i < N; i++){
        out[index] += matrix1[row * N + i] * matrix2[column + i * N];
    }
}

int* getArray(int n) {
    int* res= new int[n*n];
    for(int i = 0; i < n*n; i++) {
        res[i] = i+1;
    }
    return res;
}

void globalMultiply() {
    int* pmatrix1;
    int* pmatrix2;
    int* out = (int*)malloc(N*N*sizeof(int));

//    int matrix1[N*N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//    int matrix2[N*N] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    pmatrix1 = getArray(N);
    pmatrix2 = getArray(N);

    int* cuda_in_matrix1;
    int* cuda_in_matrix2;
    cudaMalloc(&cuda_in_matrix1, N*N*sizeof(int));
    cudaMalloc(&cuda_in_matrix2, N*N*sizeof(int));
    cudaMemcpy( cuda_in_matrix1, pmatrix1, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( cuda_in_matrix2, pmatrix2, N*N*sizeof(int), cudaMemcpyHostToDevice);
    int* cuda_out;
    cudaMalloc(&cuda_out, N*N*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start);

    calculateMatrix<<<N, N>>>(cuda_in_matrix1, cuda_in_matrix2, cuda_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << endl;

    cudaMemcpy( out, cuda_out, N*N*sizeof(int), cudaMemcpyDeviceToHost);

//    for(int i = 0; i < N; i++){
//        for(int j = 0; j < N; j++){
//            cout << out[N * i + j] << ", ";
//        }
//        cout << endl;
//    }

    cudaFree(cuda_in_matrix1);
    cudaFree(cuda_in_matrix2);
    free(out);
}

__constant__ int cuda_in_matrix1_constant[N * N];
__constant__ int cuda_in_matrix2_constant[N * N];


__global__ void calculateConstantMatrix(int* out) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x);
    int row = index / N;
    int column = index % N;
    out[index] = 0;
    for(int i = 0; i < N; i++){
        out[index] += cuda_in_matrix1_constant[row * N + i] * cuda_in_matrix2_constant[column + i * N];
    }
}

void constantMultiply() {
    int* pmatrix1;
    int* pmatrix2;
    int* out = (int*)malloc(N*N*sizeof(int));

//    int matrix1[N*N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//    int matrix2[N*N] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    pmatrix1 = getArray(N);
    pmatrix2 = getArray(N);

    cudaMemcpyToSymbol(cuda_in_matrix1_constant, pmatrix1, N * N * sizeof(int));
    cudaMemcpyToSymbol(cuda_in_matrix2_constant, pmatrix2, N * N * sizeof(int));
    int* cuda_out;
    cudaMalloc(&cuda_out, N*N*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start);

    calculateConstantMatrix<<<N, N>>>(cuda_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << endl;

    cudaMemcpy( out, cuda_out, N*N*sizeof(int), cudaMemcpyDeviceToHost);

//    for(int i = 0; i < N; i++){
//        for(int j = 0; j < N; j++){
//            cout << out[N * i + j] << ", ";
//        }
//        cout << endl;
//    }

//    cudaFree(cuda_in_matrix1);
//    cudaFree(cuda_in_matrix2);
    free(out);
}

__global__ void calculateBlockedSharedMatrix( int* cuda_in_matrix1,int* cuda_in_matrix2, int* out) {
    __shared__ int cuda_in_matrix1_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cuda_in_matrix2_shared[BLOCK_SIZE][BLOCK_SIZE];

    int multiplier = N / BLOCK_SIZE;

    int ix = threadIdx.x;
    int iy = threadIdx.y;

    int row = blockIdx.y * BLOCK_SIZE+ iy;
    int col = blockIdx.x * BLOCK_SIZE + ix;

    int res = 0;
    for (int t = 0; t < multiplier; t++) {
        // Load tiles into shared memory
        cuda_in_matrix1_shared[iy][ix] = cuda_in_matrix1[row * N + t * BLOCK_SIZE + ix];
        cuda_in_matrix2_shared[iy][ix] = cuda_in_matrix2[(t * BLOCK_SIZE + iy) * N + col];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            res += cuda_in_matrix1_shared[iy][k] * cuda_in_matrix2_shared[k][ix];
        }

        __syncthreads();
    }
    out[row * N + col] = res;
}

void sharedMultiply() {
    int* pmatrix1;
    int* pmatrix2;
    int* out = (int*)malloc(N*N*sizeof(int));

//    int matrix1[N*N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//    int matrix2[N*N] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    pmatrix1 = getArray(N);
    pmatrix2 = getArray(N);

    int* cuda_in_matrix1;
    int* cuda_in_matrix2;
    cudaMalloc(&cuda_in_matrix1, N*N*sizeof(int));
    cudaMalloc(&cuda_in_matrix2, N*N*sizeof(int));
    cudaMemcpy( cuda_in_matrix1, pmatrix1, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( cuda_in_matrix2, pmatrix2, N*N*sizeof(int), cudaMemcpyHostToDevice);
    int* cuda_out;
    cudaMalloc(&cuda_out, N*N*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start);


    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(N / BLOCK_SIZE, N / BLOCK_SIZE);
    calculateBlockedSharedMatrix<<<gridSize, blockSize>>>(cuda_in_matrix1, cuda_in_matrix2, cuda_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << endl;

    cudaMemcpy( out, cuda_out, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << out[N * i + j] << ", ";
        }
        cout << endl;
    }


    free(out);
}


int main() {
//    for(int i = 0; i < 50; i++)
        globalMultiply();
//    cout << "GLOBAL" << endl;
//    for(int i = 0; i < 100; i++)
//        globalMultiply();
//    cout << "CONSTANT" << endl;
//    for(int i = 0; i < 100; i++)
//        constantMultiply();
//    cout << "SHARED" << endl;
//    for(int i = 0; i < 100; i++)
        sharedMultiply();


    return 0;
}
