//#include <iostream>
//#include <cstdint>      // Data types
//#include <iostream>     // File operations
//
//// #define M 512       // Lenna width
//// #define N 512       // Lenna height
//#define M 941       // VR width
//#define N 704       // VR height
//#define C 3         // Colors
//#define OFFSET 15   // Header length
//
//
//// https://imagetostl.com/view-ppm-online
//
//using namespace std;
//
//uint8_t* get_image_array(void){
//    /*
//     * Get the data of an (RGB) image as a 1D array.
//     *
//     * Returns: Flattened image array.
//     *
//     * Noets:
//     *  - Images data is flattened per color, column, row.
//     *  - The first 3 data elements are the RGB components
//     *  - The first 3*M data elements represent the firts row of the image
//     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
//     *
//     */
//    // Try opening the file
//    FILE *imageFile;
//    imageFile=fopen("./input_image.ppm","rb");
//    if(imageFile==NULL){
//        perror("ERROR: Cannot open output file");
//        exit(EXIT_FAILURE);
//    }
//
//    // Initialize empty image array
//    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
//
//    // Read the image
//    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
//
//    // Close the file
//    fclose(imageFile);
//
//    // Move the starting pointer and return the flattened image array
//    return image_array + OFFSET;
//}
//
//
//void save_image_array(uint8_t* image_array){
//    /*
//     * Save the data of an (RGB) image as a pixel map.
//     *
//     * Parameters:
//     *  - param1: The data of an (RGB) image as a 1D array
//     *
//     */
//    // Try opening the file
//    FILE *imageFile;
//    imageFile=fopen("./output_image.ppm","wb");
//    if(imageFile==NULL){
//        perror("ERROR: Cannot open output file");
//        exit(EXIT_FAILURE);
//    }
//
//
//    // Configure the file
//    fprintf(imageFile,"P6\n");               // P6 filetype
//    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
//    fprintf(imageFile,"255\n");              // Max pixel
//
//    // Write the image
//    fwrite(image_array, 1, M*N*C, imageFile);
//
//    // Close the file
//    fclose(imageFile);
//}
//
//__global__ void calculateAverage(uint8_t * in, uint8_t* out) {
//    int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
//    int k = 0;
//    while(i + k * gridDim.x * blockDim.x < M*N*C) {
//        int index = i + k * gridDim.x * blockDim.x;
//        int avg = (in[index] + in[index+1] + in[index+2])/3;
//        out[index] = avg;
//        out[index+1] = avg;
//        out[index+2] = avg;
//        k++;
//    }
//}
//
//__global__ void calculateAverageRRR(uint8_t* in, uint8_t* out) {
//    int i = (blockIdx.x * blockDim.x + threadIdx.x);
//    int k = 0;
//    while(i + k * gridDim.x * blockDim.x < M*N) {
//        int index = i + k * gridDim.x * blockDim.x;
//        int avg = (in[index] + in[index+M*N] + in[index+2*M*N])/3;
//        out[index] = avg;
//        out[index+M*N] = avg;
//        out[index+2*M*N] = avg;
//        k++;
//    }
//}
//
//void process(uint8_t* image_array, bool sort) {
//    // sort array
//    uint8_t* sorted_image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
//
//    if(sort) {
//        for (int i = 0; i < M * N * C; i++) {
//            if (i % 3 == 0) sorted_image_array[i / 3] = image_array[i];
//            else if (i % 3 == 1) sorted_image_array[(M * N) + i / 3] = image_array[i];
//            else sorted_image_array[(2 * M * N) + i / 3] = image_array[i];
//        }
//    }
//
//    uint8_t* output_image = (uint8_t*) malloc(M*N*C*sizeof(uint8_t));
//    uint8_t* cuda_in;
//    cudaMalloc(&cuda_in, M*N*C*sizeof(uint8_t));
//    if(sort) cudaMemcpy( cuda_in, sorted_image_array, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);
//    else cudaMemcpy( cuda_in, image_array, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);
//    uint8_t* cuda_out;
//    cudaMalloc(&cuda_out, M*N*C*sizeof(uint8_t));
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord( start);
//
//    if(sort) calculateAverageRRR<<<1, 1024>>>(cuda_in, cuda_out);
//    else calculateAverage<<<1, 1024>>>(cuda_in, cuda_out);
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << milliseconds << endl;
//
//    cudaMemcpy( output_image, cuda_out, M*N*C*sizeof(uint8_t), cudaMemcpyDeviceToHost);
//
//    if(sort){
//        uint8_t* unsorted_output_image = (uint8_t*) malloc(M*N*C);
//
//        for(int i = 0; i < M*N*C; i+=3){
//            unsorted_output_image[i] = output_image[i/3];
//            unsorted_output_image[i+1] = output_image[i/3 + M*N];
//            unsorted_output_image[i+2] = output_image[i/3 + 2*M*N];
//        }
//        //save_image_array(unsorted_output_image);
//        free(unsorted_output_image);
//    }
//    else{
//        //save_image_array(output_image);
//    }
//
//
//
//    cudaFree(cuda_out);
//    cudaFree(cuda_in);
//    free(output_image);
//    free(sorted_image_array);
//}
//
//int main() {
////    uint8_t* image_array = get_image_array();
////
////    for(int i = 0; i < 50; i++) process(image_array, false);
////    cout << "UNCOALESCED" << endl;
////    for(int i = 0; i < 100; i++) process(image_array, false);
////    cout << "COALESCED" << endl;
////    for(int i = 0; i < 100; i++) process(image_array, true);
//
//
//    return 0;
//}
