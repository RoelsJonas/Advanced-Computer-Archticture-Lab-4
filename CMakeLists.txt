cmake_minimum_required(VERSION 3.21)
project(lab4 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab4 matrix.cu)

set_target_properties(lab4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
