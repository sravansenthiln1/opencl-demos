__kernel void MatMul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     const int l,
                     const int m,
                     const int n) {

    // get element indexes
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;

    for (int k = 0; k < m; k++) {
        sum += A[row * m + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

__kernel void Add(__global const float *A, 
                  __global const float *B,
                  __global float *C) {
 
    // get element index
    int i = get_global_id(0);
 
    // operate for that index object
    C[i] = A[i] + B[i];
}

__kernel void ReLU(__global const float *A, 
                   __global float *B) {
 
    // get element index
    int i = get_global_id(0);
 
    // operate for that index object
    B[i] = fmax(A[i], 0);
}
