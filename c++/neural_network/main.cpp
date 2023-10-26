/*
 * Dense Neural network for Sin(x) approximation Accelerated with OpenCL
 *    
 *  Model structure:
 * 
 *     Input (1)
 *       |
 *    Dense (16)    shape(1, 16)
 *       |
 *    Dense (16)    shape(1, 16)
 *       |
 *    Dense (1)     shape(1,1)
 *       |
 *   Output (1)
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

// OpenCL includes
#include <CL/opencl.hpp>

// Neural network Weights and Biases
#include "weights.hpp"

// kernel source size (in bytes)
#define SOURCE_SIZE 4096
#define KERNEL_SOURCE "kernel.cl"

// device preference
#define deviceType CL_DEVICE_TYPE_ALL

using namespace std;

int main(){

    // fetch kernel
        ifstream fileHandle (KERNEL_SOURCE, ios::in);
        string kernelStr((std::istreambuf_iterator<char>(fileHandle)), std::istreambuf_iterator<char>());

        if(kernelStr.size() == 0){
            cout << "Failed to load kernel!\n";
            exit(1);
        }

        size_t sourceSize = kernelStr.size();

        cout << "Fetched kernel source: " << sourceSize << " bytes" << endl;

    // get cl platform and device details

        vector<cl::Platform> platforms;
        vector<cl::Device> devices;
        cl_int clErr;

    // query the number of platforms

        clErr = cl::Platform::get(&platforms);
        if(clErr != CL_SUCCESS){ cout << "Error querying number of OpenCL platforms!\n"; exit(1); }

        cl::Platform default_platform = platforms[0];

    // ID the primary cl compute device (should be the mali gpu here)

        clErr = default_platform.getDevices(deviceType, &devices);
        if(clErr != CL_SUCCESS){ cout << "Error querying OpenCL device ID!\n"; exit(1); }
    
        cl::Device default_device = devices[0];

    // print diagnostic platform and device info
        
        std::cout<< "Platform Name: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout<< "Device Name: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    // create a cl context (handle) to the primary device

        cl::Context context({default_device});

    // create a command queue for the cl context and profiling event for each layer's execution

        cl::CommandQueue queue(context, default_device, CL_QUEUE_PROFILING_ENABLE, &clErr);
        cl::Event L1E1, L1E2, L1E3, L2E1, L2E2, L2E3, L3E1, L3E2;   // event for each layer

    // Create memory buffers on the device for the weights and biases

        vector<float> input = {M_PI / 4};
        
        cout << "Input value: " << input[0] << endl;

        cl::Buffer LIN(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), input.data(), NULL);
        cl::Buffer L1W(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), Layer1Weights.data(), NULL);
        cl::Buffer L1B(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), Layer1Bias.data(), NULL);
        cl::Buffer L2W(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, 16 * 16 * sizeof(float), Layer2Weights.data(), NULL);
        cl::Buffer L2B(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), Layer2Bias.data(), NULL);
        cl::Buffer L3W(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), Layer3Weights.data(), NULL);
        cl::Buffer L3B(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(float), Layer3Bias.data(), NULL);
        cl::Buffer LOUT(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float));

        cl::Buffer tmp(context, CL_MEM_READ_WRITE, 16 * sizeof(float)); // intermediate buffer

    // create the program, build it and create the cl kernel

        cl::Program::Sources kernelSource;
        kernelSource.push_back(kernelStr);

        cl::Program program(context, kernelSource);
        if (program.build({default_device}) != CL_SUCCESS) {
            std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
            exit(1);
        }

    // create the kernels

        cl::Kernel MatMul(program, "MatMul", &clErr);
        if(clErr != CL_SUCCESS){ cout << "Error creating kernel: MatMul!\n"; exit(1); }
        else{ cout << "MatMul kernel created successfully\n"; }

        cl::Kernel Add(program, "Add", &clErr);
        if(clErr != CL_SUCCESS){ cout << "Error creating kernel: Add!\n"; exit(1); }
        else{ cout << "Add kernel created successfully\n"; }

        cl::Kernel ReLU(program, "ReLU", &clErr);
        if(clErr != CL_SUCCESS){ cout << "Error creating kernel: ReLU!\n"; exit(1); }
        else{ cout << "ReLU kernel created successfully\n"; }

    // Execute the kernel layer by layer
        // Layer 1:
        MatMul.setArg(0, L1W);
        MatMul.setArg(1, LIN);
        MatMul.setArg(2, tmp);
        MatMul.setArg(3, 16);
        MatMul.setArg(4, 1);
        MatMul.setArg(5, 1);

        Add.setArg(0, L1B);
        Add.setArg(1, tmp);
        Add.setArg(2, tmp);

        ReLU.setArg(0, tmp);
        ReLU.setArg(1, tmp);

        queue.enqueueNDRangeKernel(MatMul, cl::NullRange, cl::NDRange(16, 1), cl::NullRange, NULL, &L1E1);
        queue.enqueueNDRangeKernel(Add, cl::NullRange, cl::NDRange(16), cl::NullRange, NULL, &L1E2);
        queue.enqueueNDRangeKernel(ReLU, cl::NullRange, cl::NDRange(16), cl::NullRange, NULL, &L1E3);

       // Layer 2:
        MatMul.setArg(0, L2W);
        MatMul.setArg(1, tmp);
        MatMul.setArg(2, tmp);
        MatMul.setArg(3, 16);
        MatMul.setArg(4, 16);
        MatMul.setArg(5, 1);

        Add.setArg(0, tmp);
        Add.setArg(1, L2B);
        Add.setArg(2, tmp);

        ReLU.setArg(0, tmp);
        ReLU.setArg(1, tmp);

        queue.enqueueNDRangeKernel(MatMul, cl::NullRange, cl::NDRange(16, 1), cl::NullRange, NULL, &L2E1);
        queue.enqueueNDRangeKernel(Add, cl::NullRange, cl::NDRange(16), cl::NullRange, NULL, &L2E2);
        queue.enqueueNDRangeKernel(ReLU, cl::NullRange, cl::NDRange(16), cl::NullRange, NULL, &L2E3);

        // Layer 3:
        MatMul.setArg(0, L3W);
        MatMul.setArg(1, tmp);
        MatMul.setArg(2, tmp);
        MatMul.setArg(3, 1);
        MatMul.setArg(4, 16);
        MatMul.setArg(5, 1);

        Add.setArg(0, tmp);
        Add.setArg(1, L3B);
        Add.setArg(2, LOUT);

        queue.enqueueNDRangeKernel(MatMul, cl::NullRange, cl::NDRange(1, 16), cl::NullRange, NULL, &L3E1);
        queue.enqueueNDRangeKernel(Add, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &L3E2);
        queue.finish();
        
    // map the output pointer in memory
    
        cl_float* output = (cl_float*)queue.enqueueMapBuffer(LOUT, CL_TRUE, CL_MAP_READ, 0, sizeof(float));
        cout << "output: " << output[0] << endl;

    // release buffers and memory objects

       queue.finish();
       queue.flush();

    // queue event profiling to check execution time 

        L1E1.wait();
        L1E2.wait();
        L1E3.wait();
        L2E1.wait();
        L2E2.wait();
        L2E3.wait();
        L3E1.wait();
        L3E2.wait();

        double L1E1_exec_time = L1E1.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L1E1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L1E2_exec_time = L1E2.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L1E2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L1E3_exec_time = L1E3.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L1E3.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L1_exec_time = (L1E1_exec_time + L1E2_exec_time + L1E3_exec_time);

        double L2E1_exec_time = L2E1.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L2E1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L2E2_exec_time = L2E2.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L2E2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L2E3_exec_time = L2E3.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L2E3.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L2_exec_time = (L2E1_exec_time + L2E2_exec_time + L2E3_exec_time);

        double L3E1_exec_time = L3E1.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L3E1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L3E2_exec_time = L3E2.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                L3E2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double L3_exec_time = (L3E1_exec_time + L3E2_exec_time);

        // Layer 1
        cout << "\n==== Execution Info ====" << endl;
        cout << "=== Layer 1 ===" << endl;
        cout << "Layer 1 MatMul: " << L1E1_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 1 Add: " << L1E2_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 1 ReLU: " << L1E3_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 1 elapsed time: " << L1_exec_time / 10e6 << " ms" << endl;

        // Layer 2
        cout << "\n=== Layer 2 ===" << endl;
        cout << "Layer 2 MatMul: " << L2E1_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 2 Add: " << L2E2_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 2 ReLU: " << L2E3_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 2 elapsed time: " << L2_exec_time / 10e6 << " ms" << endl;

        // Layer 3
        cout << "\n=== Layer 3 ===" << endl;
        cout << "Layer 3 MatMul: " << L3E1_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 3 Add: " << L3E2_exec_time / 10e6 << " ms" << endl;
        cout << "Layer 3 elapsed time: " << L3_exec_time / 10e6 << " ms" << endl;

        // Total inference time
        double total_inference_time = (L1_exec_time + L2_exec_time + L3_exec_time) / 10e6;
        cout << "\nTotal Inference time in: " << total_inference_time << " ms" << endl;

    return 0;
}