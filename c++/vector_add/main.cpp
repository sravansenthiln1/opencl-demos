//
//  Large vector addition Accelerated with OpenCL
//


#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <stdio.h>

// OpenCL includes
#include <CL/opencl.hpp>

// kernel source size (in bytes)
#define SOURCE_SIZE 4096
#define KERNEL_SOURCE "kernel.cl"

// device prefernce
#define deviceType CL_DEVICE_TYPE_ALL

using namespace std;

int main(){
    // allocate memory to store sample data

        const int listSize = 1024;
        vector<int> vecA(listSize);
        vector<int> vecB(listSize);
        for(int i = 0; i < listSize; i++) {
            vecA[i] = i;
            vecB[i] = listSize - i;
        }

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

    // create a command queue for the cl context and an profiling event

        cl::CommandQueue queue(context, default_device, CL_QUEUE_PROFILING_ENABLE, &clErr);
        cl::Event event;

    // Create memory buffers on the device for each vector and write data to it

        cl::Buffer objA(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, listSize * sizeof(int), vecA.data());
        cl::Buffer objB(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, listSize * sizeof(int), vecB.data());
        cl::Buffer objC(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, listSize * sizeof(int));

    // create the program, build it and create the cl kernel

        cl::Program::Sources kernelSource;
        kernelSource.push_back(kernelStr);

        cl::Program program(context, kernelSource);
        if (program.build({default_device}) != CL_SUCCESS) {
            std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
            exit(1);
        }

    // create the kernel and execute in the queue

        cl::Kernel vector_add(program, "vector_add", &clErr);
        if(clErr != CL_SUCCESS){ cout << "Error creating kernel!\n"; exit(1); }
        else{ cout << "kernel created successfully\n"; }

        vector_add.setArg(0, objA);
        vector_add.setArg(1, objB);
        vector_add.setArg(2, objC);

        queue.enqueueNDRangeKernel(vector_add, cl::NullRange, cl::NDRange(listSize), cl::NullRange, NULL, &event);
        queue.finish();

    // get a pointer to map to the result buffer

        cl_int* vecC = (cl_int*)queue.enqueueMapBuffer(objC, CL_TRUE, CL_MAP_READ, 0, listSize * sizeof(int));

    // optional: print the result
    
        //for(int i = 0; i < listSize; i+=100)
        //    printf("%d + %d = %d\n", vecA[i], vecB[i], vecC[i]);

    // release buffers and memory objects

       queue.finish();
       queue.flush();

    // queue event profiling to check execution time 

        event.wait();

        double exec_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

        cout << "Execution time in milliseconds = " <<  exec_time / 10e6 << " ms\n";
        cout << "Execution time in seconds = " << exec_time / 10e9 << " s\n";

    return 0;
}