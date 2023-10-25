import pyopencl as cl
import numpy as np
from weights import Layer1Weights, Layer1Bias, Layer2Weights, Layer2Bias, Layer3Weights, Layer3Bias
import math

# Load OpenCL kernel source from a file
KERNEL_SOURCE = "kernel.cl"
DEVICE_TYPE = cl.device_type.ALL

with open(KERNEL_SOURCE, "r") as source_file:
    kernel_str = source_file.read()

# Fetch the platform and device details
platforms = cl.get_platforms()

devices = []
for platform in platforms:
    print("Platform name: {}".format((platform.name)))
    devices.extend(platform.get_devices(DEVICE_TYPE))
    for device in devices:
        print(f"Device Name: {device.name}")

if not devices:
    print("No OpenCL devices found.")
    exit(1)

# Use the first available device
device = devices[0]

print("Executing on device: {}".format((device.name)))

# Create an OpenCL context and command queue
context = cl.Context([device])
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Define the neural network weights and biases
input_value = [math.pi / 4]
print("Input:", input_value[0])

output = np.array([0], dtype=np.float32)

# Create memory buffers on the device for the weights and biases
LIN = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, np.float32().nbytes, 
                                                                                hostbuf=np.array(input_value, dtype=np.float32))

L1W = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 16 * np.float32().nbytes, 
                                                                                hostbuf=np.array(Layer1Weights, dtype=np.float32))

L1B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 16 * np.float32().nbytes, 
                                                                                hostbuf=np.array(Layer1Bias, dtype=np.float32))

L2W = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 16 * 16 * np.float32().nbytes, 
                                                                                hostbuf=np.array(Layer2Weights, dtype=np.float32))

L2B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 16 * np.float32().nbytes, 
                                                                                hostbuf=np.array(Layer2Bias, dtype=np.float32))

L3W = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 16 * np.float32().nbytes,
                                                                                hostbuf=np.array(Layer3Weights, dtype=np.float32))

L3B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, np.float32().nbytes, 
                                                                                hostbuf=np.array(Layer3Bias, dtype=np.float32))

LOUT = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, np.float32().nbytes, 
                                                                                hostbuf=output)

tmp = cl.Buffer(context, cl.mem_flags.READ_WRITE, 16 * np.float32().nbytes)

# Create the OpenCL program, build it, and create the kernels
kernel_source = cl.Program(context, kernel_str).build()
MatMul = kernel_source.MatMul
Add = kernel_source.Add
ReLU = kernel_source.ReLU

# Execute the kernel layer by layer
# Layer 1:
L1E1 = MatMul(queue, (16,1), None, L1W, LIN, tmp, np.array(16, dtype=np.int32), np.array(1, dtype=np.int32), np.array(1, dtype=np.int32))
L1E2 = Add(queue, (16, 1), None, L1B, tmp, tmp)
L1E3 = ReLU(queue, (16,), None, tmp, tmp)

# Layer 2:
L2E1 = MatMul(queue, (16,1), None, L2W, tmp, tmp, np.array(16, dtype=np.int32), np.array(16, dtype=np.int32), np.array(1, dtype=np.int32))
L2E2 = Add(queue, (16,), None, tmp, L2B, tmp)
L2E3 = ReLU(queue, (16,), None, tmp, tmp)

# Layer 3:
L3E1 = MatMul(queue, (1, 16), None, L3W, tmp, tmp, np.array(1, dtype=np.int32), np.array(16, dtype=np.int32), np.array(1, dtype=np.int32))
L3E2 = Add(queue, (1,), None, tmp, L3B, LOUT)
queue.finish()

# Read the output from the device
cl.enqueue_copy(queue, output, LOUT).wait()
print("Output:", output[0])

# Release OpenCL resources
queue.flush()
queue.finish()

# Queue event profiling to check execution time
L1E1_exec_time = L1E1.profile.end - L1E1.profile.start
L1E2_exec_time = L1E2.profile.end - L1E2.profile.start
L1E3_exec_time = L1E3.profile.end - L1E3.profile.start
L1_exec_time = (L1E1_exec_time + L1E2_exec_time + L1E3_exec_time)

L2E1_exec_time = L2E1.profile.end - L2E1.profile.start
L2E2_exec_time = L2E2.profile.end - L2E2.profile.start
L2E3_exec_time = L2E3.profile.end - L2E3.profile.start
L2_exec_time = (L2E1_exec_time + L2E2_exec_time + L2E3_exec_time)

L3E1_exec_time = L3E1.profile.end - L3E1.profile.start
L3E2_exec_time = L3E2.profile.end - L3E2.profile.start
L3_exec_time = (L3E1_exec_time + L3E2_exec_time)

print("\n==== Execution Info ====")
print("=== Layer 1 ===")
print("Layer 1 MatMul:", L1E1_exec_time / 10e6, "ms")
print("Layer 1 Add", L1E2_exec_time / 10e6, "ms")
print("Layer 1 ReLU:", L1E3_exec_time / 10e6, "ms")
print("layer 1 elapsed time:",  L1_exec_time / 10e6, "ms")

print("\n=== Layer 2 ===")
print("Layer 2 MatMul:", L2E1_exec_time / 10e6, "ms")
print("Layer 2 Add", L2E2_exec_time / 10e6, "ms")
print("Layer 2 ReLU:", L2E3_exec_time / 10e6, "ms")
print("layer 2 elapsed time:", L2_exec_time / 10e6, "ms")

print("\n=== Layer 3 ===")
print("Layer 3 MatMul:", L3E1_exec_time / 10e6, "ms")
print("Layer 3 Add", L3E2_exec_time / 10e6, "ms")
print("layer 2 elapsed time:", L3_exec_time / 10e6, "ms")

total_inference_time = (L1_exec_time + L2_exec_time + L3_exec_time) / 10e6
print("\nTotal Inference time in:", total_inference_time, "ms")


