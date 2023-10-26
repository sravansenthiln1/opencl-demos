import pyopencl as cl
import numpy as np

# Constants
KERNEL_SOURCE = "kernel.cl"
DEVICE_TYPE = cl.device_type.ALL

# Load OpenCL kernel source from a file
with open(KERNEL_SOURCE, "r") as source_file:
    kernel_str = source_file.read()

# Create a list of numbers
list_size = 1024
vec_a = np.array([i for i in range(list_size)], dtype=np.int32)
vec_b = np.array([list_size - i for i in range(list_size)], dtype=np.int32)

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

# Create memory buffers on the device
obj_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.COPY_HOST_PTR, hostbuf=vec_a)
obj_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.COPY_HOST_PTR, hostbuf=vec_b)
obj_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR, list_size * np.int32().nbytes)

# Create the OpenCL program, build it, and create the kernel
program = cl.Program(context, kernel_str).build()
vector_add = program.vector_add

# Enqueue the kernel
global_work_size = (list_size,)
local_work_size = None
event = vector_add(queue, global_work_size, local_work_size, obj_a, obj_b, obj_c)

# Read the result buffer into local memory
vec_c = cl.enqueue_map_buffer(queue, obj_c, cl.map_flags.READ, 0, vec_a.shape, vec_a.dtype)

# Optional: Print the result
#for i in range(0, list_size, 100):
#    print(f"{vec_a[i]} + {vec_b[i]} = {vec_c[i]}")

# Queue event profiling to check execution time
exec_time = event.profile.end - event.profile.start

print("Execution time in milliseconds =", exec_time / 10e6, "ms")
print("Execution time in seconds =", exec_time / 10e9, "s")
