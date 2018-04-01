#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mysort.h"
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>
#include <string>
#ifdef APPLE
#include <OpenCL/opencl.h>
//#include "scoped_array.h"
#else
#include "CL/opencl.h"
#include "AOCL_Utils.h"
using namespace aocl_utils;
#endif
#ifdef APPLE
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 1;
cl_device_id device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel kernel; // num_devices elements
#else
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
#endif

#ifdef APPLE
static int LoadTextFromFile(const char *file_name, char **result_string, size_t *string_len);
#define LOCAL_MEM_SIZE = 1024;
void _checkError(int line,
								 const char *file,
								 cl_int error,
                 const char *msg,
                 ...);

#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)
#endif

// problem data.
scoped_array< scoped_aligned_ptr<float> > input;
scoped_array< scoped_aligned_ptr<float> > output;

const int NUM_BUCKETS = 10;

bool init_opencl(int num_of_elements);
void init_problem(float *data);
void cl_run(int num_of_elements, float *data);
void cleanup();

int fpga_sort(int num_of_elements, float *data)
{
    init_opencl();

    return 0;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  int err;
  cl_int status;

  printf("Initializing OpenCL\n");
#ifdef APPLE
  int gpu = 1;
  err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Error: Failed to create a device group!\n");
    return EXIT_FAILURE;
  }
  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");
#else
  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
 if(platform == NULL) {
   printf("ERROR: Unable to find Altera OpenCL platform.\n");
   return false;
 }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }
  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create context");
#endif

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
#ifndef APPLE
  std::string binary_file = getBoardBinaryFile("fpgasort", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  //Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);
  input_buf.reset(num_devices);
  bucket_buf.reset(num_devices);
  step_buf.reset(num_devices);
  output_buf.reset(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "fpgasort";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    n_per_device[i] = num_of_elements / num_devices;

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if (i < (num_of_elements % num_devices)) {
        n_per_device[i]++;
    }

    input_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
            n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input");

    bucket_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
            NUM_BUCKETS * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input");

    step_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
            1 * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input");

    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
  }
#else
  char *source = 0;
  size_t length = 0;
  LoadTextFromFile("fpgasort.cl", &source, &length);
  const char *kernel_name = "fpgasort";
  program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  checkError(status, "Failed to build program");

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  kernel = clCreateKernel(program, kernel_name, &status);
#endif
  return true;
}

void init_problem(float *data) {
    if (num_devices == 0) {
        checkError(-1, "No devices");
    }

    input.reset(num_devices);
    output.reset(num_devices);

    float *ptr = data;

    for (unsigned i = 0; i < num_devices; ++i) {
        input[i].reset(n_per_device[i]);
        output[i].reset(n_per_device[i]);

        for (unsigned j = 0; j < n_per_device[i]; ++j) {
            input[i][j] = *ptr;
            ptr++;
        }
    }
}

void cl_run(int num_of_elements, float *data) {
    cl_int status;

    scoped_array<cl_event> kernel_event(num_devices);
    scoped_array<cl_event> finish_event(num_devices);

    // calculate the step for the bucket.
    float max_num = *max_element(data, data+num_of_elements);
    int step = ceil(max_num / num_of_elements);
    int elem_in_bucket[NUM_BUCKETS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (unsigned i = 0; i < num_devices; ++i) {
        cl_event write_event[3];
        status = clEnqueueWriteBuffer(queue[i], input_buf[i], CL_FALSE,
                0, n_per_device[i] * sizeof(float), input[i], 0, NULL,
                &write_event[0]);
        checkError(status, "Failed to transfer input.");

        status = clEnqueueWriteBuffer(queue[i], bucket_buf[i], CL_FALSE,
                0, NUM_BUCKETS * sizeof(int), &elem_in_bucket, 0, NULL,
                &write_event[1]);
        checkError(status, "Failed to transfer elem_in_bucket.");

        status = clEnqueueWriteBuffer(queue[i], step_buf[i], CL_FALSE,
                0, 1 * sizeof(int), &step, 0, NULL,
                &write_event[2]);
        checkError(status, "Failed to transfer step.");

        unsigned argi = 0;
        status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem),
                &input_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem),
                &bucket_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem),
                &step_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem),
                &output_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        const size_t global_work_size = n_per_device[i];
        printf("Launching for device %d (%d elements)\n", i,
                global_work_size);

        status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
                &global_work_size, NULL, 2, write_event, &kernel_event[i]);
        checkError(status, "Failed to launch kernel");

        status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
                0, n_per_device[i] * sizeof(float), output[i], 1,
                &kernel_event[i], &finish_event[i]);
        clReleaseEvent(write_event[0]);
        clReleaseEvent(write_event[1]);
        clReleaseEvent(write_event[2]);

    }
    clWaitForEvents(num_devices, finish_event);
}

void cleanup() {
#ifndef APPLE
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_buf && input_buf[i]) {
        clReleaseMemObject(input_buf[i]);
    }
    if(bucket_buf && bucket_buf[i]) {
        clReleaseMemObject(bucket_buf[i]);
    }
    if(step_buf && step_buf[i]) {
        clReleaseMemObject(step_buf[i]);
    }
    if(output_buf && output_buf[i]) {
        clReleaseMemObject(output_buf[i]);
    }
  }
#else
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
#endif
  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}
#ifdef APPLE
static int LoadTextFromFile(
    const char *file_name, char **result_string, size_t *string_len)
{
    int fd;
    unsigned file_len;
    struct stat file_status;
    int ret;

    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1)
    {
        printf("Error opening file %s\n", file_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret)
    {
        printf("Error reading status for file %s\n", file_name);
        return -1;
    }
    file_len = file_status.st_size;

    *result_string = (char*)calloc(file_len + 1, sizeof(char));
    ret = read(fd, *result_string, file_len);
    if (!ret)
    {
        printf("Error reading from file %s\n", file_name);
        return -1;
    }

    close(fd);

    *string_len = file_len;
    return 0;
}

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

void _checkError(int line,
								 const char *file,
								 cl_int error,
                 const char *msg,
                 ...) {
	// If not successful
	if(error != CL_SUCCESS) {
		// Print line and file
    printf("ERROR: ");
    printf("\nLocation: %s:%d\n", file, line);

    // Print custom message.
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    // Cleanup and bail.
    cleanup();
    exit(error);
    }
}
#endif
