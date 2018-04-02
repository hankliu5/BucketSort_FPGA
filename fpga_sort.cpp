#include <stdio.h>
#include <stdlib.h>
#include "mysort.h"
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;
#ifdef APPLE
#include <OpenCL/opencl.h>
//#include "scoped_array.h"
#else
#include "CL/opencl.h"
#include "AOCL_Utils.h"
using namespace aocl_utils;
#endif


cl_platform_id platform = NULL;
cl_context context = NULL;
cl_program program = NULL;
cl_mem device_input, device_output;
const int NUM_BUCKETS = 6;

#ifdef APPLE
// OpenCL runtime configuration
unsigned num_devices = 1;
cl_device_id device; // num_devices elements
cl_command_queue queue; // num_devices elements
cl_kernel kernel; // num_devices elements
#else
// OpenCL runtime configuration
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
scoped_array<cl_command_queue> queue; // num_devices elements
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

bool init_opencl(int num_of_elements, float *data);
double getCurrentTimestamp();
void cleanup();

int fpga_sort(int num_of_elements, float *data)
{
    init_opencl(num_of_elements, data);
    return 0;
}

// Initializes the OpenCL objects.
bool init_opencl(int num_of_elements, float *data) {
  int err;
  cl_int status;
	float max_num = *max_element(data, data+num_of_elements);
	printf("max_num: %f\n", max_num);
	int step = ceil(max_num / num_of_elements);
	printf("step: %d\n", step);
	int *output_buckets = (int*) calloc(num_of_elements, sizeof(int));
	// float *output = (float*) calloc(num_of_elements, sizeof(int));
	vector<float> buckets[NUM_BUCKETS];

	// for (int i = 0; i < num_of_elements; i++) {
	// 	printf("%f ", data[i]);
	// }
	printf("\n");

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
  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "fpgasort";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");
  }
#else
  char *source = 0;
  size_t length = 0;
  LoadTextFromFile("fpgasort.cl", &source, &length);
  const char *kernel_name = "fpgasort";
  program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &err);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  checkError(status, "Failed to build program");

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  kernel = clCreateKernel(program, kernel_name, &status);
#endif

  // Allocate memory to the device.
	device_input = clCreateBuffer(context, CL_MEM_READ_ONLY,
		num_of_elements * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to create a buffer for inputs!\n");
		return EXIT_FAILURE;
	}

	device_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		num_of_elements * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to create a buffer for outputs!\n");
		return EXIT_FAILURE;
	}

        for (unsigned i = 0; i < num_devices; ++i) {
	err = clEnqueueWriteBuffer(queue[i], device_input, CL_FALSE,
        0, num_of_elements * sizeof(float), data, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to transfer buffer for input");
    exit(1);
  }

	err = clEnqueueWriteBuffer(queue[i], device_output, CL_FALSE,
        0, num_of_elements * sizeof(int), output_buckets, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to transfer buffer for input");
    exit(1);
  }
	clFinish(queue[i]);

	cl_event kernel_event;
	unsigned argi = 0;

	status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &device_input);
  checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &device_output);
  checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel[i], argi++, sizeof(step), &step);
  checkError(status, "Failed to set argument %d", argi - 1);

	const size_t global_work_size = num_of_elements;
	status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
			&global_work_size, NULL, 0, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel");
	clFinish(queue[i]);
	clReleaseEvent(kernel_event);
	status = clEnqueueReadBuffer(queue[i], device_output, CL_TRUE,
			0, num_of_elements * sizeof(int), output_buckets, 0, NULL, NULL);
	checkError(status, "Failed to read output matrix");
	clFinish(queue[i]);
}
	// int *buckets = (int*) calloc(NUM_BUCKETS, sizeof(int));
	// for (int i = 0; i < num_of_elements; i++) {
	// 	buckets[output_buckets[i]]++;
	// }
	// for (int i = 0; i < num_of_elements; i++) {
	// 	int bucket_id = output_buckets[i];
	// 	int id = --buckets[bucket_id];
	// 	output[id] = data[i];
	// }
	// for (int i = 0; i < NUM_BUCKETS - 1; i++) {
	// 	sort(output+buckets[i], output+buckets[i+1]);
	// }
	// sort(output+buckets[NUM_BUCKETS-1], output+num_of_elements);
	// memcpy(data, output, num_of_elements * sizeof(float));


  // for (int i = 0; i < num_of_elements; i++) {
	// 	printf("num %d: index %d\n", i, output_buckets[i]);
	// }
	for (int i = 0; i < num_of_elements; i++) {
    buckets[output_buckets[i]].push_back(data[i]);
  }
	for (int i = 0; i < NUM_BUCKETS; i++) {
		sort(buckets[i].begin(), buckets[i].end());
	}
	int index = 0;
  for (int i = 0; i < NUM_BUCKETS; i++)
  {
      for (vector<float>::iterator it = buckets[i].begin(); it != buckets[i].end(); it++)
      {
          data[index] = *it;
          index++;
      }
  }
	// free(buckets);
	free(output_buckets);
	// free(output);
	cleanup();
	return true;
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
