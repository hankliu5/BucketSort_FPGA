__kernel void fpgasort(__global const float *input,
                        __global int *output,
                        int step)
{
    // get index of the work item
    int index = get_global_id(0);
    output[index] = int(input[index] / step);
}
