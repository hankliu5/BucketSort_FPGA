__kernel void fpgasort(__global const float * restrict input,
                        __global int * restrict output,
                        int step)
{
    // get index of the work item
    int index = get_global_id(0); 
    output[index] = convert_int(input[index] / step);
}
