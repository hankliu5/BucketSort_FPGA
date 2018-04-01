__kernel void fpgasort(__global const float *input, 
                        __global const int *bucket, 
                        __global float *restrict output)
{
    // get index of the work item
    int index = get_global_id(0);
    

}

