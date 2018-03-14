__kernel void fpgasort(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z)
{
    // get index of the work item
    int index = get_global_id(0);

}

