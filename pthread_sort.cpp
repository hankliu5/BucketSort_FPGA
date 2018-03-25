// CSC456: Name -- Yu-Chia Liu
// CSC456: Student ID # -- 200208237
// CSC456: I am implementing -- Bucket Sort
// CSC456: Feel free to modify any of the following. You can only turnin this file.

#include <vector>
#include <iterator>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <pthread.h>
#include "mysort.h"

using namespace std;

const int NUM_BUCKETS = 10;
void *thread_work(void *buckets)
{
    // sorts each bucket.
    vector<float> *b = (vector<float>*) buckets;
    for (int i = 0; i < NUM_BUCKETS/2; i++)
        sort(b[i].begin(), b[i].end());
    return NULL;
}

int pthread_sort(int num_of_elements, float *data)
{
    vector<float> buckets[NUM_BUCKETS];
    float max_num = *max_element(data, data+num_of_elements);
    int step = ceil(max_num / num_of_elements);
    pthread_t sec_thread;

    // puts elements into bucket;
    for (int i = 0; i < num_of_elements; i++)
    {
        int index = static_cast<int>(data[i]/step);
        buckets[index].push_back(data[i]);
    }

    // adds thread work.
    if (pthread_create(&sec_thread, NULL, thread_work, buckets + NUM_BUCKETS/2))
    {
	fprintf(stderr, "Error creating thread.\n");
	return 1;
    }
    thread_work(buckets);
    if (pthread_join(sec_thread, NULL))
    {
	fprintf(stderr, "Error joining thread.\n");
	return 2;
    }   
    // puts elements back from buckets.
    int index = 0;
    for (int i = 0; i < NUM_BUCKETS; i++)
    {
        for (vector<float>::iterator it = buckets[i].begin(); it != buckets[i].end(); it++)
        {
            data[index] = *it;
            index++;
        }
    }
    return 0;
}

