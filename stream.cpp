#include<thread>
#include<vector>
#include<chrono>
#include<atomic>
#include<iostream>

#include<cmath>
#include<math.h>
#include <assert.h>

#ifdef GEM5_ANNOTATION
#include<gem5/m5ops.h>
#endif

struct error_stat {
    double total_error;
    size_t error_count;
};

// setup global variables
unsigned int total_supported_threads = -1;
unsigned barrier_valriable = 0;
// unsigned barrier_variable_v2 = 0;
std::atomic<unsigned int> barrier_variable_v2(0);
// std::atomic<size_t> part;
size_t part = 0; // has to be reset before invoking each kernel

// utility functions

void kernel_copy_v2(double *A, double *B, size_t size, size_t idx) {
    // for a portion of the array, we have to use a serial approach
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = B[i];
    barrier_variable_v2++;
}

void kernel_scale_v2(double alpha, double *A, double *B, size_t size,
        size_t idx) {
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = alpha * B[i];
    barrier_variable_v2++;
}

void kernel_sum_v2(double* A, double* B, double* C, size_t size, size_t idx) {
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = B[i] + C[i];
    barrier_variable_v2++;
}

void kernel_triad_v2(double alpha, double* A, double* B, double* C,
        size_t size, size_t idx) {
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = B[i] + alpha * C[i];
    barrier_variable_v2++;
}
// declare all the kernel functions here
void kernel_copy(double* a, double* b, size_t size) {
    // how to do the indexing in this array?
    // each call will do `total_supported_threads`th part of the array.
    size_t thread_part = part++;

    for(size_t i = thread_part * (size / total_supported_threads);
            i < (thread_part + 1) * (size / total_supported_threads);
            i++)
        a[i] = b[i];
    barrier_valriable++;
}

void kernel_scale(double alpha, double* A, double* B, size_t size) {
    // this kernel just multiplies the values
    size_t thread_part = part++;

    for(size_t i = thread_part * (size / total_supported_threads);
            i < (thread_part + 1) * (size / total_supported_threads);
            i++)
        A[i] = alpha * B[i];
    barrier_valriable++;
}

void kernel_sum(double* A, double* B, double* C, size_t size) {
    // this kernel just multiplies the values
    size_t thread_part = part++;

    for(size_t i = thread_part * (size / total_supported_threads);
            i < (thread_part + 1) * (size / total_supported_threads);
            i++)
        A[i] = B[i] + C[i];
    barrier_valriable++;
}

void kernel_triad(double alpha, double* A, double* B, double* C, size_t size) {
        // this kernel just multiplies the values
    size_t thread_part = part++;

    for(size_t i = thread_part * (size / total_supported_threads);
            i < (thread_part + 1) * (size / total_supported_threads);
            i++)
        A[i] = B[i] + alpha * C[i];
    barrier_valriable++;
}

// declare all the error functions here
struct error_stat total_error(double* a, double *b, size_t size) {
    // a is the parallel array
    // b is the ground truth
    error_stat et;
    et.total_error = 0;
    et.error_count = 0;
    for(int i = 0 ; i < size ; i++) {
        et.total_error += abs(b[i] - a[i]);
        if(a[i] != b[i]) {
            // std::cout << "index " << i << std::endl;
            et.error_count++;
        }
    }
    return et;
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: $./stream <size of each array>";
        return -1;
    }

    // allocate the arrays
    size_t array_size = (size_t)atol(argv[1]);
    double data_size_in_gb = 
            array_size * sizeof(double) / 1024.0 / 1024.0 / 1024.0;

    std::cout << "STREAM info: Array Size : " << data_size_in_gb << " GB"
            << std::endl;
    std::cout << "STREAM info: Total Memory Occupied : " <<
            4 * data_size_in_gb << " GB" << std::endl;

    double *a = new double[array_size];
    double *b = new double[array_size];
    double *c = new double[array_size];
    double alpha = 1000;

    // setup the values in all these arrays
    for(size_t i = 0 ; i < array_size ; i++) {
        a[i] = 1;
        b[i] = 2;
        c[i] = 3;
    }

    // setup threads here
    total_supported_threads = std::thread::hardware_concurrency();
    // total_supported_threads = 1;
    // sanity check for the number of threads
    assert(total_supported_threads != -1);
    std::cout << "STREAM info: Total Number of Threads : " <<
            total_supported_threads << std::endl;
    
    // end of overall stats print
    std::cout << std::endl;

    // now allocate memory for each of the threads
    std::thread* stream_threads = new std::thread[total_supported_threads];
    // std::thread stream_threads[total_supported_threads];

    // warmup : do all the kernels without timing and in serial
    //
    // TODO
    //
    // idk how to do this. so skipping this for now.
    // setup some initial numbers for uniformity
    part = 0;
    barrier_variable_v2 = 0;
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        // index will be multiplied to find the next index
        stream_threads[i] = std::thread(kernel_copy_v2, c, a, array_size, i);
    // setup initial timers for the actual kernel calls
    auto start = std::chrono::steady_clock::now();
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    while(barrier_variable_v2 != total_supported_threads);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = 
        std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
        ).count();
    double time_copy = std::chrono::duration<double>(elapsed).count();

    // verify the array. print stats --------------------------------------- //
    error_stat e_copy = total_error(c, a, array_size);
    std::cout << "STREAM info: COPY Kernel";
    std::cout << "\t\tBandwidth  : " <<
            (2 * data_size_in_gb * 1e6)/time_copy << " GB/s";
    std::cout << "\tTime: " << time_copy/1e6 << " s";
    std::cout << "\t\tTotal Error: " << e_copy.total_error;
    std::cout << "\tError Count: " << e_copy.error_count << std::endl;

    // -------------------------- end of copy kernel ----------------------- //
    
    // reset the part counter for the new kernel to run.
    part = 0;
    barrier_variable_v2 = 0;
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        stream_threads[i] = std::thread(
            kernel_scale_v2, alpha, b, c, array_size, i);

    // setup the initial timers to keep a track of time.
    start = std::chrono::steady_clock::now();
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    while(barrier_variable_v2 != total_supported_threads);
    end = std::chrono::steady_clock::now();
    elapsed = 
        std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
        ).count();
    time_copy = std::chrono::duration<double>(elapsed).count();

    // verify the array. print stats --------------------------------------- //
    // create another array which has the expected results
    double *expected_array = new double[array_size];
    for(size_t i = 0 ; i < array_size ; i++)
        expected_array[i] = alpha * a[i];
    error_stat e_scale = total_error(expected_array, b, array_size);
    delete expected_array;
    std::cout << "STREAM info: SCALE Kernel";
    std::cout << "\t\tBandwidth  : " <<
            (2 * data_size_in_gb * 1e6)/time_copy << " GB/s";
    std::cout << "\tTime: " << time_copy/1e6 << " s";
    std::cout << "\t\tTotal Error: " << e_scale.total_error;
    std::cout << "\tError Count: " << e_scale.error_count << std::endl;

    // ------------------------- end of scale kernel ----------------------- //
    
    // reset the part counter for the new kernel to run.
    part = 0;
    barrier_variable_v2 = 0;
    // create this new kernel
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        stream_threads[i] = std::thread(kernel_sum_v2, c, a, b, array_size, i);

    // setup the initial timers to keep a track of time.
    start = std::chrono::steady_clock::now();
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    // while(barrier_valriable < total_supported_threads - 1);
    while(barrier_variable_v2 != total_supported_threads);
    end = std::chrono::steady_clock::now();
    elapsed = 
        std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
        ).count();
    time_copy = std::chrono::duration<double>(elapsed).count();
    // verify the array. print stats
    // create another array which has the expected results
    expected_array = new double[array_size];
    for(size_t i = 0 ; i < array_size ; i++)
        expected_array[i] = a[i] + b[i];
    error_stat e_sum = total_error(expected_array, c, array_size);
    delete expected_array;
    std::cout << "STREAM info: SUM Kernel ";
    std::cout << "\t\tBandwidth  : " <<
            (3 * data_size_in_gb * 1e6)/time_copy << " GB/s";
    std::cout << "\tTime: " << time_copy/1e6 << " s";
    std::cout << "\t\tTotal Error: " << e_sum.total_error;
    std::cout << "\tError Count: " << e_sum.error_count << std::endl;

    // -------------------------- end of sum kernel ------------------------ //

    // reset the part counter for the new kernel to run.
    part = 0;
    barrier_variable_v2 = 0;
    // create this new kernel
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        stream_threads[i] =
            std::thread(kernel_triad_v2, alpha, a, b, c, array_size, i);

    // setup the initial timers to keep a track of time.
    start = std::chrono::steady_clock::now();
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    // while(barrier_valriable < total_supported_threads - 1);
    while(barrier_variable_v2 != total_supported_threads);
    end = std::chrono::steady_clock::now();
    elapsed = 
        std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
        ).count();
    time_copy = std::chrono::duration<double>(elapsed).count();
    // verify the array. print stats
    // create another array which has the expected results
    expected_array = new double[array_size];
    for(size_t i = 0 ; i < array_size ; i++)
        expected_array[i] = b[i] + alpha * c[i];
    error_stat e_triad = total_error(expected_array, a, array_size);
    delete expected_array;
    std::cout << "STREAM info: TRIAD Kernel";
    std::cout << "\t\tBandwidth  : " <<
            (3 * data_size_in_gb * 1e6)/time_copy << " GB/s";
    std::cout << "\tTime: " << time_copy/1e6 << " s";
    std::cout << "\t\tTotal Error: " << e_triad.total_error;
    std::cout << "\tError Count: " << e_triad.error_count << std::endl;

    // -------------------------- end of triad kernel ---------------------- //

    return 0;
}
