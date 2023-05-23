#include<thread>
#include<vector>
#include<chrono>
#include<atomic>
#include<string>
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
std::atomic<unsigned int> barrier_variable_v2(0);
// std::atomic<size_t> part;
size_t part = 0; // has to be reset before invoking each kernel

// declare all the kernel functions here
void kernel_copy(double *A, double *B, size_t size, size_t idx) {
    // for a portion of the array, we have to use a serial approach
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = B[i];
    barrier_variable_v2++;
}

void kernel_scale(double alpha, double *A, double *B, size_t size,
        size_t idx) {
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = alpha * B[i];
    barrier_variable_v2++;
}

void kernel_sum(double* A, double* B, double* C, size_t size, size_t idx) {
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = B[i] + C[i];
    barrier_variable_v2++;
}

void kernel_triad(double alpha, double* A, double* B, double* C,
        size_t size, size_t idx) {
    size_t chunk_size = size / total_supported_threads;
    for(size_t i = idx * chunk_size; i < (idx + 1) * chunk_size ; i++)
        A[i] = B[i] + alpha * C[i];
    barrier_variable_v2++;
}

// declare all the utility functions here
struct error_stat total_error(double* a, double *b, size_t size) {
    // a is the parallel array
    // b is the ground truth (the expected array)
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

void stats_printer(std::string kernel_name, double data_size_in_gb,
        double time_copy, error_stat error_stats) {
    
    double mult_factor = 3;
    if(kernel_name == "COPY" || kernel_name == "SCALE")
        mult_factor = 2;

    std::cout << "STREAM info: " << kernel_name << " Kernel";
    std::cout << "\t\tBandwidth  : " <<
            (mult_factor * data_size_in_gb * 1e6)/time_copy << " GB/s";
    std::cout << "\tTime: " << time_copy/1e6 << " s";
    std::cout << "\t\tTotal Error: " << error_stats.total_error;
    std::cout << "\tError Count: " << error_stats.error_count << std::endl;
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout 
            << "Usage: $./stream <size of each array> <optional: num threads>"
            << std::endl;
        return -1;
    }
    // if the user specified the number of threads to use, then override the
    // number of total_supported threads by the user specified number.
    // otherwise use the number of threads supported by the hardware.
    total_supported_threads = std::thread::hardware_concurrency();;
    if(argc == 3)
        // number of threads is also provided by the user
        total_supported_threads = atoi(argv[2]);

    // size of the array is supplied by the user. this has to be present.
    size_t array_size = (size_t)atol(argv[1]);

    // warn the user that the size of the array has to be a multiple of the
    // number of threads. I am too lazy to pad the arry to avoid this error.
    if(array_size % (size_t)total_supported_threads != 0)
        std::cout << "STREAM warn: Expect errors. The size of the array has"
                << " to be a multiple of the total number of threads!"
                << std::endl;

    // calculate total size of the array and print this info on the terminal.
    double data_size_in_gb = 
            array_size * sizeof(double) / 1024.0 / 1024.0 / 1024.0;
    std::cout << "STREAM info: Array Size : " << data_size_in_gb << " GB"
            << std::endl;
    std::cout << "STREAM info: Total Memory Occupied : " <<
            4 * data_size_in_gb << " GB" << std::endl;

    // allocate all the arrays.
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

    // setup threads here. sanity check for the number of threads. if not, then
    // crash the program
    assert(total_supported_threads != -1);
    std::cout << "STREAM info: Total Number of Threads : " <<
            total_supported_threads << std::endl;
    
    // end of overall stats print. this looks good.
    std::cout << std::endl;

    // now allocate memory for each of the threads
    std::thread* stream_threads = new std::thread[total_supported_threads];

    // warmup : do all the kernels without timing and in serial
    //
    std::cout << "STREAM info: NOT warming up!\n" << std::endl;
    //
    // idk how to do this. so skipping this for now.

    // setup some initial numbers for uniformity
    part = 0;
    barrier_variable_v2 = 0;
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        // index will be multiplied to find the next index
        stream_threads[i] = std::thread(kernel_copy, c, a, array_size, i);
    
    // setup initial timers for the actual kernel calls
    auto start = std::chrono::steady_clock::now();
#ifdef GEM5_ANNOTATION
    m5_work_begin(0,0);
#endif
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    
    // there has to be a barrier for the stats to be accurate.
    while(barrier_variable_v2 != total_supported_threads);
#ifdef GEM5_ANNOTATION
    m5_work_end(0,0);
#endif
    auto end = std::chrono::steady_clock::now();
    auto elapsed = 
        std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
        ).count();
    double time_copy = std::chrono::duration<double>(elapsed).count();

    // verify and print the stats of this kernel.
    stats_printer(
        "COPY", data_size_in_gb, time_copy, total_error(c, a, array_size));

    // -------------------------- end of copy kernel ----------------------- //
    
    // reset the part counter for the new kernel to run.
    part = 0;
    barrier_variable_v2 = 0;
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        stream_threads[i] = std::thread(
            kernel_scale, alpha, b, c, array_size, i);

    // setup the initial timers to keep a track of time.
    start = std::chrono::steady_clock::now();
#ifdef GEM5_ANNOTATION
    m5_work_begin(0,0);
#endif
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    while(barrier_variable_v2 != total_supported_threads);
#ifdef GEM5_ANNOTATION
    m5_work_end(0,0);
#endif
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
    
    // verify and print the stats of this kernel.
    stats_printer(
        "SCALE", data_size_in_gb, time_copy, total_error(
            expected_array, b, array_size
        )
    );
    delete expected_array;

    // ------------------------- end of scale kernel ----------------------- //
    
    // reset the part counter for the new kernel to run.
    part = 0;
    barrier_variable_v2 = 0;
    // create this new kernel
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        stream_threads[i] = std::thread(kernel_sum, c, a, b, array_size, i);

    // setup the initial timers to keep a track of time.
    start = std::chrono::steady_clock::now();
#ifdef GEM5_ANNOTATION
    m5_work_begin(0,0);
#endif
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    while(barrier_variable_v2 != total_supported_threads);
#ifdef GEM5_ANNOTATION
    m5_work_end(0,0);
#endif
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
    
    // verify and print the stats of this kernel.
    stats_printer(
        "SUM ", data_size_in_gb, time_copy, total_error(
            expected_array, c, array_size
        )
    );
    delete expected_array;

    // -------------------------- end of sum kernel ------------------------ //

    // reset the part counter for the new kernel to run.
    part = 0;
    barrier_variable_v2 = 0;
    // create this new kernel
    for(unsigned int i = 0 ; i < total_supported_threads ; i++)
        stream_threads[i] =
            std::thread(kernel_triad, alpha, a, b, c, array_size, i);

    // setup the initial timers to keep a track of time.
    start = std::chrono::steady_clock::now();
#ifdef GEM5_ANNOTATION
    m5_work_begin(0,0);
#endif
    for(unsigned int i = 0 ; i < total_supported_threads ; i++) 
        stream_threads[i].join();
    // there has to be a barrier for the stats to be accurate.
    while(barrier_variable_v2 != total_supported_threads);
#ifdef GEM5_ANNOTATION
    m5_work_end(0,0);
#endif
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
    // verify and print the stats of this kernel.
    stats_printer(
        "TRIAD", data_size_in_gb, time_copy, total_error(
            expected_array, a, array_size
        )
    );
    delete expected_array;

    // -------------------------- end of triad kernel ---------------------- //

    return 0;
}
