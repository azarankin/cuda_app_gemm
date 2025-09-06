#pragma once
#include <cuda_runtime.h>
#include <nvToolsExt.h>
//#include "third_party/NVTX/nvtx3.hpp"
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>
#include <stack>
#include <unordered_map>
#include <cstdio>


class CudaErrorHandler {
public:
    void operator<<(cudaError_t status) const {
        if (status != cudaSuccess) {
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(status) << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    void operator<<(cublasStatus_t status) const {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "[CUBLAS ERROR] " << cublasGetStatusString(status) << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    void operator()() const {
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            std::cerr << "[CUDA KERNEL or CUDA_CHECK() ERROR] " << cudaGetErrorString(status) << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

};

// שימוש חיצוני
static const CudaErrorHandler CUDA_CHECK;

//CUDA_CHECK << 







#define PROFILE_REPEAT(STMT)                                \
    do {                                                    \
        for (int _pr_i = 0; _pr_i < 5; ++_pr_i) {         \
            STMT;                                           \
        }                                                   \
    } while (0)








class CudaTimer {
    struct Timer {
        cudaEvent_t start{}, stop{};
        std::vector<float> timings;

        Timer() {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }

        ~Timer() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        void begin(cudaStream_t stream = 0) {
            cudaEventRecord(start, stream);
        }

        void end(cudaStream_t stream = 0) {
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            timings.push_back(ms);
        }

        void print(const std::string& label) const {
            if (timings.empty()) return;
            float total = 0;
            for (float t : timings)
                total += t;
            float avg = total / timings.size();
            std::printf("[CudaTimer] %s: %zu runs, avg %.3f ms\n",
                        label.c_str(), timings.size(), avg);
        }

        
    };

    struct Frame { std::string name; cudaStream_t stream; };

    inline static thread_local std::stack<Frame> frames{};

    inline static std::unordered_map<std::string, Timer> timers;


public:
    static void begin(const std::string& name, cudaStream_t stream = 0) {
        frames.push(Frame{name, stream});
        timers[name].begin(stream);
    }

    static void end(const std::string& name, cudaStream_t stream = 0) {
        if (frames.empty())
            throw std::logic_error("Profiler::end() called with empty stack");
        timers[name].end(stream);
        frames.pop();
    }

    static void end(cudaStream_t stream) {
        if (frames.empty())
            return;
        const Frame& f = frames.top();
        end(f.name, stream);
    }

    static void end() {
        if (frames.empty())
            return;
        const Frame& f = frames.top();
        end(f.name, f.stream);
    }

    static void printAll() {
        std::puts("");
        for (const auto& [name, timer] : timers)
            timer.print(name);
        std::puts("");
        resetAll();
    }

    static void resetAll() {
        timers.clear();
    }
};





enum class NvtxColor : uint32_t 
{
    Red     = 0xFFFF0000,
    Green   = 0xFF00FF00,
    Blue    = 0xFF0000FF,
    Yellow  = 0xFFFFFF00,
    Purple  = 0xFFFF00FF,
    Cyan    = 0xFF00FFFF,
    Orange  = 0xFFFFA500,
    White   = 0xFFFFFFFF,
    Gray    = 0xFF888888,
    Default = 0xFF0000FF
};

#define PROFILE_RANGE(name, color_enum, ...)                       \
    do {                                                           \
        nvtxEventAttributes_t eventAttrib = {0};                   \
        eventAttrib.version = NVTX_VERSION;                        \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;          \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                   \
        eventAttrib.color = static_cast<uint32_t>(color_enum);     \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;         \
        eventAttrib.message.ascii = name;                          \
        CudaTimer::begin(name);                                    \
        nvtxRangePushEx(&eventAttrib);                             \
        __VA_ARGS__;                                               \
        cudaDeviceSynchronize();                                   \
        nvtxRangePop();                                            \
        CudaTimer::end();                                          \
    } while (0)

    

#define PROFILE_RANGE_PRINT()                                      \
        do {                                                       \
            CudaTimer::PRINT()                                     \
            } while (0)

// PROFILE_RANGE("GEMM", NvtxColor::Red, 
//     gemm_basic_naive_kernel<<<1, threads>>>(d_A, d_B, d_C, M, N, K);
// ); // PROFILE_RANGE

struct CudaProfiler {
    static inline thread_local bool isRunning = false;
    static inline thread_local bool isStoped = false;

    static inline void BEGIN() {
        if (!isRunning && !isStoped) {
            cudaProfilerStart();
            isRunning = true;
        }
    }

    static inline void END() {
        if (isRunning && !isStoped) {
            cudaProfilerStop();
            isRunning = false;
            isStoped = true;
        }
    }
};

//CudaProfiler::BEGIN();
//..
//CudaProfiler::END();

