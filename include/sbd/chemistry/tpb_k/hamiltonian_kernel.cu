#include "hamiltonian_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256   // must be a multiple of WARP_SIZE (32)
#endif

#define WARP_SIZE 32

#define CUDA_CHECK(x)                                                    \
    try {                                                                \
        cudaError_t _err = (x);                                          \
        if (_err != cudaSuccess) {                                       \
            printf("CUDA error %s:%d  %s\n",                             \
                   __FILE__, __LINE__, cudaGetErrorString(_err));        \
            exit(_err);                                                  \
        }                                                                \
    } catch (...) { printf("CUDA t error: %s:%d\n", __FILE__, __LINE__); }

namespace cuda_impl {

    template<typename ElemT>
    void gpuContext_t<ElemT>::release() {
        for (auto p : d_row_ptr) if (p) cudaFree(p);
        for (auto p : d_exc)     if (p) cudaFree(p);

        d_row_ptr.clear(); d_exc.clear();

        if (d_dets) { cudaFree(d_dets); d_dets = nullptr; }
        if (d_one)  { cudaFree(d_one);  d_one  = nullptr; }
        if (d_two)  { cudaFree(d_two);  d_two  = nullptr; }
        if (d_T)    { cudaFree(d_T);    d_T    = nullptr; }
        if (d_Wb)   { cudaFree(d_Wb);   d_Wb   = nullptr; }
    }

    template<typename ElemT>
    void InitGpuContext(
            gpuContext_t<ElemT> &ctx,
            int rank_for_device,
            int mpi_rank_h,
            int mpi_size_h,
            const std::vector<uint64_t> &dets_flat,
            int detWords,
            const std::vector<std::vector<int>> &csr_row_ptr_per_task,
            const std::vector<std::vector<excitation_t>> &csr_exc_per_task,
            const ElemT *h_one,
            const size_t& h_one_size,
            const ElemT *h_two,
            const size_t& h_two_size,
            int norbs,
            int bit_length,
            size_t vecSize
    ) {
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        ctx.device = rank_for_device % numDevices;
        CUDA_CHECK(cudaSetDevice(ctx.device));

        ctx.detWords   = detWords;
        ctx.norbs      = norbs;
        ctx.bit_length = bit_length;
        ctx.mpi_rank_h = mpi_rank_h;
        ctx.mpi_size_h = mpi_size_h;
        ctx.vecSize    = vecSize;
        ctx.nTasks     = static_cast<int>(csr_row_ptr_per_task.size());
        ctx.nBras      = (ctx.nTasks > 0)
                         ? static_cast<int>(csr_row_ptr_per_task[0].size()) - 1
                         : 0;

        ctx.d_row_ptr.assign(ctx.nTasks, nullptr);
        ctx.d_exc.assign(ctx.nTasks, nullptr);

        for (int t = 0; t < ctx.nTasks; ++t) {
            const auto &row_ptr = csr_row_ptr_per_task[t];
            const auto &exc     = csr_exc_per_task[t];

            CUDA_CHECK(cudaMalloc(&ctx.d_row_ptr[t], row_ptr.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(ctx.d_row_ptr[t], row_ptr.data(),
                                  row_ptr.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            if (!exc.empty()) {
                CUDA_CHECK(cudaMalloc(&ctx.d_exc[t], exc.size() * sizeof(excitation_t)));
                CUDA_CHECK(cudaMemcpy(ctx.d_exc[t], exc.data(),
                                      exc.size() * sizeof(excitation_t),
                                      cudaMemcpyHostToDevice));
            }
        }

        CUDA_CHECK(cudaMalloc(&ctx.d_dets, dets_flat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(ctx.d_dets, dets_flat.data(), dets_flat.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ctx.d_one, h_one_size * sizeof(ElemT)));
        CUDA_CHECK(cudaMemcpy(ctx.d_one, h_one, h_one_size*sizeof(ElemT),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ctx.d_two, h_two_size * sizeof(ElemT)));
        CUDA_CHECK(cudaMemcpy(ctx.d_two, h_two, h_two_size * sizeof(ElemT),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ctx.d_T,  vecSize * sizeof(ElemT)));
        CUDA_CHECK(cudaMalloc(&ctx.d_Wb, vecSize * sizeof(ElemT)));
    }

    template<typename ElemT>
    void GPUKetVecH2D(gpuContext_t<ElemT> &ctx, const std::vector<ElemT> &Tvec) {
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaMemcpy(ctx.d_T, Tvec.data(),
                              ctx.vecSize * sizeof(ElemT),
                              cudaMemcpyHostToDevice));
    }

    template<typename ElemT>
    void GPUVecH2D(gpuContext_t<ElemT> &ctx, const std::vector<ElemT> &Wb) {
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaMemcpy(ctx.d_Wb, Wb.data(),
                              ctx.vecSize * sizeof(ElemT),
                              cudaMemcpyHostToDevice));
    }

    template<typename ElemT>
    void GPUVecD2H(gpuContext_t<ElemT> &ctx, std::vector<ElemT> &Wb) {
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaMemcpy(Wb.data(), ctx.d_Wb,
                              ctx.vecSize * sizeof(ElemT),
                              cudaMemcpyDeviceToHost));
    }

    template<typename ElemT>
    struct DeviceOneInt {
        const ElemT *data;
        int norbs;

        __device__ __forceinline__ ElemT operator()(int a, int i) const {
            return data[a * norbs + i];
        }
    };

    template<typename ElemT>
    struct DeviceTwoInt {
        const ElemT *data;
        int norbs;

        __device__ __forceinline__ ElemT operator()(int i, int j, int k, int l) const {
            if (!((i % 2 == j % 2) && (k % 2 == l % 2))) return ElemT(0);
            const int I = i / 2, J = j / 2;
            const int K = k / 2, L = l / 2;
            const int ij = max(I, J) * (max(I, J) + 1) / 2 + min(I, J);
            const int kl = max(K, L) * (max(K, L) + 1) / 2 + min(K, L);
            const int a = max(ij, kl);
            const int b = min(ij, kl);
            return data[a * (a + 1) / 2 + b];
        }
    };

    __device__ __forceinline__
    double parity(
            const uint64_t *det,
            int bit_length,
            int start,
            int end
    ) {
        double sgn = 1.0;

        const int blockStart0 = start / bit_length;
        const int bitStart = start % bit_length;

        const int blockEnd = end / bit_length;
        const int bitEnd = end % bit_length;

        int nonZeroBits = 0;
        int blockStart = blockStart0;

        if (blockStart == blockEnd) {
            uint64_t mask = ((1ULL << bitEnd) - 1ULL) ^ ((1ULL << bitStart) - 1ULL);
            nonZeroBits += __popcll(det[blockStart] & mask);
        } else {
            if (bitStart != 0) {
                uint64_t mask = ~((1ULL << bitStart) - 1ULL);
                nonZeroBits += __popcll(det[blockStart] & mask);
                ++blockStart;
            }

            for (int i = blockStart; i < blockEnd; ++i)
                nonZeroBits += __popcll(det[i]);

            if (bitEnd != 0) {
                uint64_t mask = (1ULL << bitEnd) - 1ULL;
                nonZeroBits += __popcll(det[blockEnd] & mask);
            }
        }

        sgn *= (nonZeroBits & 1) ? -1.0 : 1.0;

        if ((det[start / bit_length] >> (start % bit_length)) & 1ULL)
            sgn *= -1.0;
        return sgn;
    }

    __device__ __forceinline__
    double gpuOneExcite(
            const uint64_t *det,
            int bit_length,
            int detWords,
            int i,
            int a,
            DeviceOneInt<double> h1,
            DeviceTwoInt<double> h2) {
        double sgn = parity(det, bit_length, min(i, a), max(i, a));
        double energy = h1(a, i);

        for (int w = 0; w < detWords; ++w) {
            uint64_t bits = det[w];

            while (bits) {
                int pos = __ffsll(bits) - 1;
                int j = w * bit_length + pos;

                energy += h2(a, i, j, j) - h2(a, j, j, i);

                bits &= bits - 1;
            }
        }

        return energy * sgn;
    }

    __device__ __forceinline__
    double gpuTwoExcite(
            const uint64_t *det,
            int bit_length,
            int i,
            int j,
            int a,
            int b,
            DeviceTwoInt<double> h2
    ) {
        double sgn = 1.0;

        const int I = min(i, j);
        const int J = max(i, j);

        const int A = min(a, b);
        const int B = max(a, b);

        sgn *= parity(det, bit_length, min(I, A), max(I, A));
        sgn *= parity(det, bit_length, min(J, B), max(J, B));

        if (A > J || B < I)
            sgn *= -1.0;
        return sgn * (h2(A, I, B, J) - h2(A, J, B, I));
    }

    template<typename ElemT>
    __global__ void csr_matvec_kernel(
            const int *row_ptr,
            const excitation_t *exc,
            const uint64_t *dets,
            const ElemT *Tvec,
            ElemT *Wb,
            DeviceOneInt<ElemT> h1,
            DeviceTwoInt<ElemT> h2,
            int bit_length,
            int detWords,
            int nBras,
            int mpi_size_h,
            int mpi_rank_h
    ) {
        const int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
        const int warpId = globalThread / WARP_SIZE;
        const int laneId = globalThread % WARP_SIZE;

        if (warpId >= nBras) return;

        const int braIdx = warpId;

        if ((braIdx % mpi_size_h) != mpi_rank_h) return;

        const int start = row_ptr[braIdx];
        const int end = row_ptr[braIdx + 1];
        if (start == end) return;

        const uint64_t *det = dets + static_cast<ptrdiff_t>(braIdx) * detWords;

        ElemT lane_sum = ElemT(0);
        for (int k = start + laneId; k < end; k += WARP_SIZE) {
            const excitation_t e = exc[k];
            ElemT val;

            if (e.type == 0)
                val = gpuOneExcite(det, bit_length, detWords, e.i, e.a, h1, h2);
            else
                val = gpuTwoExcite(det, bit_length, e.i, e.j, e.a, e.b, h2);

            lane_sum += val * Tvec[e.ketIdx];
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, offset);

        if (laneId == 0)
            atomicAdd(&Wb[braIdx], lane_sum);
    }

    template<typename ElemT>
    void GPUSpMV(gpuContext_t<ElemT> &ctx, int taskIdx) {
        CUDA_CHECK(cudaSetDevice(ctx.device));

        if (ctx.d_exc[taskIdx] == nullptr) return;

        DeviceOneInt<ElemT> dH1{ctx.d_one, ctx.norbs};
        DeviceTwoInt<ElemT> dH2{ctx.d_two, ctx.norbs};

        const int totalThreads = ctx.nBras * WARP_SIZE;
        const int grid = (totalThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        csr_matvec_kernel<ElemT><<<grid, BLOCK_SIZE>>>(
                ctx.d_row_ptr[taskIdx],
                ctx.d_exc[taskIdx],
                ctx.d_dets,
                ctx.d_T,
                ctx.d_Wb,
                dH1,
                dH2,
                ctx.bit_length,
                ctx.detWords,
                ctx.nBras,
                ctx.mpi_size_h,
                ctx.mpi_rank_h
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    }

} // namespace cuda_impl

// explicit instantiation of double template for separable compilation
namespace cuda_impl {
    template void gpuContext_t<double>::release();

    template void InitGpuContext<double>(
            gpuContext_t<double> &,
            int, int, int,
            const std::vector<uint64_t> &,
            int,
            const std::vector<std::vector<int>> &,
            const std::vector<std::vector<excitation_t>> &,
            const double *,
            const size_t&,
            const double *,
            const size_t&,
            int,
            int,
            size_t
    );

    template void GPUKetVecH2D<double>(gpuContext_t<double> &, const std::vector<double> &);
    template void GPUVecH2D<double>(gpuContext_t<double> &, const std::vector<double> &);
    template void GPUVecD2H<double>(gpuContext_t<double> &, std::vector<double> &);
    template void GPUSpMV<double>(gpuContext_t<double> &, int);
}
