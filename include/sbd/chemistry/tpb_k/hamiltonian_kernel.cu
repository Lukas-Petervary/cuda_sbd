#include "hamiltonian_kernel.h"
#include <cuda_runtime.h>

// ----------------------------------------------------------------
// Compile-time tuning knobs
// ----------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256   // must be a multiple of WARP_SIZE (32)
#endif

#define WARP_SIZE 32

#define CUDA_CHECK(x)                                                    \
    do {                                                                 \
        cudaError_t _err = (x);                                          \
        if (_err != cudaSuccess) {                                       \
            printf("CUDA error %s:%d  %s\n",                             \
                   __FILE__, __LINE__, cudaGetErrorString(_err));        \
            exit(_err);                                                  \
        }                                                                \
    } while (0)

namespace cuda_impl {

    // ================================================================
    // Context lifecycle
    // ================================================================

    template<typename ElemT>
    void DavidsonGpuContext<ElemT>::release() {
        if (d_row_ptr)  { cudaFree(d_row_ptr);  d_row_ptr = nullptr;}
        if (d_exc)      { cudaFree(d_exc);      d_exc = nullptr;    }
        if (d_dets)     { cudaFree(d_dets);     d_dets = nullptr;   }
        if (d_one)      { cudaFree(d_one);      d_one = nullptr;    }
        if (d_two)      { cudaFree(d_two);      d_two = nullptr;    }
        if (d_T)        { cudaFree(d_T);        d_T = nullptr;      }
        if (d_Wb)       { cudaFree(d_Wb);       d_Wb = nullptr;     }
    }

    // ================================================================
    // GPU initialization
    // Uploads CSR structure, bra determinants, and integral tables.
    // All persistent data lives on the device for the Davidson lifetime.
    // ================================================================

    template<typename ElemT>
    void InitializeDavidsonGPU(
            DavidsonGpuContext<ElemT> &ctx,
            int rank,
            const std::vector<uint64_t> &dets_flat,
            int detWords,
            const std::vector<int> &csr_row_ptr,
            const std::vector<excitation_t> &csr_exc,
            const ElemT *h_one,
            const ElemT *h_two,
            int norbs,
            int bit_length,
            size_t vecSize
    ) {
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        ctx.device = rank % numDevices;
        CUDA_CHECK(cudaSetDevice(ctx.device));

        ctx.detWords = detWords;
        ctx.norbs = norbs;
        ctx.bit_length = bit_length;
        ctx.nBras = static_cast<int>(csr_row_ptr.size()) - 1;
        ctx.nExc = static_cast<int>(csr_exc.size());
        ctx.vecSize = vecSize;

        // Integral table sizes (chemist 8-fold symmetry packing)
        const size_t nPairs = static_cast<size_t>(norbs) * (norbs + 1) / 2;
        const size_t twoSize = nPairs * (nPairs + 1) / 2;

        // ---- CSR row pointers ----
        CUDA_CHECK(cudaMalloc(&ctx.d_row_ptr,
                              csr_row_ptr.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(ctx.d_row_ptr, csr_row_ptr.data(),
                              csr_row_ptr.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        // ---- CSR excitation data ----
        CUDA_CHECK(cudaMalloc(&ctx.d_exc,
                              csr_exc.size() * sizeof(excitation_t)));
        CUDA_CHECK(cudaMemcpy(ctx.d_exc, csr_exc.data(),
                              csr_exc.size() * sizeof(excitation_t),
                              cudaMemcpyHostToDevice));

        // ---- Bra determinant bitstrings ----
        CUDA_CHECK(cudaMalloc(&ctx.d_dets,
                              dets_flat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(ctx.d_dets, dets_flat.data(),
                              dets_flat.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        // ---- One-body integrals ----
        CUDA_CHECK(cudaMalloc(&ctx.d_one,
                              static_cast<size_t>(norbs) * norbs * sizeof(ElemT)));
        CUDA_CHECK(cudaMemcpy(ctx.d_one, h_one,
                              static_cast<size_t>(norbs) * norbs * sizeof(ElemT),
                              cudaMemcpyHostToDevice));

        // ---- Two-body integrals ----
        CUDA_CHECK(cudaMalloc(&ctx.d_two, twoSize * sizeof(ElemT)));
        CUDA_CHECK(cudaMemcpy(ctx.d_two, h_two,
                              twoSize * sizeof(ElemT),
                              cudaMemcpyHostToDevice));

        // ---- Working vectors ----
        CUDA_CHECK(cudaMalloc(&ctx.d_T, vecSize * sizeof(ElemT)));
        CUDA_CHECK(cudaMalloc(&ctx.d_Wb, vecSize * sizeof(ElemT)));
    }

    // ================================================================
    // Device accessor helpers (unchanged from original)
    // ================================================================

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

        sgn *= parity(
                det,
                bit_length,
                min(I, A),
                max(I, A)
        );

        sgn *= parity(
                det,
                bit_length,
                min(J, B),
                max(J, B)
        );

        if (A > J || B < I)
            sgn *= -1.0;
        return sgn * (h2(A, I, B, J) - h2(A, J, B, I));
    }

    // ================================================================
    // CSR warp-per-row matvec kernel
    //
    // Dispatch:  one warp (32 threads) per bra row.
    // Work:      each lane strides through exc[ row_ptr[bra] .. row_ptr[bra+1] )
    //            with stride WARP_SIZE, accumulating a partial sum.
    //
    // Reduction: warp-level tree reduce via __shfl_down_sync.
    //            Lane 0 writes the final sum with one atomicAdd.
    //
    // Memory access characteristics versus the old COO kernel:
    //   - exc[] reads are coalesced: adjacent lanes read adjacent entries.
    //   - The bra determinant pointer is warp-uniform → broadcast from L1,
    //     single transaction per detWords word across all excitation types.
    //   - Atomic pressure reduced from one-per-excitation to one-per-bra.
    //   - No intra-warp write conflicts: all lanes accumulate into a
    //     register before the final reduction.
    //
    // Load balance note:
    //   Warp-per-row works well when row lengths are O(WARP_SIZE).
    //   If your active space produces highly variable row lengths you may
    //   want to layer in a block-per-row path for long rows (>256) and a
    //   thread-per-row path for very short rows (<4).  The segmented CSR
    //   / merge-path approach (Bell & Garland 2012) is the principled
    //   fix, but warp-per-row is a large improvement over COO for the
    //   typical CI excitation density.
    //
    // Requires compute capability >= 6.0 for double atomicAdd.
    // ================================================================

    template<typename ElemT>
    __global__ void csr_matvec_kernel(
            const int *row_ptr,   // [nBras + 1]
            const excitation_t *exc,       // [nExc]
            const uint64_t *dets,      // [nBras * detWords]
            const ElemT *Tvec,
            ElemT *Wb,
            DeviceOneInt<ElemT> h1,
            DeviceTwoInt<ElemT> h2,
            int bit_length,
            int detWords,
            int nBras
    ) {
        // ---- warp / lane ids ----
        const int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
        const int warpId = globalThread / WARP_SIZE;
        const int laneId = globalThread % WARP_SIZE;

        if (warpId >= nBras) return;

        const int braIdx = warpId;
        const int start = row_ptr[braIdx];
        const int end = row_ptr[braIdx + 1];

        // Warp-uniform pointer: all 32 lanes reference the same bra det.
        // CUDA issues a single memory transaction (warp-uniform load / L1 broadcast).
        const uint64_t *det = dets + static_cast<ptrdiff_t>(braIdx) * detWords;

        ElemT lane_sum = ElemT(0);

        // Stride over this bra's excitations with WARP_SIZE to keep
        // consecutive lanes reading consecutive ExcitationData entries.
        for (int k = start + laneId; k < end; k += WARP_SIZE) {
            const excitation_t e = exc[k];
            ElemT val;

            if (e.type == 0)
                val = gpuOneExcite(det, bit_length, detWords, e.i, e.a, h1, h2);
            else
                val = gpuTwoExcite(det, bit_length, e.i, e.j, e.a, e.b, h2);

            lane_sum += val * Tvec[e.ketIdx];
        }

        // ---- warp-level tree reduction ----
        // Reduces lane_sum across all 32 lanes into lane 0.
        // No shared memory required; register-only.
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, offset);

        // ---- single atomic write per bra (lane 0 only) ----
        // Down from one atomicAdd per excitation in the COO kernel.
        if (laneId == 0)
            atomicAdd(&Wb[braIdx], lane_sum);
    }

    // ================================================================
    // Host-side matvec driver
    // ================================================================

    template<typename ElemT>
    void DavidsonMatvecGPU(
            DavidsonGpuContext<ElemT> &ctx,
            const std::vector<ElemT> &Tvec,
            std::vector<ElemT> &Wb
    ) {
        CUDA_CHECK(cudaSetDevice(ctx.device));

        // Upload ket vector.
        // Bug fix: the prior version passed vec_capacity (element count)
        // where cudaMemcpy expects bytes → silent under-copy for ElemT=double.
        CUDA_CHECK(cudaMemcpy(ctx.d_T, Tvec.data(),
                              ctx.vecSize * sizeof(ElemT),
                              cudaMemcpyHostToDevice));

        // Zero device output buffer.
        // Bug fix: same sizeof(ElemT) factor was missing in cudaMemset.
        CUDA_CHECK(cudaMemset(ctx.d_Wb, 0, ctx.vecSize * sizeof(ElemT)));

        DeviceOneInt<ElemT> dH1{ctx.d_one, ctx.norbs};
        DeviceTwoInt<ElemT> dH2{ctx.d_two, ctx.norbs};

        // Grid: one warp per bra row.
        // totalThreads = nBras * WARP_SIZE; round up to BLOCK_SIZE boundary.
        const int totalThreads = ctx.nBras * WARP_SIZE;
        const int grid = (totalThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        csr_matvec_kernel<ElemT><<<grid, BLOCK_SIZE>>>(
                ctx.d_row_ptr,
                ctx.d_exc,
                ctx.d_dets,
                ctx.d_T,
                ctx.d_Wb,
                dH1,
                dH2,
                ctx.bit_length,
                ctx.detWords,
                ctx.nBras
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Download and ACCUMULATE into host Wb.
        //
        // Bug fix: the prior version overwrote Wb, discarding any CPU
        // diagonal contribution that gpuMult added before this call.
        // Using += preserves the diagonal term for mpi_rank_t == 0
        // and is a no-op for other ranks (where Wb[i] was 0 before the
        // diagonal add).
        //
        // Note: gpuMult currently calls this once per task iteration in
        // a loop while also sliding T.  Since ALL excitations are baked
        // into the context at initialization time (using the initial T
        // layout), calling the kernel with a post-slide T produces
        // incorrect results for tasks beyond the first.  For single-task
        // deployments this is fine.  For multi-task multi-node use,
        // either (a) call davidsonMatvecGPU once before the slide loop
        // and skip the GPU path inside the loop, or (b) partition the
        // CSR matrix per-task and process each slide independently.
        std::vector<ElemT> gpu_result(ctx.vecSize);
        CUDA_CHECK(cudaMemcpy(gpu_result.data(), ctx.d_Wb,
                              ctx.vecSize * sizeof(ElemT),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < ctx.vecSize; ++i)
            Wb[i] += gpu_result[i];
    }


}

// explicit instantiation of double template for separable compilation
namespace cuda_impl {
    template void DavidsonGpuContext<double>::release();

    template void InitializeDavidsonGPU<double>(
            DavidsonGpuContext<double> &,
            int,
            const std::vector<uint64_t> &,
            int,
            const std::vector<int> &,
            const std::vector<excitation_t> &,
            const double *,
            const double *,
            int,
            int,
            size_t
    );

    template void DavidsonMatvecGPU<double>(
            DavidsonGpuContext<double> &,
            const std::vector<double> &,
            std::vector<double> &
    );
}