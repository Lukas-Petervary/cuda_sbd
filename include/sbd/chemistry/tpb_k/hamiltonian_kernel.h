#ifndef HAMILTONIAN_KERNEL_H
#define HAMILTONIAN_KERNEL_H

#include <vector>
#include <cstdint>
#include <cstdio>

namespace cuda_impl {
    struct excitation_t {
        int ketIdx;
        int i, j;   // annihilation orbital indices; j = 0 for singles
        int a, b;   // creation     orbital indices; b = 0 for singles
        int type;   // 0 = single, 1 = double / mixed
    };

    template<typename ElemT>
    struct gpuContext_t {
        int device = 0;
        int nTasks = 0;

        // CSR excitation structure
        std::vector<int*>          d_row_ptr;
        std::vector<excitation_t*> d_exc;

        // Bra determinant bitstrings, row-major: [nBras][detWords]
        uint64_t* d_dets = nullptr;

        // Integral tables
        ElemT* d_one = nullptr; size_t d_one_size = 0;
        ElemT* d_two = nullptr; size_t d_two_size = 0;

        // Working vectors
        ElemT* d_T  = nullptr;
        ElemT* d_Wb = nullptr;

        int    detWords   = 0;
        int    norbs      = 0;
        int    bit_length = 0;
        int    nBras      = 0;
        size_t vecSize    = 0;

        int mpi_size_h = 1;
        int mpi_rank_h = 0;

        ~gpuContext_t() { release(); }
        void release();
    };

    template<typename ElemT>
    void InitGpuContext(
            gpuContext_t<ElemT>&                            ctx,
            int                                             rank_for_device,
            int                                             mpi_rank_h,
            int                                             mpi_size_h,
            const std::vector<uint64_t>&                    dets_flat,
            int                                             detWords,
            const std::vector<std::vector<int>>&            csr_row_ptr_per_task,
            const std::vector<std::vector<excitation_t>>&   csr_exc_per_task,
            const ElemT*                                    h_one,
            const size_t&                                   h_one_size,
            const ElemT*                                    h_two,
            const size_t&                                   h_two_size,
            int                                             norbs,
            int                                             bit_length,
            size_t                                          vecSize
    );

    template<typename ElemT>
    void GPUKetVecH2D(gpuContext_t<ElemT> &ctx, const std::vector<ElemT> &Tvec);

    template<typename ElemT>
    void GPUVecH2D(gpuContext_t<ElemT> &ctx, const std::vector<ElemT> &Wb);

    template<typename ElemT>
    void GPUVecD2H(gpuContext_t<ElemT> &ctx, std::vector<ElemT> & Wb);

    template<typename ElemT>
    void GPUSpMV(gpuContext_t<ElemT> &ctx, int taskIdx);
} // namespace cuda_impl

#endif // HAMILTONIAN_KERNEL_H
