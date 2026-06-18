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
    struct DavidsonGpuContext {
        int device = 0;

        // CSR excitation structure
        int*          d_row_ptr = nullptr;
        excitation_t* d_exc     = nullptr;

        // Bra determinant bitstrings, row-major: [nBras][detWords]
        uint64_t* d_dets = nullptr;

        // Integral tables
        ElemT* d_one = nullptr;
        ElemT* d_two = nullptr;

        // Working vectors
        ElemT* d_T  = nullptr;
        ElemT* d_Wb = nullptr;

        int    detWords   = 0;
        int    norbs      = 0;
        int    bit_length = 0;
        int    nBras      = 0;
        int    nExc       = 0;
        size_t vecSize    = 0;

        ~DavidsonGpuContext() { release(); }
        void release();
    };

    template<typename ElemT>
    void InitializeDavidsonGPU(
            DavidsonGpuContext<ElemT>&          ctx,
            int                                 rank,
            const std::vector<uint64_t>&        dets_flat,
            int                                 detWords,
            const std::vector<int>&             csr_row_ptr,
            const std::vector<excitation_t>&  csr_exc,
            const ElemT*                        h_one,
            const ElemT*                        h_two,
            int                                 norbs,
            int                                 bit_length,
            size_t                              vecSize
    );

    template<typename ElemT>
    void DavidsonMatvecGPU(
            DavidsonGpuContext<ElemT>&  ctx,
            const std::vector<ElemT>&   Tvec,
            std::vector<ElemT>&         Wb
    );

} // namespace cuda_impl

#endif // HAMILTONIAN_KERNEL_H