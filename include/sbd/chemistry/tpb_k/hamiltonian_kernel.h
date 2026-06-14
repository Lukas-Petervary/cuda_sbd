#ifndef HAMILTONIAN_KERNEL_H
#define HAMILTONIAN_KERNEL_H

#include <vector>
#include <cstdint>
#include <cstdio>

namespace cuda_impl {

    // ----------------------------------------------------------------
    // ExcitationData: per-excitation payload with braIdx removed.
    //
    // In CSR layout, the bra index is IMPLICIT: all excitations for
    // bra[i] live in csr_exc[ row_ptr[i] .. row_ptr[i+1] ).
    // This eliminates the redundant integer stored per-entry in the
    // old COO Excitation struct and enables warp-level reduction
    // on the kernel side (one warp owns one bra row → no intra-warp
    // atomics, one atomic per row at most).
    //
    // Field semantics mirror the original Excitation struct exactly:
    //   single (type=0): i = annihilator, a = creator, j=b=0
    //   double (type=1): i,j = annihilators, a,b = creators
    //   mixed  (type=1): i = alpha annihilator, j = beta annihilator,
    //                    a = alpha creator,     b = beta creator
    // ----------------------------------------------------------------
    struct excitation_t {
        int ketIdx;
        int i, j;   // annihilation orbital indices; j = 0 for singles
        int a, b;   // creation     orbital indices; b = 0 for singles
        int type;   // 0 = single, 1 = double / mixed
    };

    // ----------------------------------------------------------------
    // DavidsonGpuContext: persistent GPU state for one MPI rank.
    //
    // CSR arrays replace the flat COO excitation array:
    //   d_row_ptr[braIdx]     → first excitation index for that bra
    //   d_row_ptr[braIdx + 1] → one-past-last excitation index
    //   d_exc[k]              → ExcitationData for excitation k
    //
    // All sizes are in elements, not bytes.  The *sizeof(ElemT) factor
    // was absent in the prior version and caused silent under-copies.
    // ----------------------------------------------------------------
    template<typename ElemT>
    struct DavidsonGpuContext {
        int device = 0;

        // CSR excitation structure
        int*          d_row_ptr = nullptr;  // [nBras + 1]
        excitation_t* d_exc     = nullptr;  // [nExc]

        // Bra determinant bitstrings, row-major: [nBras][detWords]
        uint64_t* d_dets = nullptr;

        // Integral tables
        ElemT* d_one = nullptr;   // [norbs * norbs]
        ElemT* d_two = nullptr;   // [nPairs * (nPairs+1)/2] (chemist symmetry)

        // Working vectors
        ElemT* d_T  = nullptr;    // ket vector   [vecSize]
        ElemT* d_Wb = nullptr;    // bra result   [vecSize]

        int    detWords   = 0;
        int    norbs      = 0;
        int    bit_length = 0;
        int    nBras      = 0;    // number of distinct bra determinants
        int    nExc       = 0;    // total off-diagonal excitations
        size_t vecSize    = 0;    // wave-function vector length (elements)

        ~DavidsonGpuContext() { release(); }
        void release();
    };

    // ----------------------------------------------------------------
    // GPU entry points (separately compiled in hamiltonian_kernel.cu)
    // ----------------------------------------------------------------

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

    // Accumulates GPU off-diagonal H*T result INTO Wb (does not overwrite).
    // The caller is responsible for zeroing Wb before the first contribution.
    template<typename ElemT>
    void DavidsonMatvecGPU(
            DavidsonGpuContext<ElemT>&  ctx,
            const std::vector<ElemT>&   Tvec,
            std::vector<ElemT>&         Wb
    );

} // namespace cuda_impl

#endif // HAMILTONIAN_KERNEL_H