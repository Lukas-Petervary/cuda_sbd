#ifndef HAMILTONIAN_KERNEL_H
#define HAMILTONIAN_KERNEL_H

#include <vector>
#include <cstdint>
#include <cstdio>

namespace cuda_impl {
    // remove compile-time polymorphism with static
    //  compile type for cuda separate compilation
    using ElemT = double;

    struct Excitation {
        int braIdx, ketIdx;
        int i, j, a, b; // j unused for single
        int type;       // 0 = single, 1 = double
    };

    void LaunchDavidsonMatvecGPU(
            int rank,
            const std::vector<uint64_t>& dets,
            int detWords,

            const std::vector<Excitation>& excitations,

            const ElemT* h_one,
            const ElemT* h_two,
            int norbs,
            int bit_length,

            const std::vector<ElemT>& Tvec,
            std::vector<ElemT>& Wb
    );

}

#endif

