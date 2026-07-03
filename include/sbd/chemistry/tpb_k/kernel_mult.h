#ifndef KERNEL_MULT_H
#define KERNEL_MULT_H

#include <vector>
#include <cstdint>
#include <numeric>
#include <cassert>

#include "../basic/determinants.h"
#include "../tpb/helper.h"
#include "hamiltonian_kernel.h"

namespace cuda_impl {
    template<typename ElemT>
    void BuildCsrExcitationsPerTask(
            const std::vector<std::vector<size_t>>& adets,
            const std::vector<std::vector<size_t>>& bdets,
            const std::vector<sbd::TaskHelpers>&    helper,
            size_t                                  bit_length,
            size_t                                  norbs,
            std::vector<uint64_t>&                  dets_flat,
            std::vector<std::vector<int>>&          task_row_vec,
            std::vector<std::vector<excitation_t>>& task_exc_vec,
            int&                                    detWords
    ) {
        detWords = static_cast<int>((2 * norbs + bit_length - 1) / bit_length);

        task_row_vec.clear();
        task_exc_vec.clear();
        dets_flat.clear();

        if (helper.empty()) return;

        const size_t nTasks = helper.size();

        const size_t braAlphaStart = helper[0].braAlphaStart;
        const size_t braAlphaEnd   = helper[0].braAlphaEnd;
        const size_t braBetaStart  = helper[0].braBetaStart;
        const size_t braBetaEnd    = helper[0].braBetaEnd;
        const size_t braAlphaSize  = braAlphaEnd - braAlphaStart;
        const size_t braBetaSize   = braBetaEnd  - braBetaStart;
        const size_t totalBras     = braAlphaSize * braBetaSize;

        dets_flat.resize(totalBras * static_cast<size_t>(detWords));
        #pragma omp parallel
        {
            std::vector<uint64_t> DetI(static_cast<size_t>(detWords));
            #pragma omp for schedule(dynamic)
            for (size_t bIdx = 0; bIdx < totalBras; ++bIdx) {
                const size_t ia = braAlphaStart + bIdx / braBetaSize;
                const size_t ib = braBetaStart  + bIdx % braBetaSize;
                sbd::DetFromAlphaBeta(adets[ia], bdets[ib], bit_length, norbs, DetI);
                const size_t detOff = bIdx * static_cast<size_t>(detWords);
                for (int w = 0; w < detWords; ++w)
                    dets_flat[detOff + w] = DetI[w];
            }
        }

        task_row_vec.resize(nTasks);
        task_exc_vec.resize(nTasks);

        for (size_t t = 0; t < nTasks; ++t) {
            const sbd::TaskHelpers& h = helper[t];

            assert(h.braAlphaStart == braAlphaStart && h.braAlphaEnd == braAlphaEnd &&
                   h.braBetaStart  == braBetaStart  && h.braBetaEnd  == braBetaEnd &&
                   "BuildCsrExcitationsPerTask: every task must share the same bra "
                   "block, same assumption CPU mult() relies on for braIdx addressing.");

            auto& row_ptr = task_row_vec[t];
            auto& exc     = task_exc_vec[t];
            row_ptr.assign(totalBras + 1, 0);

            const size_t ketBetaSize = h.ketBetaEnd - h.ketBetaStart;

            #pragma omp parallel for schedule(dynamic)
            for (size_t bIdx = 0; bIdx < totalBras; ++bIdx) {
                const size_t ia = braAlphaStart + bIdx / braBetaSize;
                const size_t ib = braBetaStart  + bIdx % braBetaSize;
                const size_t ioff = ia - h.braAlphaStart;
                const size_t boff = ib - h.braBetaStart;

                int cnt = static_cast<int>(
                        h.taskType == 2 ? h.SinglesFromAlphaLen[ioff] + h.DoublesFromAlphaLen[ioff] :
                        h.taskType == 1 ? h.SinglesFromBetaLen[boff] + h.DoublesFromBetaLen[boff] :
                        h.SinglesFromAlphaLen[ioff] * h.SinglesFromBetaLen[boff]
                );
                row_ptr[bIdx + 1] = cnt;
            }
            for (size_t i = 0; i < totalBras; ++i)
                row_ptr[i + 1] += row_ptr[i];

            exc.resize(static_cast<size_t>(row_ptr[totalBras]));

            #pragma omp parallel for schedule(dynamic)
            for (size_t bIdx = 0; bIdx < totalBras; ++bIdx) {
                const size_t ia = braAlphaStart + bIdx / braBetaSize;
                const size_t ib = braBetaStart  + bIdx % braBetaSize;
                const size_t ioff = ia - h.braAlphaStart;
                const size_t boff = ib - h.braBetaStart;

                int wp = row_ptr[bIdx];

                if (h.taskType == 2) {
                    // ---- single alpha ----
                    for (size_t j = 0; j < h.SinglesFromAlphaLen[ioff]; ++j) {
                        const size_t ja   = h.SinglesFromAlphaSM[ioff][j];
                        const int    kIdx = static_cast<int>(
                                (ja - h.ketAlphaStart) * ketBetaSize
                                + (ib - h.ketBetaStart));
                        const int* cr = h.SinglesAlphaCrAnSM[ioff];
                        exc[wp++] = { kIdx, cr[2*j+0], 0, cr[2*j+1], 0, 0 };
                    }
                    // ---- double alpha ----
                    for (size_t j = 0; j < h.DoublesFromAlphaLen[ioff]; ++j) {
                        const size_t ja   = h.DoublesFromAlphaSM[ioff][j];
                        const int    kIdx = static_cast<int>(
                                (ja - h.ketAlphaStart) * ketBetaSize
                                + (ib - h.ketBetaStart));
                        const int* cr = h.DoublesAlphaCrAnSM[ioff];
                        exc[wp++] = { kIdx, cr[4*j+0], cr[4*j+1], cr[4*j+2], cr[4*j+3], 1 };
                    }

                } else if (h.taskType == 1) {
                    // ---- single beta ----
                    for (size_t j = 0; j < h.SinglesFromBetaLen[boff]; ++j) {
                        const size_t jb   = h.SinglesFromBetaSM[boff][j];
                        const int    kIdx = static_cast<int>(
                                (ia - h.ketAlphaStart) * ketBetaSize
                                + (jb - h.ketBetaStart));
                        const int* cr = h.SinglesBetaCrAnSM[boff];
                        exc[wp++] = { kIdx, cr[2*j+0], 0, cr[2*j+1], 0, 0 };
                    }
                    // ---- double beta ----
                    for (size_t j = 0; j < h.DoublesFromBetaLen[boff]; ++j) {
                        const size_t jb   = h.DoublesFromBetaSM[boff][j];
                        const int    kIdx = static_cast<int>(
                                (ia - h.ketAlphaStart) * ketBetaSize
                                + (jb - h.ketBetaStart));
                        const int* cr = h.DoublesBetaCrAnSM[boff];
                        exc[wp++] = { kIdx, cr[4*j+0], cr[4*j+1], cr[4*j+2], cr[4*j+3], 1 };
                    }

                } else { // taskType == 0: mixed alpha-beta double
                    for (size_t j = 0; j < h.SinglesFromAlphaLen[ioff]; ++j) {
                        const size_t ja    = h.SinglesFromAlphaSM[ioff][j];
                        const int*   alpha = h.SinglesAlphaCrAnSM[ioff];
                        for (size_t k = 0; k < h.SinglesFromBetaLen[boff]; ++k) {
                            const size_t jb   = h.SinglesFromBetaSM[boff][k];
                            const int    kIdx = static_cast<int>(
                                    (ja - h.ketAlphaStart) * ketBetaSize
                                    + (jb - h.ketBetaStart));
                            const int* beta = h.SinglesBetaCrAnSM[boff];
                            exc[wp++] = {
                                    kIdx,
                                    alpha[2*j+0], beta[2*k+0],
                                    alpha[2*j+1], beta[2*k+1],
                                    1
                            };
                        }
                    }
                }
            } // end for bIdx
        } // end for tasks
    }

} // namespace cuda_impl

// Wrappers for benchmarking with separable compilation
namespace cuda_impl {
    template<typename ElemT>
    inline void initGpuContext(
            gpuContext_t<ElemT>& ctx,
            int rank_for_device, int mpi_rank_h, int mpi_size_h,
            const std::vector<uint64_t>& dets_flat, int detWords,
            const std::vector<std::vector<int>>& csr_row_ptr_per_task,
            const std::vector<std::vector<excitation_t>>& csr_exc_per_task,
            const ElemT* h_one, const size_t& h_one_size, const ElemT* h_two,
            const size_t& h_two_size, int norbs, int bit_length, size_t vecSize) {
        InitGpuContext(ctx, rank_for_device, mpi_rank_h, mpi_size_h,
                       dets_flat, detWords, csr_row_ptr_per_task, csr_exc_per_task,
                       h_one, h_one_size, h_two, h_two_size, norbs, bit_length, vecSize);
    }

    template<typename ElemT>
    inline void gpuKetVecH2D(gpuContext_t<ElemT>& ctx, const std::vector<ElemT>& Tvec) { GPUKetVecH2D(ctx, Tvec); }

    template<typename ElemT>
    inline void gpuVecH2D(gpuContext_t<ElemT>& ctx, const std::vector<ElemT>& Wb) { GPUVecH2D(ctx, Wb); }

    template<typename ElemT>
    inline void gpuVecD2H(gpuContext_t<ElemT>& ctx, std::vector<ElemT>& Wb) { GPUVecD2H(ctx, Wb); }

    template<typename ElemT>
    inline void gpuSpMV(gpuContext_t<ElemT>& ctx, int taskIdx) { GPUSpMV(ctx, taskIdx); }
}

#endif // KERNEL_MULT_H