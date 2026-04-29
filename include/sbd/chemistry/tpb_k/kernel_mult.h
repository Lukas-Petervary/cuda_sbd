#ifndef KERNEL_MULT_H
#define KERNEL_MULT_H

#include <vector>
#include <cstdint>
#include <cstdio>

#include "../basic/determinants.h"
#include "../tpb/helper.h"
#include "hamiltonian_kernel.h"

namespace cuda_impl {
    template<typename ElemT>
    inline void davidsonMatvecGPU(
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
    ) {
        // wrapper around CUDA entrypoint for profiling call tree
        LaunchDavidsonMatvecGPU(rank, dets, detWords, excitations, h_one, h_two, norbs, bit_length, Tvec, Wb);
    }

    template<typename ElemT>
    void FlattenExcitations(
            const std::vector<std::vector<size_t>> &adets,
            const std::vector<std::vector<size_t>> &bdets,
            const std::vector<sbd::TaskHelpers> &helper,

            size_t bit_length,
            size_t norbs,

            size_t braAlphaSize,
            size_t braBetaSize,

            std::vector<uint64_t> &dets_flat,
            std::vector<Excitation> &excitations,
            int &detWords
    ) {
        detWords = (2 * norbs + bit_length - 1) / bit_length;

        size_t nTasks = helper.size();
        std::vector<size_t> taskBraOffset(nTasks);

        size_t totalBraSize = 0;
        for (size_t t = 0; t < nTasks; ++t) {
            auto &h = helper[t];
            size_t localBraSize = (h.braAlphaEnd - h.braAlphaStart) * (h.braBetaEnd - h.braBetaStart);
            taskBraOffset[t] = totalBraSize;
            totalBraSize += localBraSize;
        }

        // PASS 1: thread-local counts
        int nthreads = omp_get_max_threads();
        std::vector<size_t> threadCounts(nthreads, 0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t localCount = 0;

            #pragma omp for schedule(dynamic)
            for (size_t t = 0; t < nTasks; t++) {
                const auto &h = helper[t];

                for (size_t ia = h.braAlphaStart; ia < h.braAlphaEnd; ia++) {
                    size_t ioff = ia - h.braAlphaStart;

                    for (size_t ib = h.braBetaStart; ib < h.braBetaEnd; ib++) {

                        if (h.taskType == 2) {
                            localCount += h.SinglesFromAlphaLen[ioff];
                            localCount += h.DoublesFromAlphaLen[ioff];
                        }
                        else if (h.taskType == 1) {
                            size_t boff = ib - h.braBetaStart;
                            localCount += h.SinglesFromBetaLen[boff];
                            localCount += h.DoublesFromBetaLen[boff];
                        }
                        else {
                            size_t boff = ib - h.braBetaStart;
                            localCount +=
                                    h.SinglesFromAlphaLen[ioff] *
                                    h.SinglesFromBetaLen[boff];
                        }
                    }
                }
            }

            threadCounts[tid] = localCount;
        }

        // Prefix sum over threads
        std::vector<size_t> threadOffset(nthreads, 0);
        size_t totalExcitations = 0;

        for (int t = 0; t < nthreads; t++) {
            threadOffset[t] = totalExcitations;
            totalExcitations += threadCounts[t];
        }

        // Allocate outputs
        excitations.resize(totalExcitations);
        dets_flat.resize(totalBraSize * detWords);

        // PASS 2: thread-local scan + write
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t write_ptr = threadOffset[tid];

            std::vector<uint64_t> DetI(detWords);

            #pragma omp for schedule(dynamic)
            for (size_t t = 0; t < nTasks; ++t) {
                const auto &h = helper[t];
                size_t ketBetaSize = h.ketBetaEnd - h.ketBetaStart;
                size_t base = taskBraOffset[t];

                for (size_t ia = h.braAlphaStart; ia < h.braAlphaEnd; ia++) {
                    size_t ioff = ia - h.braAlphaStart;

                    for (size_t ib = h.braBetaStart; ib < h.braBetaEnd; ib++) {

                        size_t local_bIdx =
                                (ia - h.braAlphaStart) *
                                (h.braBetaEnd - h.braBetaStart) +
                                (ib - h.braBetaStart);

                        size_t bIdx = base + local_bIdx;

                        // ---- determinant write ----
                        sbd::DetFromAlphaBeta(
                                adets[ia], bdets[ib],
                                bit_length, norbs,
                                DetI
                        );

                        size_t detOffset = bIdx * detWords;
                        for (int w = 0; w < detWords; w++)
                            dets_flat[detOffset + w] = DetI[w];

                        // ---- generate excitations ----
                        if (h.taskType == 2) {

                            for (size_t j = 0; j < h.SinglesFromAlphaLen[ioff]; j++) {
                                size_t ja = h.SinglesFromAlphaSM[ioff][j];
                                size_t kIdx = (ja - h.ketAlphaStart) * ketBetaSize + (ib - h.ketBetaStart);

                                auto &cr = h.SinglesAlphaCrAnSM[ioff];
                                excitations[write_ptr++] = {
                                        (int)bIdx, (int)kIdx,
                                        cr[2*j+0], 0,
                                        cr[2*j+1], 0,
                                        0
                                };
                            }

                            for (size_t j = 0; j < h.DoublesFromAlphaLen[ioff]; j++) {
                                size_t ja = h.DoublesFromAlphaSM[ioff][j];
                                size_t kIdx = (ja - h.ketAlphaStart) * ketBetaSize + (ib - h.ketBetaStart);

                                auto &cr = h.DoublesAlphaCrAnSM[ioff];
                                excitations[write_ptr++] = {
                                        (int)bIdx, (int)kIdx,
                                        cr[4*j+0], cr[4*j+1],
                                        cr[4*j+2], cr[4*j+3],
                                        1
                                };
                            }
                        }
                        else if (h.taskType == 1) {
                            size_t boff = ib - h.braBetaStart;

                            for (size_t j = 0; j < h.SinglesFromBetaLen[boff]; j++) {
                                size_t jb = h.SinglesFromBetaSM[boff][j];
                                size_t kIdx = (ia - h.ketAlphaStart) * ketBetaSize + (jb - h.ketBetaStart);

                                auto &cr = h.SinglesBetaCrAnSM[boff];
                                excitations[write_ptr++] = {
                                        (int)bIdx, (int)kIdx,
                                        cr[2*j+0], 0,
                                        cr[2*j+1], 0,
                                        0
                                };
                            }

                            for (size_t j = 0; j < h.DoublesFromBetaLen[boff]; j++) {
                                size_t jb = h.DoublesFromBetaSM[boff][j];
                                size_t kIdx = (ia - h.ketAlphaStart) * ketBetaSize + (jb - h.ketBetaStart);

                                auto &cr = h.DoublesBetaCrAnSM[boff];
                                excitations[write_ptr++] = {
                                        (int)bIdx, (int)kIdx,
                                        cr[4*j+0], cr[4*j+1],
                                        cr[4*j+2], cr[4*j+3],
                                        1
                                };
                            }
                        }
                        else {
                            size_t boff = ib - h.braBetaStart;

                            for (size_t j = 0; j < h.SinglesFromAlphaLen[ioff]; j++) {
                                size_t ja = h.SinglesFromAlphaSM[ioff][j];

                                for (size_t k = 0; k < h.SinglesFromBetaLen[boff]; k++) {
                                    size_t jb = h.SinglesFromBetaSM[boff][k];
                                    size_t kIdx = (ja - h.ketAlphaStart) * ketBetaSize + (jb - h.ketBetaStart);

                                    auto &alpha = h.SinglesAlphaCrAnSM[ioff];
                                    auto &beta  = h.SinglesBetaCrAnSM[boff];
                                    // pack mixed excitations with {alpha, beta} scheme marked as double
                                    excitations[write_ptr++] = {
                                            (int)bIdx, (int)kIdx,
                                            alpha[2*j+0], beta[2*k+0],
                                            alpha[2*j+1], beta[2*k+1],
                                            1
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

#endif
