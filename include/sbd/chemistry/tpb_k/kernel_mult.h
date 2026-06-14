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

    // ----------------------------------------------------------------
    // BuildCsrExcitations
    //
    // ROOT CAUSE OF CRASH (fixed here):
    //   The previous version assigned each task its own slice of bra
    //   index space via taskBraOffset[], making nBras = nTasks * W.size().
    //   With even 2 tasks this caused atomicAdd(&d_Wb[braIdx], ...) and
    //   the det pointer (dets + braIdx * detWords) to go out of bounds.
    //
    //   The CPU mult() never does this. It treats braIdx as LOCAL to the
    //   shared bra window [0, braAlphaSize * braBetaSize), and all tasks
    //   ACCUMULATE into the same output range.  The CSR builder must match.
    //
    // Correct model:
    //   - nBras = braAlphaSize * braBetaSize  (same as W.size(), single node)
    //   - bIdx  = (ia - braAlphaStart) * braBetaSize + (ib - braBetaStart)
    //             using helper[0] as the global reference (matching mult())
    //   - csr_row_ptr[bIdx+1] += count   (ACCUMULATE across tasks, not =)
    //   - dets_flat stores each bra determinant exactly once
    //
    // Race-free parallel fill:
    //   Outer loop over bras (OMP). Inner serial loop over tasks.
    //   Each bIdx owns [row_ptr[bIdx], row_ptr[bIdx+1]) exclusively.
    // ----------------------------------------------------------------
    template<typename ElemT>
    void BuildCsrExcitations(
            const std::vector<std::vector<size_t>>& adets,
            const std::vector<std::vector<size_t>>& bdets,
            const std::vector<sbd::TaskHelpers>&    helper,
            size_t                                  bit_length,
            size_t                                  norbs,
            std::vector<uint64_t>&                  dets_flat,
            std::vector<int>&                       csr_row_ptr,
            std::vector<excitation_t>&              csr_exc,
            int&                                    detWords
    ) {
        detWords = static_cast<int>((2 * norbs + bit_length - 1) / bit_length);

        if (helper.empty()) {
            csr_row_ptr.assign(1, 0);
            csr_exc.clear();
            dets_flat.clear();
            return;
        }

        const size_t nTasks = helper.size();

        // ---- Global bra reference (shared by all tasks, mirrors mult()) ----
        // helper[0] defines the bra window for this MPI rank.
        // All tasks process excitations from this same bra set into different
        // parts of the ket space; they don't own separate bra index ranges.
        const size_t braAlphaStart = helper[0].braAlphaStart;
        const size_t braAlphaEnd   = helper[0].braAlphaEnd;
        const size_t braBetaStart  = helper[0].braBetaStart;
        const size_t braBetaEnd    = helper[0].braBetaEnd;
        const size_t braAlphaSize  = braAlphaEnd - braAlphaStart;
        const size_t braBetaSize   = braBetaEnd  - braBetaStart;
        const size_t totalBras     = braAlphaSize * braBetaSize;

        // ---- Step 1: per-bra excitation count across ALL tasks ----
        // csr_row_ptr[bIdx+1] accumulates counts from every task whose bra
        // range includes (ia, ib).  += is critical — tasks write to the same
        // bIdx and their contributions must be summed, not overwritten.
        csr_row_ptr.assign(totalBras + 1, 0);

        for (size_t t = 0; t < nTasks; ++t) {
            const auto& h = helper[t];
            for (size_t ia = h.braAlphaStart; ia < h.braAlphaEnd; ++ia) {
                const size_t ioff = ia - h.braAlphaStart;
                for (size_t ib = h.braBetaStart; ib < h.braBetaEnd; ++ib) {
                    const size_t boff = ib - h.braBetaStart;
                    // Global bra index — consistent with braIdx in mult()
                    const size_t bIdx = (ia - braAlphaStart) * braBetaSize + (ib - braBetaStart);

                    int cnt = static_cast<int>(
                            h.taskType == 2 ? h.SinglesFromAlphaLen[ioff] + h.DoublesFromAlphaLen[ioff] :
                            h.taskType == 1 ? h.SinglesFromBetaLen[boff] + h.DoublesFromBetaLen[boff] :
                            h.SinglesFromAlphaLen[ioff] * h.SinglesFromBetaLen[boff]
                    );
                    csr_row_ptr[bIdx + 1] += cnt;   // ACCUMULATE, not assign
                }
            }
        }

        // ---- Step 2: inclusive prefix sum → final row_ptr ----
        for (size_t i = 0; i < totalBras; ++i)
            csr_row_ptr[i + 1] += csr_row_ptr[i];
        const size_t totalExc = static_cast<size_t>(csr_row_ptr[totalBras]);

        // ---- Step 3: allocate ----
        dets_flat.resize(totalBras * static_cast<size_t>(detWords));
        csr_exc.resize(totalExc);

        // ---- Step 4: fill (parallel over bras, serial over tasks per bra) ----
        //
        // Outer loop is over bra indices (OMP parallel, schedule dynamic).
        // Inner loop iterates all tasks for that bra.
        //
        // Race-free because every bIdx has exclusive ownership of:
        //   csr_exc   [ row_ptr[bIdx] .. row_ptr[bIdx+1] )
        //   dets_flat [ bIdx*detWords .. (bIdx+1)*detWords )
        //
        // The write pointer 'wp' starts at row_ptr[bIdx] and advances to
        // row_ptr[bIdx+1]; it is private to the thread handling bIdx.
        #pragma omp parallel
        {
            std::vector<uint64_t> DetI(static_cast<size_t>(detWords));

            #pragma omp for schedule(dynamic)
            for (size_t bIdx = 0; bIdx < totalBras; ++bIdx) {
                // Recover (ia, ib) from the flat bra index
                const size_t ia = braAlphaStart + bIdx / braBetaSize;
                const size_t ib = braBetaStart  + bIdx % braBetaSize;

                // Write bra determinant once (same bitstring regardless of task)
                sbd::DetFromAlphaBeta(
                        adets[ia], bdets[ib], bit_length, norbs, DetI);
                const size_t detOff = bIdx * static_cast<size_t>(detWords);
                for (int w = 0; w < detWords; ++w)
                    dets_flat[detOff + w] = DetI[w];

                // Walk tasks; write each task's excitations for (ia, ib).
                // wp is private to this bIdx — no synchronization needed.
                int wp = csr_row_ptr[bIdx];

                for (size_t t = 0; t < nTasks; ++t) {
                    const auto& h = helper[t];

                    // Skip tasks whose bra range does not contain (ia, ib).
                    // For the typical single-node case all tasks span the full
                    // local bra window, so these checks are always false.
                    if (ia < h.braAlphaStart || ia >= h.braAlphaEnd) continue;
                    if (ib < h.braBetaStart  || ib >= h.braBetaEnd)  continue;

                    const size_t ioff        = ia - h.braAlphaStart;
                    const size_t boff        = ib - h.braBetaStart;
                    const size_t ketBetaSize = h.ketBetaEnd - h.ketBetaStart;

                    if (h.taskType == 2) {
                        // ---- single alpha ----
                        for (size_t j = 0; j < h.SinglesFromAlphaLen[ioff]; ++j) {
                            const size_t ja   = h.SinglesFromAlphaSM[ioff][j];
                            const int    kIdx = static_cast<int>(
                                    (ja - h.ketAlphaStart) * ketBetaSize
                                    + (ib - h.ketBetaStart));
                            const int* cr = h.SinglesAlphaCrAnSM[ioff];
                            csr_exc[wp++] = {
                                    kIdx, cr[2*j+0], 0, cr[2*j+1], 0, 0
                            };
                        }
                        // ---- double alpha ----
                        for (size_t j = 0; j < h.DoublesFromAlphaLen[ioff]; ++j) {
                            const size_t ja   = h.DoublesFromAlphaSM[ioff][j];
                            const int    kIdx = static_cast<int>(
                                    (ja - h.ketAlphaStart) * ketBetaSize
                                    + (ib - h.ketBetaStart));
                            const int* cr = h.DoublesAlphaCrAnSM[ioff];
                            csr_exc[wp++] = {
                                    kIdx,
                                    cr[4*j+0], cr[4*j+1],
                                    cr[4*j+2], cr[4*j+3],
                                    1
                            };
                        }

                    } else if (h.taskType == 1) {
                        // ---- single beta ----
                        for (size_t j = 0; j < h.SinglesFromBetaLen[boff]; ++j) {
                            const size_t jb   = h.SinglesFromBetaSM[boff][j];
                            const int    kIdx = static_cast<int>(
                                    (ia - h.ketAlphaStart) * ketBetaSize
                                    + (jb - h.ketBetaStart));
                            const int* cr = h.SinglesBetaCrAnSM[boff];
                            csr_exc[wp++] = {
                                    kIdx, cr[2*j+0], 0, cr[2*j+1], 0, 0
                            };
                        }
                        // ---- double beta ----
                        for (size_t j = 0; j < h.DoublesFromBetaLen[boff]; ++j) {
                            const size_t jb   = h.DoublesFromBetaSM[boff][j];
                            const int    kIdx = static_cast<int>(
                                    (ia - h.ketAlphaStart) * ketBetaSize
                                    + (jb - h.ketBetaStart));
                            const int* cr = h.DoublesBetaCrAnSM[boff];
                            csr_exc[wp++] = {
                                    kIdx,
                                    cr[4*j+0], cr[4*j+1],
                                    cr[4*j+2], cr[4*j+3],
                                    1
                            };
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
                                csr_exc[wp++] = {
                                        kIdx,
                                        alpha[2*j+0], beta[2*k+0],
                                        alpha[2*j+1], beta[2*k+1],
                                        1
                                };
                            }
                        }
                    }
                } // end for tasks
            } // end for bIdx
        } // end omp parallel
    }

    // ----------------------------------------------------------------
    // Thin wrappers for profiling call-tree attribution.
    // Signatures are unchanged from the first version.
    // ----------------------------------------------------------------

    template<typename ElemT>
    inline void initializeDavidsonGPU(
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
    ) {
        InitializeDavidsonGPU(ctx, rank,
                              dets_flat, detWords,
                              csr_row_ptr, csr_exc,
                              h_one, h_two,
                              norbs, bit_length, vecSize);
    }

    template<typename ElemT>
    inline void davidsonMatvecGPU(
            DavidsonGpuContext<ElemT>&  gpu_ctx,
            const std::vector<ElemT>&   Tvec,
            std::vector<ElemT>&         Wb
    ) {
        DavidsonMatvecGPU(gpu_ctx, Tvec, Wb);
    }

} // namespace cuda_impl

#endif // KERNEL_MULT_H