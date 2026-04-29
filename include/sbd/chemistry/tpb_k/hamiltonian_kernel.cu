#include "hamiltonian_kernel.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(err);} \
    } while(0)

namespace cuda_impl {
    template<typename ElemT>
    struct DeviceOneInt {
        const ElemT* data;
        int norbs;
        __device__ inline ElemT operator()(int i, int j) const {
            return data[i * norbs + j];
        }
    };

    template<typename ElemT>
    struct DeviceTwoInt {
        const ElemT* data;
        int norbs;
        __device__ inline ElemT operator()(int i, int j, int k, int l) const {
            if (!((i % 2 == j % 2) && (k % 2 == l % 2))) return 0.0;

            int I = i / 2, J = j / 2;
            int K = k / 2, L = l / 2;

            int ij = max(I, J) * (max(I, J) + 1) / 2 + min(I, J);
            int kl = max(K, L) * (max(K, L) + 1) / 2 + min(K, L);

            int a = max(ij, kl);
            int b = min(ij, kl);

            return data[a * (a + 1) / 2 + b];
        }
    };

    __device__ inline int parity_single(const uint64_t* det, int bit_length, int i, int a) {
        int start = min(i,a);
        int end   = max(i,a);

        int count = 0;
        for (int p = start+1; p < end; p++) {
            int w = p / bit_length;
            int b = p % bit_length;
            count += (det[w] >> b) & 1ULL;
        }
        return (count % 2 == 0) ? 1 : -1;
    }

    template<typename ElemT>
    __global__ void fused_kernel(
            const uint64_t* dets,
            int detWords,

            const Excitation* exc,

            const ElemT* Tvec,
            ElemT* Wb,

            DeviceOneInt<ElemT> h1,
            DeviceTwoInt<ElemT> h2,

            int bit_length,
            int N
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= N) return;

        Excitation e = exc[tid];

        const uint64_t* det = dets + e.braIdx * detWords;

        double val = 0.0;

        if (e.type == 0) { // single
            int i = e.i;
            int a = e.a;

            val = h1(a, i);

            for (int w=0; w<detWords; w++) {
                uint64_t bits = det[w];
                while (bits) {
                    int pos = __ffsll(bits) - 1;
                    int j = w * bit_length + pos;

                    val += h2(a,j,i,j) - h2(a,j,j,i);
                    bits &= bits - 1;
                }
            }

            val *= parity_single(det, bit_length, i, a);
        } else {
            int I = min(e.i, e.j);
            int J = max(e.i, e.j);
            int A = min(e.a, e.b);
            int B = max(e.a, e.b);

            val = h2(A,I,B,J) - h2(A,J,B,I);
        }

        atomicAdd(&Wb[e.braIdx], val * Tvec[e.ketIdx]);
    }

#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 256
#endif
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
    ) {
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        int device = rank % numDevices;
        CUDA_CHECK(cudaSetDevice(device));

        int N = excitations.size();

        uint64_t *d_dets = nullptr;
        Excitation *d_exc = nullptr;
        ElemT *d_T = nullptr, *d_Wb = nullptr;
        ElemT *d_one = nullptr, *d_two = nullptr;

        // this entire section is memory bottle-necking every GPU
        //   that it touches, and half the vars can be persistent...

        // determinants
        size_t detSize = dets.size() * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&d_dets, detSize));
        CUDA_CHECK(cudaMemcpy(d_dets, dets.data(), detSize, cudaMemcpyHostToDevice));

        // excitations
        size_t excSize = N * sizeof(Excitation);
        CUDA_CHECK(cudaMalloc(&d_exc, excSize));
        CUDA_CHECK(cudaMemcpy(d_exc, excitations.data(), excSize, cudaMemcpyHostToDevice));

        // vectors
        size_t vecSize = Tvec.size() * sizeof(ElemT);
        CUDA_CHECK(cudaMalloc(&d_T, vecSize));
        CUDA_CHECK(cudaMemcpy(d_T, Tvec.data(), vecSize, cudaMemcpyHostToDevice));

        size_t hamSize = Wb.size() * sizeof(ElemT);
        CUDA_CHECK(cudaMalloc(&d_Wb, hamSize));
        CUDA_CHECK(cudaMemcpy(d_Wb, Wb.data(), hamSize, cudaMemcpyHostToDevice));

        // integrals
        size_t intSize = norbs * norbs * sizeof(ElemT);
        CUDA_CHECK(cudaMalloc(&d_one, intSize));
        CUDA_CHECK(cudaMemcpy(d_one, h_one, intSize, cudaMemcpyHostToDevice));

        size_t nPairs = norbs * (norbs + 1) / 2;
        size_t twoSize = nPairs * (nPairs + 1) / 2;

        size_t orbSize = twoSize * sizeof(ElemT);
        CUDA_CHECK(cudaMalloc(&d_two, orbSize));
        CUDA_CHECK(cudaMemcpy(d_two, h_two, orbSize, cudaMemcpyHostToDevice));

#if 0
        printf(
                "detSize = %zu, excSize = %zu, vecSize = %zu, hamSize = %zu, intSize = %zu, orbSize = %zu\n",
                detSize,        excSize,       vecSize,       hamSize,       intSize,       orbSize
        );
#endif

        DeviceOneInt<ElemT> dH1{d_one, norbs};
        DeviceTwoInt<ElemT> dH2{d_two, norbs};

        // kernel launch
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fused_kernel<<<grid, BLOCK_SIZE>>>(
                d_dets, detWords,
                d_exc,
                d_T, d_Wb,
                dH1, dH2,
                bit_length,
                N
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(Wb.data(), d_Wb, Wb.size() * sizeof(ElemT), cudaMemcpyDeviceToHost));

        // cleanup
        cudaFree(d_dets);
        cudaFree(d_exc);
        cudaFree(d_T);
        cudaFree(d_Wb);
        cudaFree(d_one);
        cudaFree(d_two);
    }

} // end namespace cuda_impl