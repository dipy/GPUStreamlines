template<int BDIM_X,
         typename VAL_T>
__device__ VAL_T max_d(const int n, const VAL_T *__restrict__ src, const VAL_T minVal) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        VAL_T __m = minVal;

        for(int i = tidx; i < n; i += BDIM_X) {
		__m = MAX(__m, src[i]);
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const VAL_T __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MAX(__m, __tmp);
        }

        return __m;
}

template<int BDIM_X,
         typename LEN_T,
         typename VAL_T>
__device__ VAL_T max_mask_transl_d(const int n,
				   const LEN_T *__restrict__ srcMsk,
                                   const VAL_T *__restrict__ srcVal,
                                   const VAL_T offset,
                                   const VAL_T minVal) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        VAL_T __m = minVal;

        for(int i = tidx; i < n; i += BDIM_X) {
		const LEN_T sel = srcMsk[i];
		if (sel > 0) {
			__m = MAX(__m, srcVal[i]+offset);
		}
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const VAL_T __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MAX(__m, __tmp);
        }

        return __m;
}

template<int BDIM_X,
         typename VAL_T>
__device__ VAL_T min_d(const int n, const VAL_T *__restrict__ src, const VAL_T maxVal) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        VAL_T __m = maxVal;

        for(int i = tidx; i < n; i += BDIM_X) {
		__m = MIN(__m, src[i]);
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const VAL_T __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MIN(__m, __tmp);
        }

        return __m;
}

template<int BDIM_X, typename REAL_T>
__device__ void prefix_sum_sh_d(REAL_T *num_sh, int __len) {
    const int tidx = threadIdx.x;

    const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
    const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

#if 0
    // for debugging
    __syncwarp(WMASK);
    if (tidx == 0) {
        for (int j = 1; j < __len; j++) {
            num_sh[j] += num_sh[j-1];
        }
    }
    __syncwarp(WMASK);
#else
    for (int j = 0; j < __len; j += BDIM_X) {
        if ((tidx == 0) && (j != 0)) {
            num_sh[j] += num_sh[j-1];
        }
        __syncwarp(WMASK);

        REAL_T __t_pmf;
        if (j+tidx < __len) {
            __t_pmf = num_sh[j+tidx];
        }
        for (int i = 1; i < BDIM_X; i*=2) {
            REAL_T __tmp = __shfl_up_sync(WMASK, __t_pmf, i, BDIM_X);
            if ((tidx >= i) && (j+tidx < __len)) {
                    __t_pmf += __tmp;
            }
        }
        if (j+tidx < __len) {
            num_sh[j+tidx] = __t_pmf;
        }
        __syncwarp(WMASK);
    }
#endif
}

template<typename REAL_T>
__device__ void printArrayAlways(const char *name, int ncol, int n, REAL_T *arr) {
        printf("%s:\n", name);

        for(int j = 0; j < n; j++) {
                if ((j%ncol)==0) printf("\n");
                printf("%10.8f ", arr[j]);
        }
        printf("\n");
}

template<typename REAL_T>
__device__ void printArray(const char *name, int ncol, int n, REAL_T *arr) {
	if (!threadIdx.x && !threadIdx.y && !blockIdx.x) {
		printArrayAlways(name, ncol, n, arr);
	}
}
