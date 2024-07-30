
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

template<typename REAL_T>
__device__ REAL_T interpolation_helper_d(const REAL_T* dataf, const REAL_T wgh[3][2], const long long coo[3][2], int dimy, int dimz, int dimt, int t) {
    REAL_T __tmp = 0;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                __tmp += wgh[0][i] * wgh[1][j] * wgh[2][k] *
                         dataf[coo[0][i] * dimy * dimz * dimt +
                               coo[1][j] * dimz * dimt +
                               coo[2][k] * dimt +
                               t];
            }
        }
    }
    return __tmp;
}

template<int BDIM_X,
         typename REAL_T,
         typename REAL3_T>
__device__ int trilinear_interp_d(const int dimx,
                                  const int dimy,
                                  const int dimz,
                                  const int dimt,
                                  int dimt_idx, // If -1, get all
                                  const REAL_T *__restrict__ dataf,
                                  const REAL3_T point,
                                  REAL_T *__restrict__ __vox_data) {
        const REAL_T HALF = static_cast<REAL_T>(0.5);

        // all thr compute the same here
        if (point.x < -HALF || point.x+HALF >= dimx ||
            point.y < -HALF || point.y+HALF >= dimy ||
               point.z < -HALF || point.z+HALF >= dimz) {
                return -1;
        }

        long long  coo[3][2];
        REAL_T wgh[3][2]; // could use just one...

        const REAL_T ONE  = static_cast<REAL_T>(1.0);

        const REAL3_T fl = MAKE_REAL3(FLOOR(point.x),
                                      FLOOR(point.y),
                                      FLOOR(point.z));

        wgh[0][1] = point.x - fl.x; 
        wgh[0][0] = ONE-wgh[0][1]; 
        coo[0][0] = MAX(0, fl.x);
        coo[0][1] = MIN(dimx-1, coo[0][0]+1);

        wgh[1][1] = point.y - fl.y; 
        wgh[1][0] = ONE-wgh[1][1]; 
        coo[1][0] = MAX(0, fl.y);
        coo[1][1] = MIN(dimy-1, coo[1][0]+1);

        wgh[2][1] = point.z - fl.z; 
        wgh[2][0] = ONE-wgh[2][1]; 
        coo[2][0] = MAX(0, fl.z);
        coo[2][1] = MIN(dimz-1, coo[2][0]+1);

        if (dimt_idx == -1) {
                for (int t = threadIdx.x; t < dimt; t += BDIM_X) {
                        __vox_data[t] = interpolation_helper_d(dataf, wgh, coo, dimy, dimz, dimt, t);
                }
        } else {
                *__vox_data = interpolation_helper_d(dataf, wgh, coo, dimy, dimz, dimt, dimt_idx);
        }

        /*
        __syncwarp(WMASK);
        if (tidx == 0 && threadIdx.y == 0) {
                printf("point: %f, %f, %f\n", point.x, point.y, point.z);
                for(int i = 0; i < dimt; i++) {
                        printf("__vox_data[%d]: %f\n", i, __vox_data[i]);
                }
        }
        */
        return 0;
}
