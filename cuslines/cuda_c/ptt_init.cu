template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__global__ void getNumStreamlinesPtt_k(
                                        const REAL_T max_angle,
				        const REAL_T relative_peak_thres,
				        const REAL_T min_separation_angle,
				        const long long rndSeed,
                                        const int nseed,
                                        const REAL3_T *__restrict__ seeds,
                                        const cudaTextureObject_t *__restrict__ pmf,
                                        const REAL3_T *__restrict__ sphere_vertices,
                                        const int2 *__restrict__ sphere_edges,
                                        const int num_edges,
                                        REAL3_T *__restrict__ shDir0,
                                        int *slineOutOff) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        
        const int slid = blockIdx.x*blockDim.y + threadIdx.y;
        const size_t gid = blockIdx.x * blockDim.y * blockDim.x + blockDim.x * threadIdx.y + threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        if (slid >= nseed) {
                return;
        }

        REAL3_T *__restrict__ __shDir = shDir0+slid*DIMT;
        curandStatePhilox4_32_10_t st;
        curand_init(rndSeed, gid, 0, &st);

        extern __shared__ REAL_T __sh[];
        REAL_T *__pmf_data_sh = __sh + tidy*N32DIMT;

        REAL3_T point = seeds[slid];

        #pragma unroll
        for (int i = tidx; i < DIMT; i += BDIM_X) {
                const int grid_col = i & WIDTH_MASK;
                const int grid_row = i >> LOG2_WIDTH;

                const REAL_T x_query = (REAL_T)(grid_col * DIMX) + point.x;
                const REAL_T y_query = (REAL_T)(grid_row * DIMY) + point.y;
                __pmf_data_sh[i] = tex3D<REAL_T>(*pmf, x_query, y_query, point.z);
                if (__pmf_data_sh[i] < PMF_THRESHOLD_P) {
                        __pmf_data_sh[i] = 0.0;
                }
        }
        __syncwarp(WMASK);

        int *__shInd = reinterpret_cast<int *>(__sh + BDIM_Y*N32DIMT) + tidy*N32DIMT;
        int ndir = peak_directions_d<
            BDIM_X,
            BDIM_Y>(__pmf_data_sh,
                    __shDir,
                    sphere_vertices,
                    sphere_edges,
                    num_edges,
                    DIMT,
                    __shInd,
                    relative_peak_thres,
                    min_separation_angle);

        if (tidx == 0) {
                slineOutOff[slid] = ndir;
        }

        return;
}
