template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__global__ void getNumStreamlinesPtt_k( const int nseed,
                                        const REAL3_T *__restrict__ seeds,
                                        const cudaTextureObject_t *__restrict__ pmf,
                                        const REAL3_T *__restrict__ sphere_vertices,
                                        const int2 *__restrict__ sphere_edges,
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
        curand_init(RNG_SEED, gid, 0, &st);

        __shared__ REAL_T pmf_data_sh[BDIM_Y][DIMT];
        REAL_T* __pmf_data_sh = pmf_data_sh[tidy];

        REAL3_T point = seeds[slid];

        #pragma unroll
        for (int i = tidx; i < DIMT; i += BDIM_X) {
                const int tx = i & WIDTH_MASK;
                const int ty = (i >> LOG2_X) & HEIGHT_MASK;
                const int tz = (i >> (LOG2_X + LOG2_Y));

                const REAL_T x_query = (REAL_T)(tx * DIMX) + point.x;
                const REAL_T y_query = (REAL_T)(ty * DIMY) + point.y;
                const REAL_T z_query = (REAL_T)(tz * DIMZ) + point.z;
                __pmf_data_sh[i] = tex3D<REAL_T>(*pmf, x_query, y_query, z_query);
                if (__pmf_data_sh[i] < PMF_THRESHOLD_P) {
                        __pmf_data_sh[i] = 0.0;
                }
        }
        __syncwarp(WMASK);

        int ndir = peak_directions_d<
            BDIM_X,
            BDIM_Y>(__pmf_data_sh,
                    __shDir,
                    sphere_vertices,
                    sphere_edges);

        if (tidx == 0) {
                slineOutOff[slid] = ndir;
        }

        return;
}
