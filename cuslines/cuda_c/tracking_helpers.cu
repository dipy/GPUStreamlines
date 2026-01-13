
using namespace cuwsort;

template<typename REAL_T>
__device__ REAL_T interpolation_helper_d(const REAL_T*__restrict__ dataf, const REAL_T wgh[3][2], const long long coo[3][2], int dimy, int dimz, int dimt, int t) {
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

        // if (threadIdx.x == 0) {
        //         printf("point: %f, %f, %f\n", point.x, point.y, point.z);
        //         printf("dimt_idx: %d\n", dimt_idx);
        //         // for(int i = 0; i < dimt; i++) {
        //         //         printf("__vox_data[%d]: %f\n", i, __vox_data[i]);
        //         // }
        // }
        return 0;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int check_point_d(const REAL_T tc_threshold,
			     const REAL3_T point,
                             const int dimx,
                             const int dimy,
                             const int dimz,
                             const REAL_T *__restrict__ metric_map) {

        const int tidy = threadIdx.y;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        __shared__ REAL_T __shInterpOut[BDIM_Y];

        const int rv = trilinear_interp_d<BDIM_X>(dimx, dimy, dimz, 1, 0, metric_map, point, __shInterpOut+tidy);
        __syncwarp(WMASK);
#if 0
        if (threadIdx.y == 1 && threadIdx.x == 0) {
                printf("__shInterpOut[tidy]: %f, TC_THRESHOLD_P: %f\n", __shInterpOut[tidy], TC_THRESHOLD_P);
        }
#endif
        if (rv != 0) {
                return OUTSIDEIMAGE;
        }
        //return (__shInterpOut[tidy] > TC_THRESHOLD_P) ? TRACKPOINT : ENDPOINT;
        return (__shInterpOut[tidy] > tc_threshold) ? TRACKPOINT : ENDPOINT;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int peak_directions_d(const REAL_T  *__restrict__ odf,
                                       REAL3_T *__restrict__ dirs,
                                 const REAL3_T *__restrict__ sphere_vertices,
                                 const int2 *__restrict__ sphere_edges,
                                 const int num_edges,
				 int samplm_nr,
				 int *__restrict__ __shInd,
				 const REAL_T relative_peak_thres,
				 const REAL_T min_separation_angle) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        const unsigned int lmask = (1 << lid)-1;

//        __shared__ int __shInd[BDIM_Y][SAMPLM_NR];

        #pragma unroll
        for(int j = tidx; j < samplm_nr; j += BDIM_X) {
		__shInd[j] = 0;
        }

        REAL_T odf_min = min_d<BDIM_X>(samplm_nr, odf, REAL_MAX);
        odf_min = MAX(0, odf_min);

        __syncwarp(WMASK);

        // local_maxima() + _compare_neighbors()
        // selecting only the indices corrisponding to maxima Ms
        // such that M-odf_min >= relative_peak_thres
        //#pragma unroll
        for(int j = 0; j < num_edges; j += BDIM_X) {
                if (j+tidx < num_edges) {
                        const int u_ind = sphere_edges[j+tidx].x;
                        const int v_ind = sphere_edges[j+tidx].y;

                        //if (u_ind >= NUM_EDGES || v_ind >= NUM_EDGES) { ERROR; }

                        const REAL_T u_val = odf[u_ind];
                        const REAL_T v_val = odf[v_ind];

                        //if (u_val != u_val || v_val != v_val) { ERROR_NANs; }

                        // only check that they are not equal
                        //if (u_val != v_val) {
                        //        __shInd[tidy][u_val < v_val ? u_ind : v_ind] = -1; // benign race conditions...
                        //}
                        if (u_val < v_val) {
                                atomicExch(__shInd+u_ind, -1);
                                atomicOr(  __shInd+v_ind,  1);
                        } else if (v_val < u_val) {
                                atomicExch(__shInd+v_ind, -1);
                                atomicOr(  __shInd+u_ind,  1);
                        }
                }
        }
        __syncwarp(WMASK);

        const REAL_T compThres = relative_peak_thres*max_mask_transl_d<BDIM_X>(samplm_nr, __shInd, odf, -odf_min, REAL_MIN);
#if 1
/*
        if (!tidy && !tidx) {
                for(int j = 0; j < SAMPLM_NR; j++) {
                        printf("local_max[%d]: %d (%f)\n", j, __shInd[tidy][j], odf[j]);
                }
                printf("maxMax with offset %f: %f\n", -odf_min, compThres);
        }
        __syncwarp(WMASK);
*/
        // compact indices of positive values to the right
        int n = 0;

        for(int j = 0; j < samplm_nr; j += BDIM_X) {

                const int __v = (j+tidx < samplm_nr) ? __shInd[j+tidx] : -1;
                const int __keep = (__v > 0) && ((odf[j+tidx]-odf_min) >= compThres);
                const int __msk = __ballot_sync(WMASK, __keep);

//__syncwarp(WMASK); // unnecessary
                if (__keep) {
                        const int myoff = __popc(__msk & lmask);
                        __shInd[n + myoff] = j+tidx;
                }
                n += __popc(__msk);
//__syncwarp(WMASK); // should be unnecessary
        }
        __syncwarp(WMASK);
/*
        if (!tidy && !tidx) {
                for(int j = 0; j < n; j++) {
                        printf("local_max_compact[%d]: %d\n", j, __shInd[tidy][j]);
                }
        }
        __syncwarp(WMASK);
*/

        // sort local maxima indices
        if (n < BDIM_X) {
                REAL_T k = REAL_MIN;
                int    v = 0;
                if (tidx < n) {
                        v = __shInd[tidx];
                        k = odf[v];
                }
                warp_sort<32, BDIM_X, WSORT_DIR_DEC>(&k, &v);
                __syncwarp(WMASK);

                if (tidx < n) {
                        __shInd[tidx] = v;
                }
        } else {
                // ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        }
        __syncwarp(WMASK);

        // __shInd[tidy][] contains the indices in odf correspoding to
        // normalized maxima NOT sorted!
        if (n != 0) {
                // remove_similar_vertices()
                // PRELIMINARY INEFFICIENT, SINGLE TH, IMPLEMENTATION
                if (tidx == 0) {
                        const REAL_T cos_similarity = COS(min_separation_angle);

                        dirs[0] = sphere_vertices[__shInd[0]];

                        int k = 1;
                        for(int i = 1; i < n; i++) {

                                const REAL3_T abc = sphere_vertices[__shInd[i]];

                                int j = 0;
                                for(; j < k; j++) {
                                        const REAL_T cos = FABS(abc.x*dirs[j].x+
                                                                abc.y*dirs[j].y+
                                                                abc.z*dirs[j].z);
                                        if (cos > cos_similarity) {
                                                break;
                                        }
                                }
                                if (j == k) {
                                        dirs[k++] = abc;
                                }
                        }
                        n = k;
                }
                n = __shfl_sync(WMASK, n, 0, BDIM_X);
                __syncwarp(WMASK);

        }
/*
        if (!tidy && !tidx) {
                for(int j = 0; j < n; j++) {
                        printf("local_max_compact_uniq[%d]: %d\n", j, __shInd[tidy][j]);
                }
        }
        __syncwarp(WMASK);
*/
#else
        const int indMax = max_d<BDIM_X, SAMPLM_NR>(__shInd[tidy], -1);
        if (indMax != -1) {
                __ret = MAKE_REAL3(sphere_vertices[indMax][0],
                                   sphere_vertices[indMax][1],
                                   sphere_vertices[indMax][2]);
        }
#endif
        return n;
}
