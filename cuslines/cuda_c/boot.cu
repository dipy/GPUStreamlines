//#define USE_FIXED_PERMUTATION
#ifdef USE_FIXED_PERMUTATION
//__device__ const int fixedPerm[] = {44, 47, 53,  0,  3,  3, 39,  9, 19, 21, 50, 36, 23,
//                                     6, 24, 24, 12,  1, 38, 39, 23, 46, 24, 17, 37, 25, 
//                                    13,  8,  9, 20, 51, 16, 51,  5, 15, 47,  0, 18, 35, 
//                                    24, 49, 51, 29, 19, 19, 14, 39, 32,  1,  9, 32, 31,
//                                    10, 52, 23};
__device__ const int fixedPerm[] = {
  47, 117,  67, 103,   9,  21,  36,  87,  70,  88, 140,  58,  39,  87,  88,  81,  25,  77,
  72,   9, 148, 115,  79,  82,  99,  29, 147, 147, 142,  32,   9, 127,  32,  31, 114,  28,
  34, 128, 128,  53, 133,  38,  17,  79, 132, 105,  42,  31, 120,   1,  65,  57,  35, 102,
 119,  11,  82,  91, 128, 142,  99,  53, 140, 121,  84,  68,   6,  47, 127, 131, 100,  78,
 143, 148,  23, 141, 117,  85,  48,  49,  69,  95,  94,   0, 113,  36,  48,  93, 131,  98,
  42, 112, 149, 127,   0, 138, 114,  43, 127,  23, 130, 121,  98,  62, 123,  82, 148,  50,
  14,  41,  58,  36,  10,  86,  43, 104,  11,   2,  51,  80,  32, 128,  38,  19,  42, 115,
  77,  30,  24, 125,   2,   3,  94, 107,  13, 112,  40,  72,  19,  95,  72,  67,  61,  14,
  96,   4, 139,  86, 121, 109};
#endif

template<int BDIM_X,
         typename VAL_T>
__device__ VAL_T avgMask(const int mskLen,
			 const int *__restrict__ mask,
			 const VAL_T *__restrict__ data) {
        
	const int tidx = threadIdx.x;
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;

        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        int   __myCnt = 0;
        VAL_T __mySum = 0;

        for(int i = tidx; i < mskLen; i += BDIM_X) {
		if(mask[i]) {
			__myCnt++;
			__mySum += data[i];
		}
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                __mySum += __shfl_xor_sync(WMASK, __mySum, i, BDIM_X);
                __myCnt += __shfl_xor_sync(WMASK, __myCnt, i, BDIM_X);
        }

        return __mySum/__myCnt;

}

template<
    int BDIM_X,
    typename LEN_T,
    typename MSK_T,
    typename VAL_T>
__device__ LEN_T maskGet(const LEN_T n, 
			 const MSK_T *__restrict__ mask,
			 const VAL_T *__restrict__ plain,
			       VAL_T *__restrict__ masked) {

	const int tidx = threadIdx.x;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int __laneMask = (1 << tidx)-1;

	int woff = 0;
	for(int j = 0; j < n; j += BDIM_X) {

		const int __act = (j+tidx < n) ? !mask[j+tidx] : 0;
		const int __msk = __ballot_sync(WMASK, __act);

		const int toff = __popc(__msk & __laneMask);
		if (__act) {
			masked[woff+toff] = plain[j+tidx];
		}
		woff += __popc(__msk);
	}
	return woff;
}

template<
    int BDIM_X,
    typename LEN_T,
    typename MSK_T,
    typename VAL_T>
__device__ void maskPut(const LEN_T n, 
			const MSK_T *__restrict__ mask,
			const VAL_T *__restrict__ masked,
			      VAL_T *__restrict__ plain) {

	const int tidx = threadIdx.x;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int __laneMask = (1 << tidx)-1;

	int woff = 0;
	for(int j = 0; j < n; j += BDIM_X) {

		const int __act = (j+tidx < n) ? !mask[j+tidx] : 0;
		const int __msk = __ballot_sync(WMASK, __act);

		const int toff = __popc(__msk & __laneMask);
		if (__act) {
			plain[j+tidx] = masked[woff+toff];
		}
		woff += __popc(__msk);
	}
	return;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int closest_peak_d(const REAL_T max_angle,
			      const REAL3_T  direction, //dir
                              const int npeaks,
                              const REAL3_T *__restrict__ peaks,
                                    REAL3_T *__restrict__ peak) {// dirs,

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        //const REAL_T cos_similarity = COS(MAX_ANGLE_P);
        const REAL_T cos_similarity = COS(max_angle);
#if 0
        if (!threadIdx.y && !tidx) {
                printf("direction: (%f, %f, %f)\n",
                        direction.x, direction.y, direction.z);
        }
        __syncwarp(WMASK);
#endif
        REAL_T cpeak_dot = 0;
        int    cpeak_idx = -1;
        for(int j = 0; j < npeaks; j += BDIM_X) {
                if (j+tidx < npeaks) {
#if 0
                        if (!threadIdx.y && !tidx) {
                                printf("j+tidx: %d, peaks[j+tidx]: (%f, %f, %f)\n",
                                        j+tidx, peaks[j+tidx].x, peaks[j+tidx].y, peaks[j+tidx].z);
                        }
#endif
                        const REAL_T dot = direction.x*peaks[j+tidx].x+
                                           direction.y*peaks[j+tidx].y+
                                           direction.z*peaks[j+tidx].z;

                        if (FABS(dot) > FABS(cpeak_dot)) {
                                cpeak_dot = dot;
                                cpeak_idx = j+tidx;
                        }
                }
        }
#if 0
        if (!threadIdx.y && !tidx) {
                printf("cpeak_idx: %d, cpeak_dot: %f\n", cpeak_idx, cpeak_dot);
        }
        __syncwarp(WMASK);
#endif

        #pragma unroll
        for(int j = BDIM_X/2; j; j /= 2) {

                const REAL_T dot = __shfl_xor_sync(WMASK, cpeak_dot, j, BDIM_X);
                const int    idx = __shfl_xor_sync(WMASK, cpeak_idx, j, BDIM_X);
                if (FABS(dot) > FABS(cpeak_dot)) {
                        cpeak_dot = dot;
                        cpeak_idx = idx;
                }
        }
#if 0
        if (!threadIdx.y && !tidx) {
                printf("cpeak_idx: %d, cpeak_dot: %f, cos_similarity: %f\n", cpeak_idx, cpeak_dot, cos_similarity);
        }
        __syncwarp(WMASK);
#endif
        if (cpeak_idx >= 0) {
                if (cpeak_dot >= cos_similarity) {
                        peak[0] = peaks[cpeak_idx];
                        return 1;
                }
                if (cpeak_dot <= -cos_similarity) {
                        peak[0] = MAKE_REAL3(-peaks[cpeak_idx].x,
                                             -peaks[cpeak_idx].y,
                                             -peaks[cpeak_idx].z);
                        return 1;
                }
        }
        return 0;
}

template<int BDIM_X,
         typename VAL_T>
__device__ void ndotp_d(const int N,
			const int M,
			const VAL_T *__restrict__ srcV,
                        const VAL_T *__restrict__ srcM,
                              VAL_T *__restrict__ dstV) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        //#pragma unroll
        for(int i = 0; i < N; i++) {

                VAL_T __tmp = 0;

                //#pragma unroll
                for(int j = 0; j < M; j += BDIM_X) {
                        if (j+tidx < M) {
                                __tmp += srcV[j+tidx]*srcM[i*M + j+tidx];
                        }
                }
                #pragma unroll
                for(int j = BDIM_X/2; j; j /= 2) {
#if 0
                        __tmp += __shfl_xor_sync(WMASK, __tmp, j, BDIM_X);
#else
                        __tmp += __shfl_down_sync(WMASK, __tmp, j, BDIM_X);
#endif
                }
                // values could be held by BDIM_X threads and written
                // together every BDIM_X iterations...

                if (tidx == 0) {
                        dstV[i] = __tmp;
                }
        }
        return;
}


template<int BDIM_X,
         typename VAL_T>
__device__ void ndotp_log_opdt_d(const int N,
			    const int M,
			    const VAL_T *__restrict__ srcV,
                            const VAL_T *__restrict__ srcM,
                                  VAL_T *__restrict__ dstV) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
         const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        const VAL_T ONEP5 = static_cast<VAL_T>(1.5);

        //#pragma unroll
        for(int i = 0; i < N; i++) {

                VAL_T __tmp = 0;

                //#pragma unroll
                for(int j = 0; j < M; j += BDIM_X) {
                        if (j+tidx < M) {
                                const VAL_T v = srcV[j+tidx];
                                __tmp += -LOG(v)*(ONEP5+LOG(v))*v * srcM[i*M + j+tidx];
                        }
                }
                #pragma unroll
                for(int j = BDIM_X/2; j; j /= 2) {
#if 0
                        __tmp += __shfl_xor_sync(WMASK, __tmp, j, BDIM_X);
#else
                        __tmp += __shfl_down_sync(WMASK, __tmp, j, BDIM_X);
#endif
                }
                // values could be held by BDIM_X threads and written
                // together every BDIM_X iterations...

                if (tidx == 0) {
                        dstV[i] = __tmp;
                }
        }
        return;
}

template<int BDIM_X,
	 typename VAL_T>
__device__ void ndotp_log_csa_d(const int N,
				const int M,
				const VAL_T *__restrict__ srcV,
				const VAL_T *__restrict__ srcM,
				VAL_T *__restrict__ dstV) {

	const int tidx = threadIdx.x;

	const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
	const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));
	// Clamp values
	constexpr VAL_T min = .001;
	constexpr VAL_T max = .999;

	//#pragma unroll
	for(int i = 0; i < N; i++) {

		VAL_T __tmp = 0;

		//#pragma unroll
		for(int j = 0; j < M; j += BDIM_X) {
			if (j+tidx < M) {
				const VAL_T v = MIN(MAX(srcV[j+tidx], min), max);
				__tmp += LOG(-LOG(v)) * srcM[i*M + j+tidx];
			}
		}
		#pragma unroll
		for(int j = BDIM_X/2; j; j /= 2) {
#if 0
			__tmp += __shfl_xor_sync(WMASK, __tmp, j, BDIM_X);
#else
			__tmp += __shfl_down_sync(WMASK, __tmp, j, BDIM_X);
#endif
		}
		// values could be held by BDIM_X threads and written
		// together every BDIM_X iterations...

		if (tidx == 0) {
			dstV[i] = __tmp;
		}
	}
	return;
}


template<int BDIM_X,
         typename REAL_T>
__device__ void fit_opdt(const int delta_nr,
                         const int hr_side,
                         const REAL_T *__restrict__ delta_q,
                         const REAL_T *__restrict__ delta_b,
                         const REAL_T *__restrict__ __msk_data_sh,
                         REAL_T *__restrict__ __h_sh,
                         REAL_T *__restrict__ __r_sh) {
        const int tidx = threadIdx.x;
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        ndotp_log_opdt_d<BDIM_X>(delta_nr, hr_side, __msk_data_sh, delta_q, __r_sh);
        ndotp_d         <BDIM_X>(delta_nr, hr_side, __msk_data_sh, delta_b, __h_sh);
        __syncwarp(WMASK);
        #pragma unroll
        for(int j = tidx; j < delta_nr; j += BDIM_X) {
                __r_sh[j] -= __h_sh[j];
        }
        __syncwarp(WMASK);
}

template<int BDIM_X, typename REAL_T>
__device__ void fit_csa(const int delta_nr,
                        const int hr_side,
                        const REAL_T *__restrict__ fit_matrix,
                        const REAL_T *__restrict__ __msk_data_sh,
                        REAL_T *__restrict__ __r_sh) {
        const int tidx = threadIdx.x;
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        constexpr REAL _n0_const = 0.28209479177387814; // .5 / sqrt(pi)
        ndotp_log_csa_d<BDIM_X>(delta_nr, hr_side, __msk_data_sh, fit_matrix, __r_sh);
        __syncwarp(WMASK);
        if (tidx == 0) {
                __r_sh[0] = _n0_const;
        }
        __syncwarp(WMASK);
}

template<int BDIM_X, ModelType MODEL_T, typename REAL_T>
__device__ void fit_model_coef(const int delta_nr, // delta_nr is number of ODF directions
                               const int hr_side, // hr_side is number of data directions
                               const REAL_T *__restrict__ delta_q,
                               const REAL_T *__restrict__ delta_b, // these are fit matrices the model can use, different for each model
                               const REAL_T *__restrict__ __msk_data_sh, // __msk_data_sh is the part of the data currently being operated on by this block
                               REAL_T *__restrict__ __h_sh, // these last two are modifications to the coefficients that will be returned
                               REAL_T *__restrict__ __r_sh) {
        switch(MODEL_T) {
                case OPDT:
                        fit_opdt<BDIM_X>(delta_nr, hr_side, delta_q, delta_b, __msk_data_sh, __h_sh, __r_sh);
                        break;
                case CSA:
                        fit_csa<BDIM_X>(delta_nr, hr_side, delta_q, __msk_data_sh, __r_sh);
                        break;
                default:
                        printf("FATAL: Invalid Model Type.\n");
                        break;
        }
}

template<int BDIM_X,
         int BDIM_Y,
         int NATTEMPTS,
         ModelType MODEL_T,
         typename REAL_T,
         typename REAL3_T>
__device__ int get_direction_boot_d(
                                curandStatePhilox4_32_10_t *st,
                                const REAL_T max_angle,
                                const REAL_T min_signal,
                                const REAL_T relative_peak_thres,
                                const REAL_T min_separation_angle,
                                REAL3_T dir,
                                const int dimx,
                                const int dimy,
                                const int dimz,
                                const int dimt,
                                const REAL_T *__restrict__ dataf,
                                const int *__restrict__ b0s_mask, // not using this (and its opposite, dwi_mask)
                                                                  // but not clear if it will never be needed so
                                                                  // we'll keep it here for now...
                                const REAL3_T point,
                                const REAL_T *__restrict__ H, 
                                const REAL_T *__restrict__ R,
                                // model unused
                                // max_angle, pmf_threshold from global defines
                                // b0s_mask already passed
                                // min_signal from global defines
                                const int delta_nr,
                                const REAL_T *__restrict__ delta_b,
                                const REAL_T *__restrict__ delta_q, // fit_matrix
                                const int samplm_nr,
                                const REAL_T *__restrict__ sampling_matrix,
                                const REAL3_T *__restrict__ sphere_vertices,
                                const int2 *__restrict__ sphere_edges,
                                const int num_edges,
                                REAL3_T *__restrict__ dirs) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int n32dimt = ((dimt+31)/32)*32;

	extern REAL_T __shared__ __sh[];

	REAL_T *__vox_data_sh = reinterpret_cast<REAL_T *>(__sh);
	REAL_T *__msk_data_sh = __vox_data_sh + BDIM_Y*n32dimt;

	REAL_T *__r_sh = __msk_data_sh + BDIM_Y*n32dimt;
	REAL_T *__h_sh = __r_sh + BDIM_Y*MAX(n32dimt, samplm_nr);

	__vox_data_sh += tidy*n32dimt;
	__msk_data_sh += tidy*n32dimt;

	__r_sh += tidy*MAX(n32dimt, samplm_nr);
	__h_sh += tidy*MAX(n32dimt, samplm_nr);
	
	// compute hr_side (may be passed from python)
	int hr_side = 0;
	for(int j = tidx; j < dimt; j += BDIM_X) {
		hr_side += !b0s_mask[j] ? 1 : 0;
	}
        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                hr_side += __shfl_xor_sync(WMASK, hr_side, i, BDIM_X);
        }

        #pragma unroll
        for(int i = 0; i < NATTEMPTS; i++) {

                const int rv = trilinear_interp_d<BDIM_X>(dimx, dimy, dimz, dimt, -1, dataf, point, __vox_data_sh);

		const int nmsk = maskGet<BDIM_X>(dimt, b0s_mask, __vox_data_sh, __msk_data_sh);

		//if (!tidx && !threadIdx.y && !blockIdx.x) {
		//
		//	printf("interp of %f, %f, %f\n", point.x, point.y, point.z);
		//	printf("hr_side: %d\n", hr_side);
		//	printArray("vox_data", 6, dimt, __vox_data_sh[tidy]);
		//	printArray("msk_data", 6, nmsk, __msk_data_sh[tidy]);
		//}
		//break;

                __syncwarp(WMASK);

                if (rv == 0) {

                        ndotp_d<BDIM_X>(hr_side, hr_side, __msk_data_sh, R, __r_sh);
			//__syncwarp();
			//printArray("__r", 5, hr_side*hr_side, R);
			//printArray("__r_sh", 6, hr_side, __r_sh[tidy]);

                        ndotp_d<BDIM_X>(hr_side, hr_side, __msk_data_sh, H, __h_sh);
			//__syncwarp();
			//printArray("__h_sh", 6, hr_side, __h_sh[tidy]);

                        __syncwarp(WMASK);

                        for(int j = 0; j < hr_side; j += BDIM_X) {
                                if (j+tidx < hr_side) {
#ifdef USE_FIXED_PERMUTATION
                                        const int srcPermInd = fixedPerm[j+tidx];
#else
                                        const int srcPermInd = curand(st) % hr_side;
//                                        if (srcPermInd < 0 || srcPermInd >= hr_side) {
//                                                printf("srcPermInd: %d\n", srcPermInd);
//                                        }
#endif
					__h_sh[j+tidx] += __r_sh[srcPermInd];
					//__h_sh[j+tidx] += __r_sh[j+tidx];
                                }
                        }
			__syncwarp(WMASK);

			//printArray("h+perm(r):", 6, hr_side, __h_sh[tidy]);
			//__syncwarp();
		
			// vox_data[dwi_mask] = masked_data
			maskPut<BDIM_X>(dimt, b0s_mask, __h_sh, __vox_data_sh);
			__syncwarp(WMASK);

			//printArray("vox_data[dwi_mask]:", 6, dimt, __vox_data_sh[tidy]);
			//__syncwarp();

			for(int j = tidx; j < dimt; j += BDIM_X) {
				//__vox_data_sh[j] = MAX(MIN_SIGNAL_P, __vox_data_sh[j]);
				__vox_data_sh[j] = MAX(min_signal, __vox_data_sh[j]);
			}
			__syncwarp(WMASK);

			const REAL_T denom = avgMask<BDIM_X>(dimt, b0s_mask, __vox_data_sh);

			for(int j = tidx; j < dimt; j += BDIM_X) {
				__vox_data_sh[j] /= denom;
			}
			__syncwarp();

			//if (!tidx && !threadIdx.y && !blockIdx.x) {
			//	printf("denom: %f\n", denom);
			//}
			////break;
			//if (!tidx && !threadIdx.y && !blockIdx.x) {
			//
			//	printf("__vox_data_sh:\n");
			//	printArray("vox_data", 6, dimt, __vox_data_sh[tidy]);
			//}
			//break;

			maskGet<BDIM_X>(dimt, b0s_mask, __vox_data_sh, __msk_data_sh);
			__syncwarp(WMASK);

                        fit_model_coef<BDIM_X, MODEL_T>(delta_nr, hr_side, delta_q, delta_b, __msk_data_sh, __h_sh, __r_sh);

                        // __r_sh[tidy] <- python 'coef'

                        ndotp_d<BDIM_X>(samplm_nr, delta_nr, __r_sh, sampling_matrix, __h_sh);

                        // __h_sh[tidy] <- python 'pmf'
                } else {
                        #pragma unroll
                        for(int j = tidx; j < samplm_nr; j += BDIM_X) {
				__h_sh[j] = 0;
                        }
                        // __h_sh[tidy] <- python 'pmf'
                }
                __syncwarp(WMASK);
#if 0
                if (!threadIdx.y && threadIdx.x == 0) {
                        for(int j = 0; j < samplm_nr; j++) {
                                printf("pmf[%d]: %f\n", j, __h_sh[tidy][j]);
                        }
                }
                //return;
#endif
                const REAL_T abs_pmf_thr = PMF_THRESHOLD_P*max_d<BDIM_X>(samplm_nr, __h_sh, REAL_MIN);
                __syncwarp(WMASK);

                #pragma unroll
                for(int j = tidx; j < samplm_nr; j += BDIM_X) {
			const REAL_T __v = __h_sh[j];
			if (__v < abs_pmf_thr) {
				__h_sh[j] = 0;
			}
                }
                __syncwarp(WMASK);
#if 0
                if (!threadIdx.y && threadIdx.x == 0) {
                        printf("abs_pmf_thr: %f\n", abs_pmf_thr);
                        for(int j = 0; j < samplm_nr; j++) {
                                printf("pmfNORM[%d]: %f\n", j, __h_sh[tidy][j]);
                        }
                }
                //return;
#endif
#if 0
                if init:
                        directions = peak_directions(pmf, sphere)[0]
                        return directions
                else:
                        peaks = peak_directions(pmf, sphere)[0]
                        if (len(peaks) > 0):
                                return closest_peak(directions, peaks, cos_similarity)
#endif
                const int ndir = peak_directions_d<BDIM_X,
                                                   BDIM_Y>(__h_sh, dirs,
                                                           sphere_vertices,
                                                           sphere_edges,
                                                           num_edges,
							   samplm_nr,
							   reinterpret_cast<int *>(__r_sh), // reuse __r_sh as shInd in func which is large enough
							   relative_peak_thres,
							   min_separation_angle);
                if (NATTEMPTS == 1) { // init=True...
                        return ndir; // and dirs;
                } else { // init=False...
                        if (ndir > 0) {
                                /*
                                if (!threadIdx.y && threadIdx.x == 0 && ndir > 1) {
                                        printf("NATTEMPTS=5 and ndir: %d!!!\n", ndir);
                                }
                                */
                                REAL3_T peak;
                                const int foundPeak = closest_peak_d<BDIM_X, BDIM_Y, REAL_T, REAL3_T>(max_angle, dir, ndir, dirs, &peak);
                                __syncwarp(WMASK);
                                if (foundPeak) {
                                        if (tidx == 0) {
                                                dirs[0] = peak;
                                        }
                                        return 1;
                                }
                        }
                }
        }
        return 0;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__global__ void getNumStreamlinesBoot_k(
                                    const ModelType model_type,
                                    const REAL_T max_angle,
		                    const REAL_T min_signal,
		                    const REAL_T relative_peak_thres,
	                            const REAL_T min_separation_angle,
		                    const long long rndSeed,
                                    const int nseed,
                                    const REAL3_T *__restrict__ seeds,
                                    const int dimx,
                                    const int dimy,
                                    const int dimz,
                                    const int dimt,
                                    const REAL_T *__restrict__ dataf,
                                    const REAL_T *__restrict__ H,
                                    const REAL_T *__restrict__ R,
		                    const int delta_nr,
                                    const REAL_T *__restrict__ delta_b,
                                    const REAL_T *__restrict__ delta_q,
                                    const int  *__restrict__ b0s_mask, // change to int
		                    const int samplm_nr,
                                    const REAL_T *__restrict__ sampling_matrix,
                                    const REAL3_T *__restrict__ sphere_vertices,
                                    const int2 *__restrict__ sphere_edges,
                                    const int num_edges,
                                    REAL3_T *__restrict__ shDir0,
                                    int *slineOutOff) {

        const int tidx = threadIdx.x;
        const int slid = blockIdx.x*blockDim.y + threadIdx.y;
        const size_t gid = blockIdx.x * blockDim.y * blockDim.x + blockDim.x * threadIdx.y + threadIdx.x;

        if (slid >= nseed) {
                return;
        }

        REAL3_T seed = seeds[slid]; 
        // seed = lin_mat*seed + offset

        REAL3_T *__restrict__ __shDir = shDir0+slid*samplm_nr;

	// const int hr_side = dimt-1;

        curandStatePhilox4_32_10_t st;
        //curand_init(rndSeed, slid + rndOffset, DIV_UP(hr_side, BDIM_X)*tidx, &st); // each thread uses DIV_UP(hr_side/BDIM_X)
        curand_init(rndSeed, gid, 0, &st); // each thread uses DIV_UP(hr_side/BDIM_X)
                                                                                   // elements of the same sequence
        // python:
        //directions = get_direction(None, dataf, dwi_mask, sphere, s, H, R, model, max_angle,
        //                pmf_threshold, b0s_mask, min_signal, fit_matrix,
        //                sampling_matrix, init=True)

	//if (!tidx && !threadIdx.y && !blockIdx.x) {
	//	printf("seed: %f, %f, %f\n", seed.x, seed.y, seed.z);
	//}

        int ndir;
        switch(model_type) {
            case OPDT:
                ndir = get_direction_boot_d<BDIM_X,
                                            BDIM_Y,
                                            1,
                                            OPDT>(
                                                &st,
                                                max_angle,
                                                min_signal,
                                                relative_peak_thres,
                                                min_separation_angle,
                                                MAKE_REAL3(0,0,0),
                                                dimx, dimy, dimz, dimt, dataf,
                                                b0s_mask /* !dwi_mask */,
                                                seed,
                                                H, R,
                                                // model unused
                                                // max_angle, pmf_threshold from global defines
                                                // b0s_mask already passed
                                                // min_signal from global defines
                                                delta_nr,
                                                delta_b, delta_q, // fit_matrix
                                                samplm_nr,
                                                sampling_matrix,
                                                sphere_vertices,
                                                sphere_edges,
                                                num_edges,
                                                __shDir);
                break;
            case CSA:
                ndir = get_direction_boot_d<BDIM_X,
                                            BDIM_Y,
                                            1,
                                            CSA>(
                                                &st,
                                                max_angle,
                                                min_signal,
                                                relative_peak_thres,
                                                min_separation_angle,
                                                MAKE_REAL3(0,0,0),
                                                dimx, dimy, dimz, dimt, dataf,
                                                b0s_mask /* !dwi_mask */,
                                                seed,
                                                H, R,
                                                // model unused
                                                // max_angle, pmf_threshold from global defines
                                                // b0s_mask already passed
                                                // min_signal from global defines
                                                delta_nr,
                                                delta_b, delta_q, // fit_matrix
                                                samplm_nr,
                                                sampling_matrix,
                                                sphere_vertices,
                                                sphere_edges,
                                                num_edges,
                                                __shDir);
                break;
            default:
                printf("FATAL: Invalid Model Type.\n");
                break;
        }

        if (tidx == 0) {
                slineOutOff[slid] = ndir;
        }

        return;
}

template<int BDIM_X,
         int BDIM_Y,
         ModelType MODEL_T,
         typename REAL_T,
         typename REAL3_T>
__device__ int tracker_boot_d(
                        curandStatePhilox4_32_10_t *st,
			            const REAL_T max_angle,
			            const REAL_T tc_threshold,
			            const REAL_T step_size,
			            const REAL_T relative_peak_thres,
			            const REAL_T min_separation_angle,
                         REAL3_T seed,
                         REAL3_T first_step,
                         REAL3_T voxel_size,
                         const int dimx,
                         const int dimy,
                         const int dimz,
                         const int dimt,
                         const REAL_T *__restrict__ dataf,
                         const REAL_T *__restrict__ metric_map,
		                 const int samplm_nr,
                         const REAL3_T *__restrict__ sphere_vertices,
                         const int2 *__restrict__ sphere_edges,
                         const int num_edges,
                        /*BOOT specific params*/
                        const REAL_T min_signal,
                        const int delta_nr,
                        const REAL_T *__restrict__ H,
                        const REAL_T *__restrict__ R,
                        const REAL_T *__restrict__ delta_b,
                        const REAL_T *__restrict__ delta_q,
                        const REAL_T *__restrict__ sampling_matrix,
                        const int    *__restrict__ b0s_mask,
                        /*BOOT specific params*/
                         int *__restrict__ nsteps,
                         REAL3_T *__restrict__ streamline) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        int tissue_class = TRACKPOINT;

        REAL3_T point = seed;
        REAL3_T direction = first_step;
        __shared__ REAL3_T __sh_new_dir[BDIM_Y];

        if (tidx == 0) {
                streamline[0] = point;
#if 0
                if (threadIdx.y == 1) {
                        printf("streamline[0]: %f, %f, %f\n", point.x, point.y, point.z);
                }
#endif
        }
        __syncwarp(WMASK);

        int step_frac = 1;

        int i;
        for(i = 1; i < MAX_SLINE_LEN*step_frac; i++) {
                int ndir = get_direction_boot_d<BDIM_X,
                                                BDIM_Y,
                                                5,
                                                MODEL_T>(
                                                        st,
                                                        max_angle,
                                                        min_signal,
                                                        relative_peak_thres,
                                                        min_separation_angle,
                                                        direction,
                                                        dimx, dimy, dimz, dimt, dataf,
                                                        b0s_mask /* !dwi_mask */,
                                                        point,
                                                        H, R,
                                                        delta_nr,
                                                        delta_b, delta_q, // fit_matrix
                                                        samplm_nr,
                                                        sampling_matrix,
                                                        sphere_vertices,
                                                        sphere_edges,
                                                        num_edges,
                                                        __sh_new_dir + tidy);
                __syncwarp(WMASK);
                direction = __sh_new_dir[tidy];
                __syncwarp(WMASK);

                if (ndir == 0) {
                        break;
                }

                point.x += (direction.x / voxel_size.x) * (step_size / step_frac);
                point.y += (direction.y / voxel_size.y) * (step_size / step_frac);
                point.z += (direction.z / voxel_size.z) * (step_size / step_frac);

                if ((tidx == 0) && ((i % step_frac) == 0)){
                        streamline[i/step_frac] = point;
                }
                __syncwarp(WMASK);

                tissue_class = check_point_d<BDIM_X, BDIM_Y>(tc_threshold, point, dimx, dimy, dimz, metric_map);

                if (tissue_class == ENDPOINT ||
                    tissue_class == INVALIDPOINT ||
                    tissue_class == OUTSIDEIMAGE) {
                        break;
                }
        }
        nsteps[0] = i/step_frac;
        if (((i % step_frac) != 0) && i < step_frac*(MAX_SLINE_LEN - 1)){
                nsteps[0]++;
                if (tidx == 0) {
                        streamline[nsteps[0]] = point;
                }
        }

        return tissue_class;
}

template<int BDIM_X,
         int BDIM_Y,
         ModelType MODEL_T,
         typename REAL_T,
         typename REAL3_T>
__global__ void genStreamlinesMergeBoot_k(
				      const REAL_T max_angle,
				      const REAL_T tc_threshold,
				      const REAL_T step_size,
				      const REAL_T relative_peak_thres,
				      const REAL_T min_separation_angle,
				      const long long rndSeed,
                      const int rndOffset,
                      const int nseed,
                      const REAL3_T *__restrict__ seeds,
                      const int dimx,
                      const int dimy,
                      const int dimz,
                      const int dimt,
                      const REAL_T *__restrict__ dataf,
                      const REAL_T *__restrict__ metric_map,
				      const int samplm_nr,
                      const REAL3_T *__restrict__ sphere_vertices,
                      const int2 *__restrict__ sphere_edges,
                      const int num_edges,
                      /*BOOT specific params*/
				      const REAL_T min_signal,
				      const int delta_nr,
                      const REAL_T *__restrict__ H,
                      const REAL_T *__restrict__ R,
                      const REAL_T *__restrict__ delta_b,
                      const REAL_T *__restrict__ delta_q,
                      const REAL_T *__restrict__ sampling_matrix,
                      const int    *__restrict__ b0s_mask,
                      /*BOOT specific params*/
                      const int    *__restrict__ slineOutOff,
                      REAL3_T *__restrict__ shDir0,
                      int     *__restrict__ slineSeed,
                      int     *__restrict__ slineLen,
                      REAL3_T *__restrict__ sline) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int slid = blockIdx.x*blockDim.y + threadIdx.y;

        const int lid = (tidy*BDIM_X + tidx) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        curandStatePhilox4_32_10_t st;
        // const int gbid = blockIdx.y*gridDim.x + blockIdx.x;
        const size_t gid = blockIdx.x * blockDim.y * blockDim.x + blockDim.x * threadIdx.y + threadIdx.x;
        //curand_init(rndSeed, slid+rndOffset, DIV_UP(hr_side, BDIM_X)*tidx, &st); // each thread uses DIV_UP(HR_SIDE/BDIM_X)
        curand_init(rndSeed, gid+1, 0, &st); // each thread uses DIV_UP(hr_side/BDIM_X)
                                                                                 // elements of the same sequence
        if (slid >= nseed) {
                return;
        }

        REAL3_T seed = seeds[slid]; 

        int ndir = slineOutOff[slid+1]-slineOutOff[slid];
#if 0
        if (threadIdx.y == 0 && threadIdx.x == 0) {
                printf("%s: ndir: %d\n", __func__, ndir);
                for(int i = 0; i < ndir; i++) {
                        printf("__shDir[%d][%d]: (%f, %f, %f)\n",
                                tidy, i, __shDir[tidy][i].x, __shDir[tidy][i].y, __shDir[tidy][i].z);
                }
        }
#endif
        __syncwarp(WMASK);

        int slineOff = slineOutOff[slid];

        for(int i = 0; i < ndir; i++) {
                REAL3_T first_step = shDir0[slid*samplm_nr + i];

		REAL3_T *__restrict__ currSline = sline + slineOff*MAX_SLINE_LEN*2;

                if (tidx == 0) {
                        slineSeed[slineOff] = slid;
                }
#if 0
                if (threadIdx.y == 0 && threadIdx.x == 0) {
                        printf("calling trackerF from: (%f, %f, %f)\n", first_step.x, first_step.y, first_step.z);
                }
#endif

                int stepsB;
                const int tissue_classB = tracker_boot_d<BDIM_X,
                                                         BDIM_Y,
                                                         MODEL_T>(
                                                        &st,
		                		                        max_angle,
			        			                        tc_threshold,
                                                        step_size,
                                                        relative_peak_thres,
                                                        min_separation_angle,
                                                        seed,
                                                        MAKE_REAL3(-first_step.x, -first_step.y, -first_step.z),
                                                        MAKE_REAL3(1, 1, 1),
                                                        dimx, dimy, dimz, dimt, dataf,
                                                        metric_map,
                                                        samplm_nr,
                                                        sphere_vertices,
                                                        sphere_edges,
                                                        num_edges,
                                                        min_signal,
                                                        delta_nr,
                                                        H,
                                                        R,
                                                        delta_b,
                                                        delta_q,
                                                        sampling_matrix,
                                                        b0s_mask,
                                                        &stepsB,
                                                        currSline);

                // reverse backward sline
                for(int j = 0; j < stepsB/2; j += BDIM_X) {
                        if (j+tidx < stepsB/2) {
                                const REAL3_T __p = currSline[j+tidx];
                                currSline[j+tidx] = currSline[stepsB-1 - (j+tidx)];
                                currSline[stepsB-1 - (j+tidx)] = __p;
                        }
                }

                int stepsF;
                const int tissue_classF = tracker_boot_d<BDIM_X,
                                                         BDIM_Y,
                                                         MODEL_T>(
                                                        &st,
     	                    			                max_angle,
		        				                        tc_threshold,
	                				                    step_size,
			    				                        relative_peak_thres,
			            			                    min_separation_angle,
                                                        seed,
                                                        first_step,
                                                        MAKE_REAL3(1, 1, 1),
                                                        dimx, dimy, dimz, dimt, dataf,
                                                        metric_map,
			        			                        samplm_nr,
                                                        sphere_vertices,
                                                        sphere_edges,
                                                        num_edges,
                                                        min_signal,
                                                        delta_nr,
                                                        H,
                                                        R,
                                                        delta_b,
                                                        delta_q,
                                                        sampling_matrix,
                                                        b0s_mask,
                                                        &stepsF,
                                                        currSline + stepsB-1);
                if (tidx == 0) {
                        slineLen[slineOff] = stepsB-1+stepsF;
                }
                
                slineOff += 1;
#if 0
                if (threadIdx.y == 0 && threadIdx.x == 0) {
                        printf("%s: stepsF: %d, tissue_classF: %d\n", __func__, stepsF, tissue_classF);
                }
                __syncwarp(WMASK);
#endif
                //if (/* !return_all || */0 &&
                //    tissue_classF != ENDPOINT &&
                //    tissue_classF != OUTSIDEIMAGE) {
                //        continue;
                //}
                //if (/* !return_all || */ 0 &&
                //    tissue_classB != ENDPOINT &&
                //    tissue_classB != OUTSIDEIMAGE) {
                //        continue;
                //}
        }
        return;
}
