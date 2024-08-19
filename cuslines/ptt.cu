template<typename REAL_T>
__device__ void norm3_d(REAL_T *num, int fail_ind) {
    const REAL_T scale = SQRT(num[0] * num[0] + num[1] * num[1] + num[2] * num[2]);

    if (scale != 0) {
        num[0] /= scale;
        num[1] /= scale;
        num[2] /= scale;
    } else {
        num[fail_ind] = 1.0; // this can happen randomly during propogation, though is exceedingly rare
    }
}

template<typename REAL_T>
__device__ void crossnorm3_d(REAL_T *dest, const REAL_T *src1, const REAL_T *src2, int fail_ind) {
    dest[0] = src1[1] * src2[2] - src1[2] * src2[1];
    dest[1] = src1[2] * src2[0] - src1[0] * src2[2];
    dest[2] = src1[0] * src2[1] - src1[1] * src2[0];

    norm3_d(dest, fail_ind);
}

template<typename REAL_T, typename REAL3_T>
__device__ REAL_T interp4_d(const REAL3_T pos, const REAL_T* frame, const REAL_T *__restrict__ pmf,
                            const int dimx, const int dimy, const int dimz, const int dimt,
                            const REAL3_T *__restrict__ odf_sphere_vertices) {
    int closest_odf_idx = 0;
    REAL_T __max_cos = 0;
    for (int ii = 0; ii < dimt; ii++) {
        REAL_T cos_sim = FABS(
            odf_sphere_vertices[ii].x * frame[0] \
            + odf_sphere_vertices[ii].y * frame[1] \
            + odf_sphere_vertices[ii].z * frame[2]);
        if (cos_sim > __max_cos) {
            __max_cos = cos_sim;
            closest_odf_idx = ii;
        }
    }

    const int rv = trilinear_interp_d<THR_X_SL>(dimx, dimy, dimz, dimt, closest_odf_idx, pmf, pos, &__max_cos);

#if 0
    printf("inerpolated %f at %f, %f, %f, %i\n", __max_cos, pos.x, pos.y, pos.z, closest_odf_idx);
#endif

    if (rv != 0) {
        return -1;
    } else {
        return __max_cos;
    }
}

template<typename REAL_T>
__device__ void prepare_propagator_d(REAL_T k1, REAL_T k2, REAL_T arclength,
                                     REAL_T *propagator) {
    if ((FABS(k1) < K_SMALL) && (FABS(k2) < K_SMALL)) {
        propagator[0] = arclength;
        propagator[1] = 0;
        propagator[2] = 0;
        propagator[3] = 1;
        propagator[4] = 0;
        propagator[5] = 0;
        propagator[6] = 0;
        propagator[7] = 0;
        propagator[8] = 1;
    } else {
        if (FABS(k1) < K_SMALL) {
            k1 = K_SMALL;
        }
        if (FABS(k2) < K_SMALL) {
            k2 = K_SMALL;
        }
        const REAL_T k     = SQRT(k1*k1+k2*k2);
        const REAL_T sinkt = SIN(k*arclength);
        const REAL_T coskt = COS(k*arclength);
        const REAL_T kk    = 1/(k*k);

        propagator[0] = sinkt/k;
        propagator[1] = k1*(1-coskt)*kk;
        propagator[2] = k2*(1-coskt)*kk;
        propagator[3] = coskt;
        propagator[4] = k1*sinkt/k;
        propagator[5] = k2*sinkt/k;
        propagator[6] = -propagator[5];
        propagator[7] = k1*k2*(coskt-1)*kk;
        propagator[8] = (k1*k1+k2*k2*coskt)*kk;
    }
}

template<bool IS_INIT, typename REAL_T>
__device__ void get_probing_frame_d(const REAL_T* frame, curandStatePhilox4_32_10_t *st, REAL_T* probing_frame) {
    if (IS_INIT) {
        for (int ii = 0; ii < 3; ii++) { // tangent
            probing_frame[ii] = frame[ii];
        }
        if ((probing_frame[0] != 0) && (probing_frame[1] != 0)) { // norm
            probing_frame[3] = -probing_frame[1];
            probing_frame[4] = probing_frame[0];
            probing_frame[5] = 0;
        } else {
            probing_frame[3] = 0;
            probing_frame[4] = -probing_frame[2];
            probing_frame[5] = 0;
        }

        norm3_d(probing_frame, 0); // tangent
        norm3_d(probing_frame + 3, 1); // norm
        crossnorm3_d(probing_frame + 2*3, probing_frame, probing_frame + 3, 2); // binorm
    } else {
        for (int ii = 0; ii < 9; ii++) {
            probing_frame[ii] =  frame[ii];
        }
    }
}

template<typename REAL_T>
__device__ void propogate_frame_d(REAL_T* propagator, REAL_T* frame, REAL_T* direc) {
    REAL_T __tmp[3];

    for (int ii = 0; ii < 3; ii++) {
        direc[ii]       = propagator[0]*frame[ii] + propagator[1]*frame[3+ii] + propagator[2]*frame[6+ii];
        __tmp[ii]       = propagator[3]*frame[ii] + propagator[4]*frame[3+ii] + propagator[5]*frame[6+ii];
        frame[2*3 + ii] = propagator[6]*frame[ii] + propagator[7]*frame[3+ii] + propagator[8]*frame[6+ii];
    }

#if 1
    norm3_d(__tmp, 0); // normalize tangent
    crossnorm3_d(frame + 3, frame + 2*3, __tmp, 1); // calc normal
    crossnorm3_d(frame + 2*3, __tmp, frame + 3, 2); // calculate binorm from tangent, norm
#else
    norm3_d(__tmp, 0); // normalize tangent
    norm3_d(frame + 2*3, 2); // normalize binorm
    crossnorm3_d(frame + 3, frame + 2*3, __tmp, 1); // calculate normal from binorm, tangent
#endif

    for (int ii = 0; ii < 3; ii++) {
        frame[ii] = __tmp[ii];
    }
}

template<typename REAL_T, typename REAL3_T>
__device__ REAL_T calculate_data_support_d(REAL_T support,
                                           const REAL3_T pos, const REAL_T *__restrict__ pmf,
                                           const int dimx, const int dimy, const int dimz, const int dimt,
                                           const REAL_T probe_step_size,
                                           const REAL3_T *__restrict__ odf_sphere_vertices,
                                           REAL_T k1, REAL_T k2,
                                           REAL_T* probing_frame) {
    REAL_T probing_prop[9];
    REAL_T direc[3];
    REAL3_T probing_pos;
    REAL_T fod_amp;

    prepare_propagator_d(k1, k2, probe_step_size, probing_prop);
    probing_pos.x = pos.x;
    probing_pos.y = pos.y;
    probing_pos.z = pos.z;

    for (int ii = 0; ii < PROBE_QUALITY; ii++) { // we spend about 2/3 of our time in this loop when doing PTT
        propogate_frame_d(probing_prop, probing_frame, direc);

        probing_pos.x += direc[0];
        probing_pos.y += direc[1];
        probing_pos.z += direc[2];

        fod_amp = interp4_d(probing_pos, probing_frame, pmf,
                            dimx, dimy, dimz, dimt,
                            odf_sphere_vertices);
        if (!ALLOW_WEAK_LINK && (fod_amp < PMF_THRESHOLD_P)) {
            return 0;
        }
        support += fod_amp;
    }
    return support;
}

template<int BDIM_X,
         int BDIM_Y,
         bool IS_INIT,
         typename REAL_T,
         typename REAL3_T>
__device__ int get_direction_ptt_d(
    curandStatePhilox4_32_10_t *st,
    const REAL_T *__restrict__ pmf,
    const REAL_T max_angle,
    const REAL_T step_size,
    REAL3_T dir,
    REAL_T *__frame_sh,
    const int dimx, const int dimy, const int dimz, const int dimt,
    REAL3_T pos,
    const REAL3_T *__restrict__ odf_sphere_vertices,
    REAL3_T *__restrict__ dirs) {
    // Aydogan DB, Shi Y. Parallel Transport Tractography. IEEE Trans
    // Med Imaging. 2021 Feb;40(2):635-647. doi: 10.1109/TMI.2020.3034038.
    // Epub 2021 Feb 2. PMID: 33104507; PMCID: PMC7931442.
    // https://github.com/nibrary/nibrary/blob/main/src/dMRI/tractography/algorithms/ptt
    // Assumes probe count 1, data_support_exponent 1 for now
    // Implemented with new CDF sampling strategy
    // And using initial directions from voxel-wise peaks as in DIPY

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
    const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	REAL_T __shared__ face_cdf_sh[BDIM_Y*DISC_FACE_CNT];
    REAL_T __shared__ vert_pdf_sh[BDIM_Y*DISC_VERT_CNT];
    REAL_T __shared__ first_val_sh[BDIM_Y];

    REAL_T *__face_cdf_sh = face_cdf_sh + tidy*DISC_FACE_CNT;
    REAL_T *__vert_pdf_sh = vert_pdf_sh + tidy*DISC_VERT_CNT;
    REAL_T *__first_val_sh = first_val_sh + tidy;

    const REAL_T max_curvature = SIN(max_angle / 2) / step_size; // bigger numbers means wiggle more
    const REAL_T probe_step_size = ((step_size / 2) / (PROBE_QUALITY - 1));

    REAL_T __tmp;

    __syncwarp(WMASK);
    if (IS_INIT) {
        if (tidx==0) {
            __frame_sh[0] = dir.x;
            __frame_sh[1] = dir.y;
            __frame_sh[2] = dir.z;
        }
    }
    if (tidx==0) {
        *__first_val_sh = interp4_d(pos, __frame_sh, pmf,
                                    dimx, dimy, dimz, dimt,
                                    odf_sphere_vertices);
    }
    __syncwarp(WMASK);

    // Calculate __vert_pdf_sh
    REAL_T probing_frame[9];
    REAL_T k1_probe, k2_probe;
    bool support_found = 0;
    for (int ii = tidx; ii < DISC_VERT_CNT; ii += BDIM_X) {
        k1_probe = DISC_VERT[ii*2] * max_curvature;
        k2_probe = DISC_VERT[ii*2+1] * max_curvature;

        get_probing_frame_d<IS_INIT>(__frame_sh, st, probing_frame);

        const REAL_T this_support = calculate_data_support_d(
            *__first_val_sh,
            pos, pmf, dimx, dimy, dimz, dimt,
            probe_step_size,
            odf_sphere_vertices,
            k1_probe, k2_probe,
            probing_frame);

#if 0
        if (threadIdx.y == 1 && ii == 0) { 
            printf("    k1_probe: %f, k2_probe %f, support %f for id: %i\n", k1_probe, k2_probe, this_support, tidx);
        }
#endif

        if (this_support < NORM_MIN_SUPPORT) {
            __vert_pdf_sh[ii] = 0;
        } else {
            __vert_pdf_sh[ii] = this_support;
            support_found = 1;
        }
    }
    const int __msk = __ballot_sync(WMASK, support_found);
    if (__msk == 0) {
        return 0;
    }

#if 0
    __syncwarp(WMASK);
    if (threadIdx.y == 1 && threadIdx.x == 0) {
        printArrayAlways("VERT PDF", 8, DISC_VERT_CNT, __vert_pdf_sh);
    }
    __syncwarp(WMASK);
#endif

    // Initialize __face_cdf_sh
    for (int ii = tidx; ii < DISC_FACE_CNT; ii+=BDIM_X) { 
        __face_cdf_sh[ii] = 0;
    }
    __syncwarp(WMASK);

    // Move vert to face
    for (int ii = tidx; ii < DISC_FACE_CNT; ii+=BDIM_X) {
        bool all_verts_valid = 1;
        for (int jj = 0; jj < 3; jj++) {
            REAL_T vert_val = __vert_pdf_sh[DISC_FACE[ii*3 + jj]];
            if (vert_val == 0) {
                all_verts_valid = IS_INIT; // On init, even go with faces that are not fully supported
            }
            __face_cdf_sh[ii] += vert_val;
        }
        if (!all_verts_valid) {
            __face_cdf_sh[ii] = 0;
        }
    }
    __syncwarp(WMASK);

#if 0
    __syncwarp(WMASK);
    if (threadIdx.y == 1 && threadIdx.x == 0) {
        printArrayAlways("Face PDF", 8, DISC_FACE_CNT, __face_cdf_sh);
    }
    __syncwarp(WMASK);
#endif

    // Prefix sum __face_cdf_sh and return 0 if all 0
    prefix_sum_sh_d<BDIM_X>(__face_cdf_sh, DISC_FACE_CNT);
    REAL_T last_cdf = __face_cdf_sh[DISC_FACE_CNT - 1];

    if (last_cdf == 0) {
        return 0;
    }

#if 0
    __syncwarp(WMASK);
    if (threadIdx.y == 1 && threadIdx.x == 0) {
        printArrayAlways("Face CDF", 8, DISC_FACE_CNT, __face_cdf_sh);
    }
    __syncwarp(WMASK);
#endif

    // Sample random valid faces randomly
    REAL_T r1, r2;
    for (int ii = 0; ii < TRIES_PER_REJECTION_SAMPLING / BDIM_X; ii++) {
        r1 = curand_uniform(st);
        r2 = curand_uniform(st);
		if (r1 + r2 > 1) {
			r1 = 1 - r1;
			r2 = 1 - r2;
		}

        __tmp = curand_uniform(st) * last_cdf;
		int jj;
		for (jj = 0; jj < DISC_FACE_CNT; jj++) {
			if (__face_cdf_sh[jj] >= __tmp)
				break;
		}

        const REAL_T vx0 = max_curvature * DISC_VERT[DISC_FACE[jj*3]*2];
        const REAL_T vx1 = max_curvature * DISC_VERT[DISC_FACE[jj*3+1]*2];
        const REAL_T vx2 = max_curvature * DISC_VERT[DISC_FACE[jj*3+2]*2];

        const REAL_T vy0 = max_curvature * DISC_VERT[DISC_FACE[jj*3]*2 + 1];
        const REAL_T vy1 = max_curvature * DISC_VERT[DISC_FACE[jj*3+1]*2 + 1];
        const REAL_T vy2 = max_curvature * DISC_VERT[DISC_FACE[jj*3+2]*2 + 1];

        k1_probe = vx0 + r1 * (vx1 - vx0) + r2 * (vx2 - vx0);
        k2_probe = vy0 + r1 * (vy1 - vy0) + r2 * (vy2 - vy0);

        get_probing_frame_d<IS_INIT>(__frame_sh, st, probing_frame);

        const REAL_T this_support = calculate_data_support_d(*__first_val_sh,
                                                             pos, pmf, dimx, dimy, dimz, dimt,
                                                             probe_step_size,
                                                             odf_sphere_vertices,
                                                             k1_probe, k2_probe,
                                                             probing_frame);


        __syncwarp(WMASK);
        int winning_lane = -1; // -1 indicates nobody won
        int __msk = __ballot_sync(WMASK, this_support >= NORM_MIN_SUPPORT);
        if (__msk != 0) {
            REAL_T group_max_support = this_support;
            #pragma unroll
            for(int j = 1; j < PROBABILISTIC_GROUP_SZ; j *= 2) {
                __tmp = __shfl_xor_sync(WMASK, group_max_support, j, BDIM_X);
                group_max_support = MAX(group_max_support, __tmp);
            }

            __msk &= __ballot_sync(WMASK, this_support == group_max_support);
            winning_lane = __ffs(__msk) - 1;
        }
        if (winning_lane != -1) {
            if (tidx == winning_lane) {
#if 0
                if (threadIdx.y == 1) {
                    printf("winning k1 %f, k2 %f, cdf %f, cdf_idx %i", k1_probe, k2_probe, __tmp, jj);
                }
#endif
                if (IS_INIT) {
                    dirs[0] = dir;
                } else {
                    REAL_T __prop[9];
                    REAL_T __dir[3];
                    prepare_propagator_d(k1_probe, k2_probe, step_size/STEP_FRAC, __prop);
                    propogate_frame_d(__prop, probing_frame, __dir);
                    norm3_d(__dir, 0); // this will be scaled by the generic stepping code
                    dirs[0] = (REAL3_T) {__dir[0], __dir[1], __dir[2]};
                }

                for (int jj = 0; jj < 9; jj++) {
                    __frame_sh[jj] = probing_frame[jj];
                }
            }
            __syncwarp(WMASK);
            return 1;
        }
    }
    return 0;
}


template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ bool init_frame_ptt_d(
    curandStatePhilox4_32_10_t *st,
    const REAL_T *__restrict__ pmf,
    const REAL_T max_angle,
    const REAL_T step_size,
    REAL3_T first_step,
    const int dimx, const int dimy, const int dimz, const int dimt,
    REAL3_T seed,
    const REAL3_T *__restrict__ sphere_vertices,
    REAL_T* __frame) {
    const int tidx = threadIdx.x;

    const int lid = (threadIdx.y*BDIM_X + tidx) % 32;
    const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

    bool init_norm_success;
    REAL3_T tmp;

    // Here we probabilistic find a good intial normal for this initial direction 
    init_norm_success = (bool) get_direction_ptt_d<BDIM_X, BDIM_Y, 1>(
        st,
        pmf,
        max_angle,
        step_size,
        MAKE_REAL3(-first_step.x, -first_step.y, -first_step.z),
        __frame,
        dimx, dimy, dimz, dimt,
        seed,
        sphere_vertices,
        &tmp);
    __syncwarp(WMASK);

    if (!init_norm_success) {
        // try the other direction
        init_norm_success = (bool) get_direction_ptt_d<BDIM_X, BDIM_Y, 1>(
            st,
            pmf,
            max_angle,
            step_size,
            MAKE_REAL3(first_step.x, first_step.y, first_step.z),
            __frame,
            dimx, dimy, dimz, dimt,
            seed,
            sphere_vertices,
            &tmp);
        __syncwarp(WMASK);

        if (!init_norm_success) { // This is rare
            return false;
        } else {
            if (tidx == 0) {
                for (int ii = 0; ii < 9; ii++) {
                    __frame[ii] = -__frame[ii];
                }
            }
            __syncwarp(WMASK);
        }
    }
    if (tidx == 0) {
        for (int ii = 0; ii < 9; ii++) {
            __frame[9+ii] = -__frame[ii]; // save flipped frame for second run
        }
    }
    __syncwarp(WMASK);
    return true;
}
