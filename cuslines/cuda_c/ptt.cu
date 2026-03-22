template<int FAIL_IND>
__device__ __forceinline__ void norm3_d(float *num) {
    const float inv_scale = rnorm3df(num[0], num[1], num[2]);

    if (isfinite(inv_scale) && inv_scale > 0.0f && inv_scale < PTT_INV_NORM_EPS) {
        num[0] *= inv_scale;
        num[1] *= inv_scale;
        num[2] *= inv_scale;
    } else {
        num[0] = num[1] = num[2] = 0;
        num[FAIL_IND] = 1.0; // this can happen randomly during propogation, though is exceedingly rare
    }
}

template<int FAIL_IND>
__device__ __forceinline__ void crossnorm3_d(float *dest, const float *src1, const float *src2) {
    dest[0] = src1[1] * src2[2] - src1[2] * src2[1];
    dest[1] = src1[2] * src2[0] - src1[0] * src2[2];
    dest[2] = src1[0] * src2[1] - src1[1] * src2[0];

    norm3_d<FAIL_IND>(dest);
}

__device__ float interp4_d(const float3 pos, const float* frame,
                           const cudaTextureObject_t *__restrict__ pmf,
                           const cudaTextureObject_t *__restrict__ sphere_vertices_lut) {
    float3 uvw = {  // Map from -1,1 to 0,1 for texture lookup
        fmaf(frame[0], 0.5f, 0.5f),
        fmaf(frame[1], 0.5f, 0.5f),
        fmaf(frame[2], 0.5f, 0.5f)
    };
    const int odf_idx = static_cast<int>(tex3D<float>(*sphere_vertices_lut, uvw.z, uvw.y, uvw.x));

    const int grid_col = odf_idx & WIDTH_MASK;
    const int grid_row = odf_idx >> LOG2_WIDTH;

    const float x_query = (float)(grid_col * DIMX) + pos.x;
    const float y_query = (float)(grid_row * DIMY) + pos.y;
    return tex3D<float>(*pmf, x_query, y_query, pos.z);
}

__device__ void prepare_propagator_d(float k1, float k2, float arclength,
                                     float *propagator) {
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
        const float k     = SQRT(k1*k1+k2*k2);
        const float sinkt = SIN(k*arclength);
        const float coskt = COS(k*arclength);
        const float kk    = 1/(k*k);

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

__device__ void random_normal(curandStatePhilox4_32_10_t *st, float* probing_frame) {
    probing_frame[3] = curand_normal(st);
    probing_frame[4] = curand_normal(st);
    probing_frame[5] = curand_normal(st);
    float dot = probing_frame[3]*probing_frame[0]
        + probing_frame[4]*probing_frame[1]
        + probing_frame[5]*probing_frame[2];

    probing_frame[3] -= dot*probing_frame[0];
    probing_frame[4] -= dot*probing_frame[1];
    probing_frame[5] -= dot*probing_frame[2];
    float n2 = probing_frame[3]*probing_frame[3]
        + probing_frame[4]*probing_frame[4]
        + probing_frame[5]*probing_frame[5];

    if (n2 < PTT_NORM_EPS) {
        float abs_x = FABS(probing_frame[0]);
        float abs_y = FABS(probing_frame[1]);
        float abs_z = FABS(probing_frame[2]);

        if (abs_x <= abs_y && abs_x <= abs_z) {
            probing_frame[3] = 0.0;
            probing_frame[4] = probing_frame[2];
            probing_frame[5] = -probing_frame[1];
        } 
        else if (abs_y <= abs_z) {
            probing_frame[3] = -probing_frame[2];
            probing_frame[4] = 0.0;
            probing_frame[5] = probing_frame[0];
        } 
        else {
            probing_frame[3] = probing_frame[1];
            probing_frame[4] = -probing_frame[0];
            probing_frame[5] = 0.0;
        }
    }
}

template<bool IS_INIT>
__device__ void get_probing_frame_d(const float* frame, curandStatePhilox4_32_10_t *st, float* probing_frame) {
    if (IS_INIT) {
        for (int ii = 0; ii < 3; ii++) { // tangent
            probing_frame[ii] = frame[ii];
        }
        norm3_d<0>(probing_frame);

        random_normal(st, probing_frame);
        norm3_d<1>(probing_frame + 3); // norm

        // calculate binorm
        crossnorm3_d<2>(probing_frame + 2*3, probing_frame, probing_frame + 3); // binorm
    } else {
        for (int ii = 0; ii < 9; ii++) {
            probing_frame[ii] =  frame[ii];
        }
    }
}

__device__ void propagate_frame_d(float* propagator, float* frame, float* direc) {
    float __tmp[3];

    for (int ii = 0; ii < 3; ii++) {
        direc[ii]       = propagator[0]*frame[ii] + propagator[1]*frame[3+ii] + propagator[2]*frame[6+ii];
        __tmp[ii]       = propagator[3]*frame[ii] + propagator[4]*frame[3+ii] + propagator[5]*frame[6+ii];
        frame[2*3 + ii] = propagator[6]*frame[ii] + propagator[7]*frame[3+ii] + propagator[8]*frame[6+ii];
    }

    norm3_d<0>(__tmp); // normalize tangent
    crossnorm3_d<1>(frame + 3, frame + 2*3, __tmp); // calc normal
    crossnorm3_d<2>(frame + 2*3, __tmp, frame + 3); // calculate binorm from tangent, norm

    for (int ii = 0; ii < 3; ii++) {
        frame[ii] = __tmp[ii];
    }
}

__device__ float calculate_data_support_d(float support,
                                           const float3 pos, const cudaTextureObject_t *__restrict__ pmf,
                                           const cudaTextureObject_t *__restrict__ sphere_vertices_lut,
                                           float k1, float k2,
                                           float* probing_frame) {
    
    float probing_prop[9];
    float direc[3];
    float3 probing_pos;
    prepare_propagator_d(
        k1, k2,
        PROBE_STEP_SIZE, probing_prop);
    probing_pos.x = pos.x;
    probing_pos.y = pos.y;
    probing_pos.z = pos.z;

    for (int ii = 0; ii < PROBE_QUALITY; ii++) { // we spend about 2/3 of our time in this loop when doing PTT
        propagate_frame_d(
            probing_prop,
            probing_frame,
            direc);

        probing_pos.x += direc[0];
        probing_pos.y += direc[1];
        probing_pos.z += direc[2];

        const float fod_amp = interp4_d( // This is the most expensive call
            probing_pos, probing_frame, pmf,
            sphere_vertices_lut);

        if (!ALLOW_WEAK_LINK && (fod_amp < PMF_THRESHOLD_P)) {
            return 0;
        }
        support += fod_amp;
    }
    return support;
}

template<int BDIM_X,
         int BDIM_Y,
         bool IS_INIT>
__device__ int get_direction_ptt_d(
    curandStatePhilox4_32_10_t *st,
    const cudaTextureObject_t *__restrict__ pmf,
    float3 dir,
    float *__frame_sh,
    float3 pos,
    const cudaTextureObject_t *__restrict__ sphere_vertices_lut,
    float3 *__restrict__ dirs) {
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

	__shared__ float face_cdf_sh[BDIM_Y*DISC_FACE_CNT];
    __shared__ float vert_pdf_sh[BDIM_Y*DISC_VERT_CNT];

    float *__face_cdf_sh = face_cdf_sh + tidy*DISC_FACE_CNT;
    float *__vert_pdf_sh = vert_pdf_sh + tidy*DISC_VERT_CNT;

    float __tmp;

    __syncwarp(WMASK);
    if (IS_INIT) {
        if (tidx==0) {
            __frame_sh[0] = dir.x;
            __frame_sh[1] = dir.y;
            __frame_sh[2] = dir.z;
        }
    }

    const float first_val = interp4_d(
        pos, __frame_sh, pmf,
        sphere_vertices_lut);
    __syncwarp(WMASK);

    // Calculate __vert_pdf_sh
    float probing_frame[9];
    float k1_probe, k2_probe;
    bool support_found = 0;
    for (int ii = tidx; ii < DISC_VERT_CNT; ii += BDIM_X) {
        k1_probe = DISC_VERT[ii*2] * MAX_CURVATURE;
        k2_probe = DISC_VERT[ii*2+1] * MAX_CURVATURE;

        get_probing_frame_d<IS_INIT>(__frame_sh, st, probing_frame);

        const float this_support = calculate_data_support_d(
            first_val,
            pos, pmf,
            sphere_vertices_lut,
            k1_probe, k2_probe,
            probing_frame);

#if 0
        if (threadIdx.y == 1 && ii == 0) { 
            printf("    k1_probe: %f, k2_probe %f, support %f for id: %i\n", k1_probe, k2_probe, this_support, tidx);
        }
#endif

        if (this_support < PROBE_QUALITY * PMF_THRESHOLD_P) {
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
            float vert_val = __vert_pdf_sh[DISC_FACE[ii*3 + jj]];
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
    float last_cdf = __face_cdf_sh[DISC_FACE_CNT - 1];

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
    float r1, r2;
    for (int ii = 0; ii < TRIES_PER_REJECTION_SAMPLING / BDIM_X; ii++) {
        r1 = curand_uniform(st);
        r2 = curand_uniform(st);
		if (r1 + r2 > 1) {
			r1 = 1 - r1;
			r2 = 1 - r2;
		}

        __tmp = curand_uniform(st) * last_cdf;
		int jj;
		for (jj = 0; jj < DISC_FACE_CNT - 1; jj++) {
			if (__face_cdf_sh[jj] >= __tmp)
				break;
		}

        const float vx0 = MAX_CURVATURE * DISC_VERT[DISC_FACE[jj*3]*2];
        const float vx1 = MAX_CURVATURE * DISC_VERT[DISC_FACE[jj*3+1]*2];
        const float vx2 = MAX_CURVATURE * DISC_VERT[DISC_FACE[jj*3+2]*2];

        const float vy0 = MAX_CURVATURE * DISC_VERT[DISC_FACE[jj*3]*2 + 1];
        const float vy1 = MAX_CURVATURE * DISC_VERT[DISC_FACE[jj*3+1]*2 + 1];
        const float vy2 = MAX_CURVATURE * DISC_VERT[DISC_FACE[jj*3+2]*2 + 1];

        k1_probe = vx0 + r1 * (vx1 - vx0) + r2 * (vx2 - vx0);
        k2_probe = vy0 + r1 * (vy1 - vy0) + r2 * (vy2 - vy0);

        get_probing_frame_d<IS_INIT>(__frame_sh, st, probing_frame);

        const float this_support = calculate_data_support_d(first_val,
                                                             pos, pmf,
                                                             sphere_vertices_lut,
                                                             k1_probe, k2_probe,
                                                             probing_frame);


        __syncwarp(WMASK);

        int winning_lane = -1; // -1 indicates nobody won
        int __msk = __ballot_sync(WMASK, this_support >= PROBE_QUALITY * PMF_THRESHOLD_P);
        if (__msk != 0) {
            winning_lane = __ffs(__msk) - 1;
        }
        if (winning_lane != -1) {
            if (tidx == winning_lane) {
                if (IS_INIT) {
                    dirs[0] = dir;
                } else {
                    float __prop[9];
                    float __dir[3];
                    prepare_propagator_d(k1_probe, k2_probe, STEP_SIZE/STEP_FRAC, __prop);
                    get_probing_frame_d<0>(__frame_sh, st, probing_frame);
                    propagate_frame_d(__prop, probing_frame, __dir);
                    norm3_d<0>(__dir); // this will be scaled by the generic stepping code
                    dirs[0] = MAKE_REAL3(__dir[0], __dir[1], __dir[2]);
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
         int BDIM_Y>
__device__ bool init_frame_ptt_d(
    curandStatePhilox4_32_10_t *st,
    const cudaTextureObject_t *__restrict__ pmf,
    float3 first_step,
    float3 seed,
    const cudaTextureObject_t *__restrict__ sphere_vertices_lut,
    float* __frame) {
    const int tidx = threadIdx.x;

    const int lid = (threadIdx.y*BDIM_X + tidx) % 32;
    const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

    bool init_norm_success;
    float3 tmp;

    // Here we probabilistic find a good intial normal for this initial direction 
    init_norm_success = (bool) get_direction_ptt_d<BDIM_X, BDIM_Y, 1>(
        st,
        pmf,
        MAKE_REAL3(-first_step.x, -first_step.y, -first_step.z),
        __frame,
        seed,
        sphere_vertices_lut,
        &tmp);
    __syncwarp(WMASK);

    if (!init_norm_success) {
        // try the other direction
        init_norm_success = (bool) get_direction_ptt_d<BDIM_X, BDIM_Y, 1>(
            st,
            pmf,
            MAKE_REAL3(first_step.x, first_step.y, first_step.z),
            __frame,
            seed,
            sphere_vertices_lut,
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
