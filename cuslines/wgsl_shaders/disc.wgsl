// disc.wgsl — Disc mesh constant data for PTT (Parallel Transport Tractography).
// Translated from cuslines/metal_shaders/disc.h.
//
// Only SAMPLING_QUALITY=2 is used. The disc mesh defines a triangulated unit disc
// on which PTT samples candidate curvatures (k1, k2). Vertices are 2D coordinates
// (24 vertices * 2 floats = 48 values), faces are triangle index triplets
// (31 faces * 3 ints = 93 values).
//
// Original source: https://github.com/nibrary/nibrary/blob/main/src/math/disc.h
// BSD 3-Clause License, Copyright (c) 2024, Dogu Baran Aydogan.

const DISC_VERT_CNT: u32 = 24u;
const DISC_FACE_CNT: u32 = 31u;

const DISC_VERT: array<f32, 48> = array<f32, 48>(
    -0.99680788, -0.07983759,
    -0.94276539,  0.33345677,
    -0.87928469, -0.47629658,
    -0.72856617,  0.68497542,
    -0.60006556, -0.79995082,
    -0.54129995, -0.02761342,
    -0.39271207,  0.37117272,
    -0.39217391,  0.91989110,
    -0.36362884, -0.40757367,
    -0.22391316, -0.97460910,
    -0.00130022,  0.53966106,
     0.00000000,  0.00000000,
     0.00973999,  0.99995257,
     0.01606516, -0.54289908,
     0.21342395, -0.97695968,
     0.38192071, -0.38666136,
     0.38897094,  0.37442837,
     0.40696681,  0.91344295,
     0.54387161, -0.01477123,
     0.59119367, -0.80652963,
     0.73955688,  0.67309406,
     0.87601150, -0.48229022,
     0.94617928,  0.32364298,
     0.99585368, -0.09096944
);

const DISC_FACE: array<i32, 93> = array<i32, 93>(
     9,  8,  4,
    11, 16, 10,
     5,  8, 11,
     5,  1,  0,
    18, 16, 11,
    11, 15, 18,
    13,  8,  9,
    11,  8, 13,
    13, 15, 11,
    22, 18, 23,
    22, 20, 16,
    16, 18, 22,
    16, 20, 17,
    12, 10, 17,
    17, 10, 16,
    15, 19, 21,
    23, 18, 21,
    21, 18, 15,
     2,  4,  8,
     2,  5,  0,
     8,  5,  2,
     7, 10, 12,
     6,  7,  3,
    10,  7,  6,
     3,  1,  6,
     1,  5,  6,
    11, 10,  6,
     6,  5, 11,
    14, 19, 15,
    15, 13, 14,
    14, 13,  9
);
