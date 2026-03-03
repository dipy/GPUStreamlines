// globals.wgsl — Constants for GPU streamline generation.
// Mirrors cuslines/metal_shaders/globals.h (Metal) and cuslines/cuda_c/globals.h (CUDA).
// WebGPU/WGSL only supports f32 (no f64), so REAL_SIZE is always 4.

// ── precision ────────────────────────────────────────────────────────
const REAL_SIZE: u32 = 4u;
const REAL_MAX: f32 = 3.4028235e+38;
const REAL_MIN: f32 = -3.4028235e+38;

// ── geometry constants ───────────────────────────────────────────────
const MAX_SLINE_LEN: i32 = 501;
const PMF_THRESHOLD_P: f32 = 0.05;

const THR_X_BL: u32 = 64u;
const THR_X_SL: u32 = 32u;
const BLOCK_Y: u32 = 2u;  // THR_X_BL / THR_X_SL
const MAX_N32DIMT: u32 = 512u;
const MAX_SLINES_PER_SEED: u32 = 10u;

const EXCESS_ALLOC_FACT: u32 = 2u;
const NORM_EPS: f32 = 1e-8;

// ── model types ──────────────────────────────────────────────────────
const MODEL_OPDT: i32 = 0;
const MODEL_CSA: i32 = 1;
const MODEL_PROB: i32 = 2;
const MODEL_PTT: i32 = 3;

// ── point status codes ──────────────────────────────────────────────
const OUTSIDEIMAGE: i32 = 0;
const INVALIDPOINT: i32 = 1;
const TRACKPOINT: i32 = 2;
const ENDPOINT: i32 = 3;

// ── utility functions ───────────────────────────────────────────────
fn div_up(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
