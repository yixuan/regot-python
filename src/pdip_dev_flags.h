#ifndef REGOT_PDIP_DEV_FLAGS_H
#define REGOT_PDIP_DEV_FLAGS_H

// -----------------------------------------------------------------------------
// PDIP CG / FP developer build (REGOT_PDIP_DEV)
//
// When defined at compile time (see setup.py: set REGOT_PDIP_DEV=1 for the
// build step), extra diagnostics are enabled:
//   - PDIP-CG: per-phase wall-time breakdown, optional PDIP_CG_TIMING, file
//     pdip_cg_timing.txt, optional env PDIP_SPARSITY_KEEP for sparsity tuning.
//   - PDIP-FP: detailed per-phase times in PDIPResult (t_build_B, t_chol_*).
//
// Default / user installs: macro undefined — no env reads for profiling, no
// timing file, no per-phase accumulation; PDIPResult timing fields stay zero.
// -----------------------------------------------------------------------------

#endif  // REGOT_PDIP_DEV_FLAGS_H
