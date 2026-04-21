#!/usr/bin/env python3
"""One-step consistency distillation entrypoint for full LDM -> LCM training.

This script reuses the full-model distillation implementation in
`train_lcm_distill_sd_wds.py` and enforces `num_ddim_timesteps=1`,
so the student is distilled for single-step generation.

Notes:
- Full-weight distillation (no weight pruning / no weight clipping).
- Keeps the original optimizer/EMA/checkpoint logic from the base script.
"""

from train_lcm_distill_sd_wds import main, parse_args


def parse_args_one_step():
    args = parse_args()

    # Force one-step consistency distillation.
    # User-provided values are overridden intentionally.
    args.num_ddim_timesteps = 1
    return args


if __name__ == "__main__":
    args = parse_args_one_step()
    main(args)
