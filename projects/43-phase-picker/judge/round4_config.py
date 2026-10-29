"""Predeclared round-4 candidate; feasibility remains BLOCKED until GPU runs.

Shared by the offline schedule generator and MLX reference only. Tests get the
contract from golden metadata and tests/api.py, never import this module.
"""
from pathlib import Path

SEED = 43043
REFERENCE_SEEDS = (43043, 43044, 43045)
STEPS = 36000
BATCH = 32
WIDTHS = (16, 32, 64, 128, 192)
PARAMETERS = 647971
LEARNING_RATE = 0.001
FINAL_LEARNING_RATE = 0.00001
LABEL_SIGMA = 10.0
WARMUP_STEPS = 1000
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data" / "phasepicker"
SCHEDULE = ROOT / "judge" / "golden" / "training_schedule_round4.npz"
WORK_ROOT = ROOT / "judge" / "reference_round4"


def work_dir(seed):
    assert seed in REFERENCE_SEEDS
    return WORK_ROOT / str(seed)


def training_contract():
    return {
        "steps": STEPS, "batch_size": BATCH, "seed": SEED,
        "schedule": SCHEDULE.name,
        "schedule_seed": SEED, "shared_schedule_across_init_seeds": True,
        "learning_rate": LEARNING_RATE, "final_learning_rate": FINAL_LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS, "label_sigma_samples": LABEL_SIGMA,
        "reference_widths": list(WIDTHS), "parameters": PARAMETERS,
        "optimizer": {
            "name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
            "bias_correction": True, "epsilon_outside_sqrt": True,
            "weight_decay": 0.0, "amsgrad": False,
            "gradient_clipping": False, "gradient_accumulation": False,
            "initial_moments": "zero", "first_update_index": 1,
        },
        "initialization": {
            "weights": "independent U(-1/sqrt(fan_in), +1/sqrt(fan_in))",
            "biases": "independent U(-1/sqrt(fan_in), +1/sqrt(fan_in)); NOT zero",
            "fan_in": "input_channels * kernel_size; groups=1",
            "rng": "MLX 0.32.4; reset mx.random.seed(seed) AFTER layer construction",
            "draw_order": "stem, down shallow-to-deep, up deep-to-shallow, head; weight then bias per layer",
            "weight_layout": "out_channels,kernel_size,in_channels",
        },
        "normalization": "float32; per-component demean, joint RMS floor 1e-8 counts, clip [-10,10]",
        "augmentation": "saved zero-filled shifts [-500,500], common +/- polarity, N/E rotation; before normalization",
        "provenance": "Final model only; no early stopping, selection, test/validation inputs or external weights",
        "budget": "Candidate requires measured GPU training <=1800 s per seed; unmeasured in builder sandbox",
        "status": "CANDIDATE_PENDING_THREE_MLX_GPU_RUNS",
    }
