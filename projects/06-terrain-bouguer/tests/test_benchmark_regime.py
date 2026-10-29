"""Lock the medium benchmark geometry and its altitude dependence."""
import numpy as np
import pytest
from outcrop_harness import PATHS, golden
from test_terrain import INPUTS, OUTPUTS, run
from tolerances import FAR_RTOL, NEAR_RTOL, SLAB_RTOL


def altitude_case(index):
    data = golden("benchmark_regime")
    sl = slice(index * 256, (index + 1) * 256)
    return {key: data[key][sl] if key in ("coordinates", *OUTPUTS) else data[key]
            for key in (*INPUTS, *OUTPUTS)}


def compare(actual, reference, height):
    errors = {}
    for key in OUTPUTS:
        ref = np.asarray(reference[key], dtype=np.float64)
        errors[key] = float(np.max(np.abs(np.asarray(actual[key], dtype=np.float64) - ref))
                            / np.max(np.abs(ref)))
    limit = NEAR_RTOL if height == 700 else FAR_RTOL
    limits = {key: SLAB_RTOL if key == "slab_gz" else limit for key in OUTPUTS}
    assert all(errors[key] <= limits[key] for key in OUTPUTS), (
        f"altitude={height} m, minimum clearance/200 m cell widths="
        f"{(height - 260) / 200:.1f}; normalized maximum errors={errors}; limits={limits}"
    )


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("index,height", list(enumerate((700, 1400, 3000, 8000))),
                         ids=lambda value: str(value))
def test_benchmark_reference(path, index, height):
    data = altitude_case(index)
    compare(run(path, data), data, height)


@pytest.mark.parametrize("index,height", list(enumerate((700, 1400, 3000, 8000))),
                         ids=lambda value: str(value))
def test_benchmark_mlx_metal_crosscheck(index, height):
    data = altitude_case(index)
    compare(run("metal", data), run("mlx", data), height)
