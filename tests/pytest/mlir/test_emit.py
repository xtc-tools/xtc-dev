from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.backends.mlir.MlirEmit import LEAKY_RELU_ALPHA

from mlir_utils import requires_mlir

SEED = 0
DTYPE = "float32"

# Matmul sizes
I, J, K = 4, 8, 16

# Conv2d sizes 
BATCH, OUT_H, OUT_W, OUT_C = 1, 3, 4, 8
FILTER_H, FILTER_W, IN_C = 3, 3, 3
STRIDE = (2, 3)
IN_H = OUT_H * STRIDE[0] + FILTER_H - 1
IN_W = OUT_W * STRIDE[1] + FILTER_W - 1


def conv2d_nhwc_hwcf(
    input_nhwc: npt.NDArray[np.float32],
    weight_hwcf: npt.NDArray[np.float32],
    *,
    stride_height: int,
    stride_width: int,
) -> npt.NDArray[np.float32]:
    batch_size, _, _, input_channels = input_nhwc.shape
    filter_height, filter_width, weight_in_c, output_channels = weight_hwcf.shape
    assert input_channels == weight_in_c
    output_height = (input_nhwc.shape[1] - filter_height) // stride_height + 1
    output_width = (input_nhwc.shape[2] - filter_width) // stride_width + 1
    output = np.zeros(
        (batch_size, output_height, output_width, output_channels),
        dtype=np.float32,
    )
    for batch in range(batch_size):
        for out_h in range(output_height):
            for out_w in range(output_width):
                for out_c in range(output_channels):
                    acc = np.float32(0.0)
                    for filter_r in range(filter_height):
                        for filter_s in range(filter_width):
                            for in_c in range(input_channels):
                                in_h = out_h * stride_height + filter_r
                                in_w = out_w * stride_width + filter_s
                                acc += (
                                    input_nhwc[batch, in_h, in_w, in_c]
                                    * weight_hwcf[filter_r, filter_s, in_c, out_c]
                                )
                    output[batch, out_h, out_w, out_c] = acc
    return output


EMIT_CASES: list[tuple[str, str, bool, bool, bool]] = []
for _dialect in ("linalg", "affine"):
    for _epilogue in (False, True):
        for _transpose_a in (False, True):
            for _transpose_b in (False, True):          
                EMIT_CASES.append(
                    ("matmul", _dialect, _transpose_a, _transpose_b, _epilogue)
                )
        EMIT_CASES.append(("conv2d", _dialect, False, False, _epilogue))


@requires_mlir
@pytest.mark.parametrize(
    "op,dialect,transpose_a,transpose_b,epilogue",
    EMIT_CASES,
)
def test_emit_matches_python(
    tmp_path: Path,
    op: str,
    dialect: str,
    transpose_a: bool,
    transpose_b: bool,
    epilogue: bool,
) -> None:
    rng = np.random.default_rng(SEED)
    kwargs: dict[str, object] = {}
    if dialect == "affine":
        kwargs["emit_affine"] = True
    else:
        kwargs["emit_from_desc"] = True
    if epilogue:
        kwargs["epilogue"] = "relu" if op == "matmul" else "leaky_relu"

    if op == "matmul":
        a_logical = rng.standard_normal((I, K), dtype=np.float32)
        b_logical = rng.standard_normal((K, J), dtype=np.float32)
        reference = a_logical @ b_logical
        if epilogue:
            reference = np.maximum(reference, 0.0)
        # Physical buffers match MLIR types when layout=[1, 0].
        a_physical = a_logical.T.copy() if transpose_a else a_logical
        b_physical = b_logical.T.copy() if transpose_b else b_logical
        a = O.tensor(
            (I, K),
            DTYPE,
            name="A",
            layout=[1, 0] if transpose_a else None,
        )
        b = O.tensor(
            (K, J),
            DTYPE,
            name="B",
            layout=[1, 0] if transpose_b else None,
        )
        with O.graph(name="emit_matmul") as gb:
            O.matmul(a, b, name="C", **kwargs)
        inputs = [a_physical, b_physical]
        output = np.empty((I, J), dtype=np.float32)
    else:
        input_nhwc = rng.standard_normal((BATCH, IN_H, IN_W, IN_C), dtype=np.float32)
        weight_hwcf = rng.standard_normal(
            (FILTER_H, FILTER_W, IN_C, OUT_C), dtype=np.float32
        )
        reference = conv2d_nhwc_hwcf(
            input_nhwc,
            weight_hwcf,
            stride_height=STRIDE[0],
            stride_width=STRIDE[1],
        )
        if epilogue:
            reference = np.maximum(reference, LEAKY_RELU_ALPHA * reference)
        inp = O.tensor((BATCH, IN_H, IN_W, IN_C), DTYPE, name="In")
        weight = O.tensor((FILTER_H, FILTER_W, IN_C, OUT_C), DTYPE, name="Wt")
        with O.graph(name="emit_conv2d") as gb:
            O.conv2d(inp, weight, stride=STRIDE, name="Out", **kwargs)
        inputs = [input_nhwc, weight_hwcf]
        output = np.empty((BATCH, OUT_H, OUT_W, OUT_C), dtype=np.float32)

    backend = Backend(gb.graph)
    label = (
        f"{op}_{dialect}"
        f"_ta{int(transpose_a)}_tb{int(transpose_b)}"
        f"_ep{int(epilogue)}"
    )
    code = (
        backend.get_compiler(
            shared_lib=True,
            dump_file=str(tmp_path / label),
        )
        .compile(backend.get_scheduler(nodes=[]).schedule())
        .get_executor(
            validate=False,
            parameters=(inputs, [output]),
        )
        .execute()
    )
    assert code == 0
    assert np.allclose(output, reference, rtol=1e-5, atol=1e-5)
