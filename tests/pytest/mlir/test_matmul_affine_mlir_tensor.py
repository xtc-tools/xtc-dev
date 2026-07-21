#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

from pathlib import Path

import numpy as np
import numpy.typing as npt

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

from mlir_utils import requires_mlir

SEED = 0
I, J, K, DTYPE = 4, 8, 16, "float32"


def _run_matmul(
    *,
    emit_affine: bool,
    A: npt.NDArray[np.float32],
    B: npt.NDArray[np.float32],
    tmp_path: Path,
) -> tuple[npt.NDArray[np.float32], int]:
    a = O.tensor((I, K), DTYPE, name="A")
    b = O.tensor((K, J), DTYPE, name="B")
    with O.graph(name="matmul") as gb:
        kwargs = {"emit_affine": True} if emit_affine else {}
        O.matmul(a, b, name="C", **kwargs)

    impl = Backend(gb.graph)
    label = "affine" if emit_affine else "linalg"
    C = np.empty((I, J), dtype=np.float32)
    module = impl.get_compiler(
        shared_lib=True,
        dump_file=str(tmp_path / f"matmul_{label}"),
    ).compile(impl.get_scheduler().schedule())
    code = module.get_executor(
        validate=True,
        parameters=([A, B], [C]),
    ).execute()
    return C.copy(), code


@requires_mlir
def test_matmul_affine_matches_linalg(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((I, K), dtype=np.float32)
    B = rng.standard_normal((K, J), dtype=np.float32)

    C_lin, code_lin = _run_matmul(emit_affine=False, A=A, B=B, tmp_path=tmp_path)
    C_aff, code_aff = _run_matmul(emit_affine=True, A=A, B=B, tmp_path=tmp_path)

    assert code_lin == 0 and code_aff == 0
    assert np.allclose(C_lin, C_aff)
