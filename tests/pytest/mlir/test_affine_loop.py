from typing import cast

import pytest
from xdsl.dialects import affine, func

from mlir_utils import requires_mlir
from xtc.backends.mlir.MlirCompiler import MlirCompiler
from xtc.backends.mlir.MlirCompilerPasses import MlirProgramInsertTransformPass
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend
from xtc.backends.mlir.MlirNodeBackend import MlirNodeBackend
from xtc.backends.mlir.MlirNodeScheduler import MlirNodeScheduler
from xtc.backends.mlir.MlirScheduler import MlirSchedule
from xtc.cli.mlir_loop import build_node_scheduler, parse_mlir_loop_module


AFFINE_NEST = """
func.func @kernel(%A: memref<16x16x16xf32>) {
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      affine.for %k = 0 to 16 {
        %v = affine.load %A[%i, %j, %k] : memref<16x16x16xf32>
        affine.store %v, %A[%i, %j, %k] : memref<16x16x16xf32>
      }
    }
  } {
    loop.dims = ["i", "j", "k"],
    loop.schedule = {
      "i" = {"parallelize"},
      "j",
      "k",
      "j#4" = {"unroll"},
      "k#2" = {"vectorize"}
    }
  }
  return
}
"""


def generate_transform(source: str) -> str:
    module = parse_mlir_loop_module(source)
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    root = next(op for op in function.walk() if isinstance(op, affine.ForOp))
    node_scheduler = build_node_scheduler(root, "__node0__", False, [], True)
    backend = MlirGraphBackend(
        xdsl_func=function,
        nodes=[cast(MlirNodeBackend, node_scheduler.backend)],
    )
    graph_scheduler = backend.get_scheduler(nodes_schedulers=[node_scheduler])
    compiler = cast(MlirCompiler, backend.get_compiler())
    program = compiler.generate_program()
    MlirProgramInsertTransformPass(
        program,
        compiler.target,
        cast(MlirSchedule, graph_scheduler.schedule()),
        always_vectorize=False,
    ).run()
    return str(program.mlir_module)


@requires_mlir
def test_affine_schedule_generates_supported_transforms():
    transform = generate_transform(AFFINE_NEST)
    assert transform.count("transform.affine.tile") == 1
    assert "transform.structured.tile_using_for" not in transform
    assert "dimensions [0, 1, 2, 1]" in transform
    assert "tile_sizes [1, 4, 2, 1]" in transform
    assert "point_dimensions [0, 1, 2]" in transform
    assert transform.count("transform.split_handle") == 2
    assert "transform.affine.parallelize" in transform
    assert "transform.loop.unroll" in transform
    assert "transform.affine.vectorize" in transform
    assert "vector_sizes [2]" in transform
    assert transform.count("transform.structured.vectorize") == 1
    assert "transform.include" not in transform
    assert '!transform.op<"affine.parallel">' in transform
    assert transform.index("transform.loop.unroll") < transform.index(
        "transform.affine.parallelize"
    )
    assert transform.index("transform.affine.parallelize") < transform.index(
        "transform.affine.vectorize"
    )


def test_affine_tile_preserves_ordered_repeated_dimensions():
    scheduler = MlirNodeScheduler("kernel", "__node0__", ["i", "j", "k"])
    scheduler.tile("i", {"i0": 16, "i1": 8})
    scheduler.tile("j", {"j0": 4})
    scheduler.interchange(["j", "i", "k", "i0", "j0", "i1"])

    _, tile_loops, dimensions, tile_sizes, point_dimensions = (
        MlirProgramInsertTransformPass._affine_tile_spec(
            scheduler.mlir_node_schedule(), "."
        )
    )

    assert tile_loops == ["./j", "./i", "./k", "./i0", "./j0", "./i1"]
    assert dimensions == [1, 0, 2, 0, 1, 0]
    assert tile_sizes == [4, 16, 1, 8, 1, 1]
    assert point_dimensions == [0, 1, 2]


@requires_mlir
def test_affine_matmul_preserves_linalg_schedule_order():
    source = """
func.func @myfun(
    %A: memref<256x512xf32>,
    %B: memref<512x256xf32>,
    %C: memref<256x256xf32>) {
  affine.for %I = 0 to 256 {
    affine.for %J = 0 to 256 {
      affine.for %K = 0 to 512 {
        %a = affine.load %A[%I, %K] : memref<256x512xf32>
        %b = affine.load %B[%K, %J] : memref<512x256xf32>
        %c = affine.load %C[%I, %J] : memref<256x256xf32>
        %product = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %product : f32
        affine.store %sum, %C[%I, %J] : memref<256x256xf32>
      }
    }
  } {
    loop.dims = ["I", "J", "K"],
    loop.schedule = {
      "I",
      "J",
      "K",
      "I#1" = {"unroll"},
      "K#8" = {"unroll"},
      "J#64" = {"vectorize"}
    }
  }
  return
}
"""
    transform = generate_transform(source)
    assert "dimensions [0, 1, 2, 0, 2]" in transform
    assert "tile_sizes [1, 64, 8, 1, 1]" in transform
    assert "point_dimensions [0, 2, 1]" in transform
    assert "transform.loop.unroll" in transform
    assert "transform.affine.vectorize" in transform
    assert "vector_sizes [64]" in transform
    assert "point_band" in transform


@requires_mlir
def test_affine_schedule_must_be_on_outermost_loop():
    source = """
func.func @kernel(%A: memref<16x16xf32>) {
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %v = affine.load %A[%i, %j] : memref<16x16xf32>
      affine.store %v, %A[%i, %j] : memref<16x16xf32>
    } {loop.dims = ["j"], loop.schedule = {"j"}}
  }
  return
}
"""
    module = parse_mlir_loop_module(source)
    loops = [op for op in module.walk() if isinstance(op, affine.ForOp)]

    with pytest.raises(Exception, match="outermost affine.for"):
        build_node_scheduler(loops[1], "__node0__", False, [], True)
