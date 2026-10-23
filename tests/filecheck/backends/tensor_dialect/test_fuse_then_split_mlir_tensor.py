# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

# Small conv2d
N, H, W, F, R, S, C, SH, SW, dtype = 1, 8, 8, 16, 5, 5, 3, 2, 2, "float32"
a = O.tensor((N, H, W, C), dtype, name="I")
b = O.tensor((R, S, C, F), dtype, name="W")

with O.graph(name="pad_conv2d_nhwc_mini") as gb:
    p = O.pad2d(a, padding=2, axes=(1, 2), name="pad")
    c = O.conv2d(p, b, stride=(SH, SW), name="conv")
    O.relu(c)

graph = gb.graph
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler(default_node="conv")
# fuse above split
sch.split("r", {"r0": 0, "r1": 2})
sch.interchange(["b", "h", "w", "r0","r1"])
sch.interchange(["s", "c", "f"],"./r0")
sch.interchange(["s", "c", "f"],"./r1")
sch.fuse_producer_at("w", 0,".")
sch.fuse_consumer_at("w",".")
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="fuse_then_split_mlir_tensor",
    # print_source_ir=True,
    # print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump After Tensor Lowering //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0 * 2)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map2 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
# CHECK-NEXT:  #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
# CHECK-NEXT:  #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
# CHECK-NEXT:  #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
# CHECK-NEXT:  #map6 = affine_map<(d0, d1, d2, d3) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @pad_conv2d_nhwc_mini(%arg0: memref<1x8x8x3xf32> {llvm.noalias}, %arg1: memref<5x5x3x16xf32> {llvm.noalias}, %arg2: memref<1x4x4x16xf32> {llvm.noalias}) {
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c5 = arith.constant 5 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c3 = arith.constant 3 : index
# CHECK-NEXT:      %c8 = arith.constant 8 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %alloc = memref.alloc() {alignment = 256 : i64} : memref<1x12x12x3xf32>
# CHECK-NEXT:      linalg.fill {__xtc_id_conv_0_} ins(%cst : f32) outs(%arg2 : memref<1x4x4x16xf32>)
# CHECK-NEXT:      %subview = memref.subview %alloc[0, 0, 0, 0] [1, 11, 11, 3] [1, 1, 1, 1] : memref<1x12x12x3xf32> to memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>>
# CHECK-NEXT:      %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (memref<1x4x4x16xf32>) {
# CHECK-NEXT:        %1 = affine.apply #map(%arg3)
# CHECK-NEXT:        %subview_0 = memref.subview %subview[0, %1, 0, 0] [1, 5, 11, 3] [1, 1, 1, 1] : memref<1x11x11x3xf32, strided<[432, 36, 3, 1]>> to memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        %2 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %subview_1) -> (memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:          %3 = affine.apply #map(%arg5)
# CHECK-NEXT:          %subview_3 = memref.subview %subview_0[0, 0, %3, 0] [1, 5, 5, 3] [1, 1, 1, 1] : memref<1x5x11x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%subview_3 : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>) attrs =  {__xtc_id_pad_} {
# CHECK-NEXT:          ^bb0(%out: f32):
# CHECK-NEXT:            %6 = linalg.index 1 : index
# CHECK-NEXT:            %7 = affine.apply #map2(%arg3)[%6]
# CHECK-NEXT:            %8 = linalg.index 2 : index
# CHECK-NEXT:            %9 = affine.apply #map2(%arg5)[%8]
# CHECK-NEXT:            %10 = linalg.index 3 : index
# CHECK-NEXT:            %11 = arith.subi %7, %c2 : index
# CHECK-NEXT:            %12 = arith.cmpi sge, %11, %c0 : index
# CHECK-NEXT:            %13 = arith.cmpi slt, %11, %c8 : index
# CHECK-NEXT:            %14 = arith.subi %9, %c2 : index
# CHECK-NEXT:            %15 = arith.cmpi sge, %14, %c0 : index
# CHECK-NEXT:            %16 = arith.cmpi slt, %14, %c8 : index
# CHECK-NEXT:            %17 = arith.cmpi sge, %10, %c0 : index
# CHECK-NEXT:            %18 = arith.cmpi slt, %10, %c3 : index
# CHECK-NEXT:            %19 = arith.andi %12, %13 : i1
# CHECK-NEXT:            %20 = arith.andi %19, %15 : i1
# CHECK-NEXT:            %21 = arith.andi %20, %16 : i1
# CHECK-NEXT:            %22 = arith.andi %21, %17 : i1
# CHECK-NEXT:            %23 = arith.andi %22, %18 : i1
# CHECK-NEXT:            %24 = scf.if %23 -> (f32) {
# CHECK-NEXT:              %25 = memref.load %arg0[%c0, %11, %14, %10] : memref<1x8x8x3xf32>
# CHECK-NEXT:              scf.yield %25 : f32
# CHECK-NEXT:            } else {
# CHECK-NEXT:              scf.yield %cst : f32
# CHECK-NEXT:            }
# CHECK-NEXT:            linalg.yield %24 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_4 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          %subview_5 = memref.subview %subview_3[0, 0, 0, 0] [1, 2, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x2x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_6 = memref.subview %arg1[0, 0, 0, 0] [2, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<2x5x3x16xf32, strided<[240, 48, 16, 1]>>
# CHECK-NEXT:          %4 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %subview_4) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_10 = memref.subview %subview_5[0, 0, %arg7, 0] [1, 2, 1, 3] [1, 1, 1, 1] : memref<1x2x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x2x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_6[0, %arg7, 0, 0] [2, 1, 3, 16] [1, 1, 1, 1] : memref<2x5x3x16xf32, strided<[240, 48, 16, 1]>> to memref<2x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %6 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_12 = memref.subview %subview_10[0, 0, 0, %arg9] [1, 2, 1, 1] [1, 1, 1, 1] : memref<1x2x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x2x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_13 = memref.subview %subview_11[0, 0, %arg9, 0] [2, 1, 1, 16] [1, 1, 1, 1] : memref<2x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<2x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %7 = scf.for %arg11 = %c0 to %c16 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_14 = memref.subview %subview_13[0, 0, 0, %arg11] [2, 1, 1, 1] [1, 1, 1, 1] : memref<2x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<2x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_12, %subview_14 : memref<1x2x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<2x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_15 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                ^bb0(%in: f32, %in_17: f32, %out: f32):
# CHECK-NEXT:                  %8 = arith.mulf %in, %in_17 fastmath<fast> : f32
# CHECK-NEXT:                  %9 = arith.addf %out, %8 fastmath<fast> : f32
# CHECK-NEXT:                  linalg.yield %9 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:                %subview_16 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_15, %subview_16 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg12 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"./r0/f"}
# CHECK-NEXT:              scf.yield %7 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"./r0/c"}
# CHECK-NEXT:            scf.yield %6 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./r0/s"}
# CHECK-NEXT:          %subview_7 = memref.subview %subview_3[0, 2, 0, 0] [1, 3, 5, 3] [1, 1, 1, 1] : memref<1x5x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x3x5x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:          %subview_8 = memref.subview %arg1[2, 0, 0, 0] [3, 5, 3, 16] [1, 1, 1, 1] : memref<5x5x3x16xf32> to memref<3x5x3x16xf32, strided<[240, 48, 16, 1], offset: 480>>
# CHECK-NEXT:          %5 = scf.for %arg7 = %c0 to %c5 step %c1 iter_args(%arg8 = %4) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:            %subview_10 = memref.subview %subview_7[0, 0, %arg7, 0] [1, 3, 1, 3] [1, 1, 1, 1] : memref<1x3x5x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x3x1x3xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_8[0, %arg7, 0, 0] [3, 1, 3, 16] [1, 1, 1, 1] : memref<3x5x3x16xf32, strided<[240, 48, 16, 1], offset: 480>> to memref<3x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:            %6 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:              %subview_12 = memref.subview %subview_10[0, 0, 0, %arg9] [1, 3, 1, 1] [1, 1, 1, 1] : memref<1x3x1x3xf32, strided<[432, 36, 3, 1], offset: ?>> to memref<1x3x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>
# CHECK-NEXT:              %subview_13 = memref.subview %subview_11[0, 0, %arg9, 0] [3, 1, 1, 16] [1, 1, 1, 1] : memref<3x1x3x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<3x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:              %7 = scf.for %arg11 = %c0 to %c16 step %c1 iter_args(%arg12 = %arg10) -> (memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) {
# CHECK-NEXT:                %subview_14 = memref.subview %subview_13[0, 0, 0, %arg11] [3, 1, 1, 1] [1, 1, 1, 1] : memref<3x1x1x16xf32, strided<[240, 48, 16, 1], offset: ?>> to memref<3x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>
# CHECK-NEXT:                %subview_15 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%subview_12, %subview_14 : memref<1x3x1x1xf32, strided<[432, 36, 3, 1], offset: ?>>, memref<3x1x1x1xf32, strided<[240, 48, 16, 1], offset: ?>>) outs(%subview_15 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {__xtc_id_conv_} {
# CHECK-NEXT:                ^bb0(%in: f32, %in_17: f32, %out: f32):
# CHECK-NEXT:                  %8 = arith.mulf %in, %in_17 fastmath<fast> : f32
# CHECK-NEXT:                  %9 = arith.addf %out, %8 fastmath<fast> : f32
# CHECK-NEXT:                  linalg.yield %9 : f32
# CHECK-NEXT:                }
# CHECK-NEXT:                %subview_16 = memref.subview %arg12[0, 0, 0, %arg11] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                memref.copy %subview_15, %subview_16 : memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:                scf.yield %arg12 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:              } {"./r1/f"}
# CHECK-NEXT:              scf.yield %7 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:            } {"./r1/c"}
# CHECK-NEXT:            scf.yield %6 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          } {"./r1/s"}
# CHECK-NEXT:          linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>, f32) outs(%5 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>) attrs =  {"__xtc_id_%4_"} {
# CHECK-NEXT:          ^bb0(%in: f32, %in_10: f32, %out: f32):
# CHECK-NEXT:            %6 = arith.maximumf %in, %in_10 : f32
# CHECK-NEXT:            linalg.yield %6 : f32
# CHECK-NEXT:          }
# CHECK-NEXT:          %subview_9 = memref.subview %arg6[0, 0, %arg5, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          memref.copy %5, %subview_9 : memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x1x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:          scf.yield %arg6 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        } {"./w"}
# CHECK-NEXT:        %subview_2 = memref.subview %arg4[0, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<1x4x4x16xf32> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        memref.copy %2, %subview_2 : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>>
# CHECK-NEXT:        scf.yield %arg4 : memref<1x4x4x16xf32>
# CHECK-NEXT:      } {"./h"}
# CHECK-NEXT:      memref.copy %0, %arg2 : memref<1x4x4x16xf32> to memref<1x4x4x16xf32>
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: pad_conv2d_nhwc_mini
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 1x8x8x3xfloat32
# CHECK-NEXT:    - %1 : 5x5x3x16xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %4 : 1x4x4x16xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: pad2d(%0, padding={1: (2, 2), 2: (2, 2)}, constant_value=0) {name = 'pad'} : [1x8x8x3xfloat32] -> [1x12x12x3xfloat32]
# CHECK-NEXT:    - %3: conv2d(%2, %1, stride=(2, 2)) {name = 'conv'} : [1x12x12x3xfloat32, 5x5x3x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:    - %4: relu(%3) : [1x4x4x16xfloat32] -> [1x4x4x16xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
