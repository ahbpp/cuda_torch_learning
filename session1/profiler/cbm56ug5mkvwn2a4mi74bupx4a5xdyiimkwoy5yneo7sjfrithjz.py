
# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_aleksei/wh/cwhborkn725w5e4wku5eiwrisqaybl7b5s4t2xb45qisczvx2sal.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_aleksei/nd/cndcn5sepdp5kdpdplsbyqolyhlcr23tkv7vta4loygycp6tghvb.py
# Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
# out_53 => convert_element_type_76
triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*(((r2 + (121*x1)) // 49) % 32))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = 0.02040816326530612
        tmp6 = tmp4 * tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.load(in_ptr2 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp9 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/xv/cxvgdw4yfgo2rdd5pvnf4ow5mi7lpvgctz2klw47dsq35un2bxlh.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/ui/cui6fiksgvuyyu2dkxdv6fxhxbit7bil5o3z5zskwv3afyvgpk4l.py
# Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
# out_53 => convert_element_type_76
triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/xy/cxykjnjdey7iefkrkhb64yxxajkzwvbndbtdk4q4m5wuxge3t3ww.py
# Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
# out_53 => convert_element_type_76
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp16', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = 0.02040816326530612
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.0006377551020408163
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp14 * tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/ha/chavebve5noaq67yo7ayyvyy62wswincxc32l5kb6re6k47gheiv.py
# Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# out_50 => convert_element_type_72
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp14 = tl.load(in_ptr2 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/u2/cu23l6mh64ocdb4zjfa6klnvqa4gl5yt4pbnhvjzenwje3krsja3.py
# Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_50 => convert_element_type_72
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp13 * tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/xr/cxrat3ttrtthiqvizbeggtvdnfbtf7w4tukv3yrjo77qqwzkboyg.py
# Source Nodes: [identity_2, out_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
# identity_2 => convert_element_type_68
# out_46 => convert_element_type_64
triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp7 = tl.load(in_ptr2 + (x0 + (512*(((r2 + (121*x1)) // 49) % 32))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = 0.02040816326530612
        tmp9 = tmp7 * tmp8
        tmp10 = tl.where(tmp6, tmp4, tmp9)
        tmp11 = tl.load(in_ptr3 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp20 = tl.load(in_ptr4 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp14 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tmp30 = tl.load(in_ptr6 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.load(in_ptr7 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp31 - tmp32
        tmp34 = tmp14 * tmp33
        tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
        tmp36 = tl.where(tmp2, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask & xmask, tmp39, _tmp38)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, xmask)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/bf/cbfgcdcged4j7uxqdbi2d7xdf7g3ttyum7qy5wgqlbzlnkbk5vnv.py
# Source Nodes: [identity_2, out_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
# identity_2 => convert_element_type_68
# out_46 => convert_element_type_64
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x0 + (512*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x3), None).to(tl.float32)
    tmp29 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 0.02040816326530612
    tmp6 = tmp4 * tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.0006377551020408163
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp11 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 * tmp17
    tmp34 = tmp33 * tmp33
    tmp35 = tmp32 * tmp34
    tmp36 = tmp30 * tmp35
    tmp37 = tmp11 - tmp36
    tmp38 = tmp37 - tmp25
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp33 * tmp40
    tmp42 = tmp38 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp19 * tmp45
    tmp47 = tmp26 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tl.store(out_ptr2 + (x3), tmp43, None)
    tl.store(out_ptr3 + (x3), tmp48, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/u7/cu72c2wvana46jz5ibopodizby43tyc3pkgsntanl4pckpa6wc35.py
# Source Nodes: [out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
# out_39 => convert_element_type_56
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/no/cnogredwvue6ek42mgejtb3roz2ptjgvz5xjmxgrzrbrwcbvrj5z.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/fb/cfboymyxk5yhlhfg3dnmhowtesj3kldoftmxqlrilix4yg6ukjw2.py
# Source Nodes: [out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
# out_39 => convert_element_type_56
triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/4l/c4lkh5epwge6rrml4r2te4dbbgiltfbyulkrswxtbjnozfrgvmwu.py
# Source Nodes: [out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_39 => convert_element_type_56
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp15 * tmp24
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/gv/cgvhaqvh2ebsb3rkl2d44irmqjeikz4y634w64rqakcqj4ky7ceb.py
# Source Nodes: [out_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# out_36 => convert_element_type_52
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/k2/ck2o4b2n5yraa43sxssba6unr3mc3etcubi4yqqh2haoh7xun4af.py
# Source Nodes: [out_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_36 => convert_element_type_52
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp13 * tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/hn/chnilwhgwaywzjdcwxaag3jxpywrsmgbsfl7dubmxa3aheo2mlwg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/l3/cl3h65nhf642oioessjix5g4k3aodgjbge32dnaqhd7wlh653k3b.py
# Source Nodes: [identity_1, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# identity_1 => convert_element_type_48
# out_32 => convert_element_type_44
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp0 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/wo/cwoy4duvmzrg3ka5emblzpwbj6dg46u3pjmrnnchzctnnurj3st6.py
# Source Nodes: [identity_1, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
# identity_1 => convert_element_type_48
# out_32 => convert_element_type_44
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 0.00015943877551020407
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp8 * tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 * tmp6
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp0 - tmp30
    tmp32 = tmp31 - tmp14
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp27 * tmp34
    tmp36 = tmp32 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/nv/cnvtg3vwe4ob4qdmowl6s55dbnoa6ebhpgyy474wrmwq4xaibp7f.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
# out_25 => convert_element_type_36
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/pb/cpb3m3xqd5sjmug3qfk73hxfuqx34gn6esablzc3mv35pf4hbmpy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/xo/cxon2gw34xcavxr4ltqkfkotwqeoafcn6vt4gmtutzeuz6o2voiv.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
# out_25 => convert_element_type_36
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/wx/cwx3qrvw2y2c6wjei7xzbmgj4rdx44iuunfyr4hfdcjxpxpkspan.py
# Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_25 => convert_element_type_36
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 3.985969387755102e-05
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp15 * tmp24
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/nw/cnwywknxiolygzwkwssfn7a3siqo6zeghiwfiveste3x52iqsfn6.py
# Source Nodes: [out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# out_22 => convert_element_type_32
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/nr/cnrnue5n2b2jshk4x4chaactbvi34wsdexxnfils6qtmyosaijyc.py
# Source Nodes: [out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_22 => convert_element_type_32
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp13 * tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/ev/cevz4wmhwpyr745gwfb77id42ch6fuqpxybt63wdbbpnfuihypzz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/uv/cuvyobk2u6nt3ezb2idpe67zyiwnczyigodql5akmdi3uupbluef.py
# Source Nodes: [identity, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# identity => convert_element_type_28
# out_18 => convert_element_type_24
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp0 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/d6/cd65hvfmw7i63yqhxwrpwo2cfp73inufmiaba2semsisw6cuzywn.py
# Source Nodes: [identity, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
# identity => convert_element_type_28
# out_18 => convert_element_type_24
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 3.985969387755102e-05
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp8 * tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 * tmp6
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp0 - tmp30
    tmp32 = tmp31 - tmp14
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp27 * tmp34
    tmp36 = tmp32 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/ga/cgasiubmflj3vhf343jzhuyyoys65yaqrzsvamvsct6smo4raj5f.py
# Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
# out_11 => convert_element_type_16
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/dj/cdjx3usvfaoeknjpk46m5aou37jnzh4tb52uhvge4tuzuqs44pk4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/hd/chdkbyj3aw37v75yivvqjo64isac2pc5jbgoitrv5txjykbgykyx.py
# Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
# out_11 => convert_element_type_16
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/y6/cy6bvcxhdofnzt3epvy4ugswdty65bcn7y3h4fezt6ph6vvubmzk.py
# Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_11 => convert_element_type_16
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 9.964923469387754e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp15 * tmp24
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/qu/cquvvmfa65akgvjbvklfejmrcv2uw2cuxwugkz5de6ovmjqw2z7a.py
# Source Nodes: [out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# out_8 => convert_element_type_12
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/gc/cgcth2foaakistbp45rqrtvykjnnupgbdyc5ijh74pzeuyf75rxp.py
# Source Nodes: [out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# out_8 => convert_element_type_12
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 9.964923469387754e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp13 * tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/p6/cp67aksij4rkfmhzhssq2qozo6ufdhcyqjkntmad36wfdwl3ixvl.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_33', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/gb/cgbhzlqnvk4z7byvyreeb2nozjq2d7msdqz6bf3pnbj47o3ri7hr.py
# Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# out_4 => convert_element_type_8
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + (64*r2) + (12544*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp1 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/64/c64bwuaflpgugwmnag55gdi57rqfegfvknohbt56ovzqt4wg5u5k.py
# Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
# out_4 => convert_element_type_8
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp9 * tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/nr/cnrbzkh64eip75yc7gcgeo2k5uy6g573mekqwjo3ckfbo7mq3354.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/45/c45irdd3ovmyb6mxpq57ve3gdwsurozjocckulsqvs65q42zwmgm.py
# Source Nodes: [x_3], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# x_3 => _low_memory_max_pool2d_offsets_to_indices
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i8', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 112
    x2 = (xindex // 7168) % 112
    x3 = (xindex // 802816)
    x6 = (xindex // 64) % 12544
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(tl.minimum(tl.maximum(0, (x1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(tl.maximum(0, (x2 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None)
    tmp12 = tl.load(in_ptr1 + (x0 + (64*(tl.minimum(tl.maximum(0, (x1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(tl.maximum(0, (x2 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (x0 + (64*(tl.minimum(1 + (tl.maximum(0, (x1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(tl.maximum(0, (x2 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None)
    tmp26 = tl.load(in_ptr1 + (x0 + (64*(tl.minimum(1 + (tl.maximum(0, (x1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(tl.maximum(0, (x2 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None).to(tl.float32)
    tmp38 = tl.load(in_ptr0 + (x0 + (64*(tl.minimum(tl.maximum(0, (x1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(1 + (tl.maximum(0, (x2 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None)
    tmp47 = tl.load(in_ptr1 + (x0 + (64*(tl.minimum(tl.maximum(0, (x1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(1 + (tl.maximum(0, (x2 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None).to(tl.float32)
    tmp57 = tl.load(in_ptr0 + (x0 + (64*(tl.minimum(1 + (tl.maximum(0, (x1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(1 + (tl.maximum(0, (x2 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None)
    tmp65 = tl.load(in_ptr1 + (x0 + (64*(tl.minimum(1 + (tl.maximum(0, (x1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2)))))) + (3584*(tl.minimum(1 + (tl.maximum(0, (x2 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2)))))) + (200704*x3)), None).to(tl.float32)
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = (-1) + (2*(tl.minimum(tl.maximum(0, (x2 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2))))))
    tmp6 = tmp5 + tmp2
    tmp7 = (-1) + (2*(tl.minimum(tl.maximum(0, (x1 // 2)), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2))))))
    tmp8 = tmp7 + tmp4
    tmp9 = tl.full([1], 112, tl.int64)
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp13 = x6
    tmp14 = tmp11 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tmp18 = tl.where((tmp17 < 0) != (tmp1 < 0), tl.where(tmp17 % tmp1 != 0, tmp17 // tmp1 - 1, tmp17 // tmp1), tmp17 // tmp1)
    tmp19 = tmp18 * tmp1
    tmp20 = tmp17 - tmp19
    tmp21 = tmp5 + tmp18
    tmp22 = (-1) + (2*(tl.minimum(1 + (tl.maximum(0, (x1 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x1) // 2))))))
    tmp23 = tmp22 + tmp20
    tmp24 = tmp21 * tmp9
    tmp25 = tmp24 + tmp23
    tmp27 = tmp25 == tmp13
    tmp28 = tl.maximum(0, (x2 // 2))
    tmp29 = tl.minimum(56, 1 + ((1 + x2) // 2))
    tmp30 = tmp28 < tmp29
    tmp31 = 1 + (tl.maximum(0, (x1 // 2)))
    tmp32 = tl.minimum(56, 1 + ((1 + x1) // 2))
    tmp33 = tmp31 < tmp32
    tmp34 = tmp30 & tmp33
    tmp35 = tmp34 & tmp27
    tmp36 = tmp16 + tmp26
    tmp37 = tl.where(tmp35, tmp36, tmp16)
    tmp39 = tl.where((tmp38 < 0) != (tmp1 < 0), tl.where(tmp38 % tmp1 != 0, tmp38 // tmp1 - 1, tmp38 // tmp1), tmp38 // tmp1)
    tmp40 = tmp39 * tmp1
    tmp41 = tmp38 - tmp40
    tmp42 = (-1) + (2*(tl.minimum(1 + (tl.maximum(0, (x2 // 2))), (-1) + (tl.minimum(56, 1 + ((1 + x2) // 2))))))
    tmp43 = tmp42 + tmp39
    tmp44 = tmp7 + tmp41
    tmp45 = tmp43 * tmp9
    tmp46 = tmp45 + tmp44
    tmp48 = tmp46 == tmp13
    tmp49 = 1 + (tl.maximum(0, (x2 // 2)))
    tmp50 = tmp49 < tmp29
    tmp51 = tl.maximum(0, (x1 // 2))
    tmp52 = tmp51 < tmp32
    tmp53 = tmp50 & tmp52
    tmp54 = tmp53 & tmp48
    tmp55 = tmp37 + tmp47
    tmp56 = tl.where(tmp54, tmp55, tmp37)
    tmp58 = tl.where((tmp57 < 0) != (tmp1 < 0), tl.where(tmp57 % tmp1 != 0, tmp57 // tmp1 - 1, tmp57 // tmp1), tmp57 // tmp1)
    tmp59 = tmp58 * tmp1
    tmp60 = tmp57 - tmp59
    tmp61 = tmp42 + tmp58
    tmp62 = tmp22 + tmp60
    tmp63 = tmp61 * tmp9
    tmp64 = tmp63 + tmp62
    tmp66 = tmp64 == tmp13
    tmp67 = tmp50 & tmp33
    tmp68 = tmp67 & tmp66
    tmp69 = tmp56 + tmp65
    tmp70 = tl.where(tmp68, tmp69, tmp56)
    tl.store(out_ptr0 + (x7), tmp70, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/xo/cxoqkhir7tibayl6z5exafr6wlo6h5zci3oczuwmueokemugbeaj.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# x_1 => convert_element_type
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*(r2 % 112)) + (7168*((r2 + (784*x1)) // 112))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*(r2 % 112)) + (7168*((r2 + (784*x1)) // 112))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + (64*(r2 % 112)) + (7168*((r2 + (784*x1)) // 112))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_aleksei/yx/cyx6oco3z6h2djfaoovkb4wkr7ujpzxiame7e6h2njbirzm66uly.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# x_1 => convert_element_type
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=68), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '3472E74E07164CD5A94178B887164652E5C68546B4AF361AEB32FE97E562B33C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 2.4912308673469386e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp13 * tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp25, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_123, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, view, permute_1, le, unsqueeze_82, unsqueeze_94, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, unsqueeze_154, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_40, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_43, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_46, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_49, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_52, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_58, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_123, (32, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (32, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (32, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_2, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_3, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_1, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_2, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_3, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(relu_3, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_4, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_4, (32, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(relu_5, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_6, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_19, (128, ), (1, ))
    assert_size_stride(convolution_7, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(relu_6, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_8, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_25, (128, ), (1, ))
    assert_size_stride(relu_7, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_9, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_28, (128, ), (1, ))
    assert_size_stride(relu_8, (32, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_10, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(relu_9, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_11, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(convolution_12, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_37, (256, ), (1, ))
    assert_size_stride(relu_10, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_13, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_40, (256, ), (1, ))
    assert_size_stride(relu_11, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_14, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(relu_12, (32, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_15, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_46, (512, ), (1, ))
    assert_size_stride(relu_13, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_16, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_49, (512, ), (1, ))
    assert_size_stride(convolution_17, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_52, (512, ), (1, ))
    assert_size_stride(relu_14, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_18, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_55, (512, ), (1, ))
    assert_size_stride(relu_15, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_19, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_58, (512, ), (1, ))
    assert_size_stride(view, (32, 512), (512, 1))
    assert_size_stride(permute_1, (1000, 512), (512, 1))
    assert_size_stride(le, (32, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(unsqueeze_82, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_94, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_106, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_118, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_130, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_142, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_154, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_166, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_190, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_214, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (32, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 512), (512, 1), torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty_strided_cuda((1000, 512), (512, 1), torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 32), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty_strided_cuda((1, 1000), (1000, 1), torch.float16)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 32, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided_cuda((512, 13), (1, 512), torch.float32)
        buf5 = empty_strided_cuda((512, 13), (1, 512), torch.float32)
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_19, unsqueeze_82, buf3, buf5, 6656, 121, grid=grid(6656), stream=stream0)
        buf4 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf8 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, buf8, 512, 13, grid=grid(512), stream=stream0)
        buf6 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf7 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_58, buf6, buf7, 512, 13, grid=grid(512), stream=stream0)
        buf9 = empty_strided_cuda((32, 512, 7, 7), (25088, 1, 3584, 512), torch.float16)
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_19, unsqueeze_82, buf6, squeeze_58, buf4, primals_59, buf9, 802816, grid=grid(802816), stream=stream0)
        del convolution_19
        del primals_59
        del squeeze_58
        del unsqueeze_82
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf10 = aten.convolution_backward.default(buf9, relu_15, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_58
        buf11 = buf10[0]
        buf12 = buf10[1]
        del buf10
        buf13 = buf5; del buf5  # reuse
        buf15 = buf3; del buf3  # reuse
        # Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_5.run(relu_15, buf11, convolution_18, unsqueeze_94, buf13, buf15, 6656, 121, grid=grid(6656), stream=stream0)
        buf14 = buf6; del buf6  # reuse
        buf18 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf13, buf14, buf18, 512, 13, grid=grid(512), stream=stream0)
        buf16 = buf4; del buf4  # reuse
        buf17 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3.run(buf15, squeeze_55, buf16, buf17, 512, 13, grid=grid(512), stream=stream0)
        buf19 = buf11; del buf11  # reuse
        # Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf19, relu_15, convolution_18, unsqueeze_94, buf16, squeeze_55, buf14, primals_56, 802816, grid=grid(802816), stream=stream0)
        del convolution_18
        del primals_56
        del relu_15
        del squeeze_55
        del unsqueeze_94
        # Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf20 = aten.convolution_backward.default(buf19, relu_14, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_55
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = buf15; del buf15  # reuse
        buf25 = buf13; del buf13  # reuse
        buf34 = empty_strided_cuda((512, 13), (1, 512), torch.float32)
        # Source Nodes: [identity_2, out_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_7.run(relu_14, le, buf0, buf21, convolution_17, unsqueeze_106, convolution_16, unsqueeze_118, buf23, buf25, buf34, 6656, 121, grid=grid(6656), stream=stream0)
        buf24 = buf16; del buf16  # reuse
        buf29 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf23, buf24, buf29, 512, 13, grid=grid(512), stream=stream0)
        del buf23
        buf26 = buf14; del buf14  # reuse
        buf28 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [identity_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3.run(buf25, squeeze_52, buf26, buf28, 512, 13, grid=grid(512), stream=stream0)
        buf35 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf37 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [out_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3.run(buf34, squeeze_49, buf35, buf37, 512, 13, grid=grid(512), stream=stream0)
        buf38 = buf19; del buf19  # reuse
        buf30 = buf9; del buf9  # reuse
        # Source Nodes: [identity_2, out_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_8.run(relu_14, le, buf0, buf21, convolution_17, unsqueeze_106, buf26, squeeze_52, buf24, convolution_16, unsqueeze_118, buf35, squeeze_49, primals_50, primals_53, buf38, buf30, 802816, grid=grid(802816), stream=stream0)
        del buf0
        del buf21
        del buf24
        del convolution_16
        del convolution_17
        del le
        del primals_50
        del primals_53
        del relu_14
        del squeeze_49
        del squeeze_52
        del unsqueeze_106
        del unsqueeze_118
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf31 = aten.convolution_backward.default(buf30, relu_12, primals_52, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf30
        del primals_52
        buf32 = buf31[0]
        buf33 = buf31[1]
        del buf31
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf39 = aten.convolution_backward.default(buf38, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf38
        del primals_49
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = buf34; del buf34  # reuse
        buf44 = buf25; del buf25  # reuse
        # Source Nodes: [out_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_5.run(relu_13, buf40, convolution_15, unsqueeze_130, buf42, buf44, 6656, 121, grid=grid(6656), stream=stream0)
        buf43 = buf35; del buf35  # reuse
        buf47 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf42, buf43, buf47, 512, 13, grid=grid(512), stream=stream0)
        del buf42
        buf45 = buf26; del buf26  # reuse
        buf46 = empty_strided_cuda((512, ), (1, ), torch.float16)
        # Source Nodes: [out_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_3.run(buf44, squeeze_46, buf45, buf46, 512, 13, grid=grid(512), stream=stream0)
        del buf44
        buf48 = buf40; del buf40  # reuse
        # Source Nodes: [out_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf48, relu_13, convolution_15, unsqueeze_130, buf45, squeeze_46, buf43, primals_47, 802816, grid=grid(802816), stream=stream0)
        del buf43
        del buf45
        del convolution_15
        del primals_47
        del relu_13
        del squeeze_46
        del unsqueeze_130
        # Source Nodes: [out_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf49 = aten.convolution_backward.default(buf48, relu_12, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf48
        del primals_46
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = empty_strided_cuda((256, 49), (1, 256), torch.float32)
        buf54 = empty_strided_cuda((256, 49), (1, 256), torch.float32)
        # Source Nodes: [out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_9.run(relu_12, buf32, buf50, convolution_14, unsqueeze_142, buf52, buf54, 12544, 128, grid=grid(12544), stream=stream0)
        buf53 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf58 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_10.run(buf52, buf53, buf58, 256, 49, grid=grid(256), stream=stream0)
        buf55 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf57 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11.run(buf54, squeeze_43, buf55, buf57, 256, 49, grid=grid(256), stream=stream0)
        buf59 = empty_strided_cuda((32, 256, 14, 14), (50176, 1, 3584, 256), torch.float16)
        # Source Nodes: [out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_12, buf32, buf50, convolution_14, unsqueeze_142, buf55, squeeze_43, buf53, primals_44, buf59, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_14
        del primals_44
        del squeeze_43
        del unsqueeze_142
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf60 = aten.convolution_backward.default(buf59, relu_11, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf59
        del primals_43
        buf61 = buf60[0]
        buf62 = buf60[1]
        del buf60
        buf63 = buf54; del buf54  # reuse
        buf65 = buf52; del buf52  # reuse
        # Source Nodes: [out_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13.run(relu_11, buf61, convolution_13, unsqueeze_154, buf63, buf65, 12544, 128, grid=grid(12544), stream=stream0)
        buf64 = buf55; del buf55  # reuse
        buf68 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_10.run(buf63, buf64, buf68, 256, 49, grid=grid(256), stream=stream0)
        buf66 = buf53; del buf53  # reuse
        buf67 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [out_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11.run(buf65, squeeze_40, buf66, buf67, 256, 49, grid=grid(256), stream=stream0)
        buf69 = buf61; del buf61  # reuse
        # Source Nodes: [out_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf69, relu_11, convolution_13, unsqueeze_154, buf66, squeeze_40, buf64, primals_41, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_13
        del primals_41
        del relu_11
        del squeeze_40
        del unsqueeze_154
        # Source Nodes: [out_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf70 = aten.convolution_backward.default(buf69, relu_10, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf69
        del primals_40
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = empty_strided_cuda((32, 256, 14, 14), (50176, 1, 3584, 256), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_15.run(relu_10, relu_12, buf32, buf50, buf71, buf73, 1605632, grid=grid(1605632), stream=stream0)
        del buf32
        del relu_10
        del relu_12
        buf74 = buf65; del buf65  # reuse
        buf76 = buf63; del buf63  # reuse
        buf84 = empty_strided_cuda((256, 49), (1, 256), torch.float32)
        # Source Nodes: [identity_1, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_16.run(buf73, convolution_12, unsqueeze_166, convolution_11, unsqueeze_178, buf74, buf76, buf84, 12544, 128, grid=grid(12544), stream=stream0)
        buf75 = buf66; del buf66  # reuse
        buf79 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_10.run(buf74, buf75, buf79, 256, 49, grid=grid(256), stream=stream0)
        del buf74
        buf77 = buf64; del buf64  # reuse
        buf78 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11.run(buf76, squeeze_37, buf77, buf78, 256, 49, grid=grid(256), stream=stream0)
        buf85 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf86 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11.run(buf84, squeeze_34, buf85, buf86, 256, 49, grid=grid(256), stream=stream0)
        buf80 = buf71; del buf71  # reuse
        buf87 = buf50; del buf50  # reuse
        # Source Nodes: [identity_1, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_17.run(buf73, convolution_12, unsqueeze_166, buf77, squeeze_37, buf75, primals_38, convolution_11, unsqueeze_178, buf85, squeeze_34, primals_35, buf80, buf87, 1605632, grid=grid(1605632), stream=stream0)
        del buf73
        del buf75
        del convolution_11
        del convolution_12
        del primals_35
        del primals_38
        del squeeze_34
        del squeeze_37
        del unsqueeze_166
        del unsqueeze_178
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        buf81 = aten.convolution_backward.default(buf80, relu_8, primals_37, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf80
        del primals_37
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        # Source Nodes: [out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        buf88 = aten.convolution_backward.default(buf87, relu_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf87
        del primals_34
        buf89 = buf88[0]
        buf90 = buf88[1]
        del buf88
        buf91 = buf84; del buf84  # reuse
        buf93 = buf76; del buf76  # reuse
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13.run(relu_9, buf89, convolution_10, unsqueeze_190, buf91, buf93, 12544, 128, grid=grid(12544), stream=stream0)
        buf92 = buf85; del buf85  # reuse
        buf96 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_10.run(buf91, buf92, buf96, 256, 49, grid=grid(256), stream=stream0)
        del buf91
        buf94 = buf77; del buf77  # reuse
        buf95 = empty_strided_cuda((256, ), (1, ), torch.float16)
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_11.run(buf93, squeeze_31, buf94, buf95, 256, 49, grid=grid(256), stream=stream0)
        del buf93
        buf97 = buf89; del buf89  # reuse
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf97, relu_9, convolution_10, unsqueeze_190, buf94, squeeze_31, buf92, primals_32, 1605632, grid=grid(1605632), stream=stream0)
        del buf92
        del buf94
        del convolution_10
        del primals_32
        del relu_9
        del squeeze_31
        del unsqueeze_190
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf98 = aten.convolution_backward.default(buf97, relu_8, primals_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf97
        del primals_31
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf101 = empty_strided_cuda((128, 196), (1, 128), torch.float32)
        buf103 = empty_strided_cuda((128, 196), (1, 128), torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_18.run(relu_8, buf82, buf99, convolution_9, unsqueeze_202, buf101, buf103, 25088, 128, grid=grid(25088), stream=stream0)
        buf102 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf107 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf101, buf102, buf107, 128, 196, grid=grid(128), stream=stream0)
        buf104 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf106 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20.run(buf103, squeeze_28, buf104, buf106, 128, 196, grid=grid(128), stream=stream0)
        buf108 = empty_strided_cuda((32, 128, 28, 28), (100352, 1, 3584, 128), torch.float16)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_8, buf82, buf99, convolution_9, unsqueeze_202, buf104, squeeze_28, buf102, primals_29, buf108, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_9
        del primals_29
        del squeeze_28
        del unsqueeze_202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf109 = aten.convolution_backward.default(buf108, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf108
        del primals_28
        buf110 = buf109[0]
        buf111 = buf109[1]
        del buf109
        buf112 = buf103; del buf103  # reuse
        buf114 = buf101; del buf101  # reuse
        # Source Nodes: [out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_22.run(relu_7, buf110, convolution_8, unsqueeze_214, buf112, buf114, 25088, 128, grid=grid(25088), stream=stream0)
        buf113 = buf104; del buf104  # reuse
        buf117 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf112, buf113, buf117, 128, 196, grid=grid(128), stream=stream0)
        buf115 = buf102; del buf102  # reuse
        buf116 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20.run(buf114, squeeze_25, buf115, buf116, 128, 196, grid=grid(128), stream=stream0)
        buf118 = buf110; del buf110  # reuse
        # Source Nodes: [out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf118, relu_7, convolution_8, unsqueeze_214, buf115, squeeze_25, buf113, primals_26, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_8
        del primals_26
        del relu_7
        del squeeze_25
        del unsqueeze_214
        # Source Nodes: [out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf119 = aten.convolution_backward.default(buf118, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf118
        del primals_25
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = empty_strided_cuda((32, 128, 28, 28), (100352, 1, 3584, 128), torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24.run(relu_6, relu_8, buf82, buf99, buf120, buf122, 3211264, grid=grid(3211264), stream=stream0)
        del buf120
        del relu_6
        del relu_8
        buf123 = buf114; del buf114  # reuse
        buf125 = buf112; del buf112  # reuse
        buf133 = empty_strided_cuda((128, 196), (1, 128), torch.float32)
        # Source Nodes: [identity, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_25.run(buf122, convolution_7, unsqueeze_226, convolution_6, unsqueeze_238, buf123, buf125, buf133, 25088, 128, grid=grid(25088), stream=stream0)
        buf124 = buf115; del buf115  # reuse
        buf128 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf123, buf124, buf128, 128, 196, grid=grid(128), stream=stream0)
        del buf123
        buf126 = buf113; del buf113  # reuse
        buf127 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [identity], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20.run(buf125, squeeze_22, buf126, buf127, 128, 196, grid=grid(128), stream=stream0)
        buf134 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf135 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20.run(buf133, squeeze_19, buf134, buf135, 128, 196, grid=grid(128), stream=stream0)
        buf129 = buf99; del buf99  # reuse
        buf136 = buf82; del buf82  # reuse
        # Source Nodes: [identity, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_26.run(buf122, convolution_7, unsqueeze_226, buf126, squeeze_22, buf124, primals_23, convolution_6, unsqueeze_238, buf134, squeeze_19, primals_20, buf129, buf136, 3211264, grid=grid(3211264), stream=stream0)
        del buf122
        del buf124
        del convolution_6
        del convolution_7
        del primals_20
        del primals_23
        del squeeze_19
        del squeeze_22
        del unsqueeze_226
        del unsqueeze_238
        # Source Nodes: [identity], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        buf130 = aten.convolution_backward.default(buf129, relu_4, primals_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf129
        del primals_22
        buf131 = buf130[0]
        buf132 = buf130[1]
        del buf130
        # Source Nodes: [out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        buf137 = aten.convolution_backward.default(buf136, relu_5, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf136
        del primals_19
        buf138 = buf137[0]
        buf139 = buf137[1]
        del buf137
        buf140 = buf133; del buf133  # reuse
        buf142 = buf125; del buf125  # reuse
        # Source Nodes: [out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_22.run(relu_5, buf138, convolution_5, unsqueeze_250, buf140, buf142, 25088, 128, grid=grid(25088), stream=stream0)
        buf141 = buf134; del buf134  # reuse
        buf145 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf140, buf141, buf145, 128, 196, grid=grid(128), stream=stream0)
        del buf140
        buf143 = buf126; del buf126  # reuse
        buf144 = empty_strided_cuda((128, ), (1, ), torch.float16)
        # Source Nodes: [out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_20.run(buf142, squeeze_16, buf143, buf144, 128, 196, grid=grid(128), stream=stream0)
        del buf142
        buf146 = buf138; del buf138  # reuse
        # Source Nodes: [out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf146, relu_5, convolution_5, unsqueeze_250, buf143, squeeze_16, buf141, primals_17, 3211264, grid=grid(3211264), stream=stream0)
        del buf141
        del buf143
        del convolution_5
        del primals_17
        del relu_5
        del squeeze_16
        del unsqueeze_250
        # Source Nodes: [out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf147 = aten.convolution_backward.default(buf146, relu_4, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf146
        del primals_16
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf150 = empty_strided_cuda((64, 512), (1, 64), torch.float32)
        buf152 = empty_strided_cuda((64, 512), (1, 64), torch.float32)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_27.run(relu_4, buf131, buf148, convolution_4, unsqueeze_262, buf150, buf152, 32768, 196, grid=grid(32768), stream=stream0)
        buf151 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf156 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_28.run(buf150, buf151, buf156, 64, 512, grid=grid(64), stream=stream0)
        buf153 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf155 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29.run(buf152, squeeze_13, buf153, buf155, 64, 512, grid=grid(64), stream=stream0)
        buf157 = empty_strided_cuda((32, 64, 56, 56), (200704, 1, 3584, 64), torch.float16)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_30.run(relu_4, buf131, buf148, convolution_4, unsqueeze_262, buf153, squeeze_13, buf151, primals_14, buf157, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_4
        del primals_14
        del squeeze_13
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf158 = aten.convolution_backward.default(buf157, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf157
        del primals_13
        buf159 = buf158[0]
        buf160 = buf158[1]
        del buf158
        buf161 = buf152; del buf152  # reuse
        buf163 = buf150; del buf150  # reuse
        # Source Nodes: [out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31.run(relu_3, buf159, convolution_3, unsqueeze_274, buf161, buf163, 32768, 196, grid=grid(32768), stream=stream0)
        buf162 = buf153; del buf153  # reuse
        buf166 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_28.run(buf161, buf162, buf166, 64, 512, grid=grid(64), stream=stream0)
        buf164 = buf151; del buf151  # reuse
        buf165 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29.run(buf163, squeeze_10, buf164, buf165, 64, 512, grid=grid(64), stream=stream0)
        buf167 = buf159; del buf159  # reuse
        # Source Nodes: [out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf167, relu_3, convolution_3, unsqueeze_274, buf164, squeeze_10, buf162, primals_11, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_3
        del primals_11
        del relu_3
        del squeeze_10
        del unsqueeze_274
        # Source Nodes: [out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf168 = aten.convolution_backward.default(buf167, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf167
        del primals_10
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_33.run(buf171, relu_2, relu_4, buf148, buf169, 6422528, grid=grid(6422528), stream=stream0)
        del buf148
        del relu_2
        del relu_4
        buf172 = buf163; del buf163  # reuse
        buf174 = buf161; del buf161  # reuse
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_34.run(buf171, convolution_2, unsqueeze_286, buf172, buf174, 32768, 196, grid=grid(32768), stream=stream0)
        buf173 = buf164; del buf164  # reuse
        buf177 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_28.run(buf172, buf173, buf177, 64, 512, grid=grid(64), stream=stream0)
        buf175 = buf162; del buf162  # reuse
        buf176 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29.run(buf174, squeeze_7, buf175, buf176, 64, 512, grid=grid(64), stream=stream0)
        buf178 = buf169; del buf169  # reuse
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_35.run(buf171, convolution_2, unsqueeze_286, buf175, squeeze_7, buf173, primals_8, buf178, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_2
        del primals_8
        del squeeze_7
        del unsqueeze_286
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward]
        buf179 = aten.convolution_backward.default(buf178, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf178
        del primals_7
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = buf174; del buf174  # reuse
        buf184 = buf172; del buf172  # reuse
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31.run(relu_1, buf180, convolution_1, unsqueeze_298, buf182, buf184, 32768, 196, grid=grid(32768), stream=stream0)
        buf183 = buf175; del buf175  # reuse
        buf187 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_28.run(buf182, buf183, buf187, 64, 512, grid=grid(64), stream=stream0)
        buf185 = buf173; del buf173  # reuse
        buf186 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29.run(buf184, squeeze_4, buf185, buf186, 64, 512, grid=grid(64), stream=stream0)
        buf188 = buf180; del buf180  # reuse
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf188, relu_1, convolution_1, unsqueeze_298, buf185, squeeze_4, buf183, primals_5, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_298
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf189 = aten.convolution_backward.default(buf188, getitem_2, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf188
        del getitem_2
        del primals_4
        buf190 = buf189[0]
        buf191 = buf189[1]
        del buf189
        buf192 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_36.run(buf192, buf190, 6422528, grid=grid(6422528), stream=stream0)
        del buf190
        buf193 = empty_strided_cuda((32, 64, 112, 112), (802816, 1, 7168, 64), torch.float16)
        # Source Nodes: [x_3], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
        triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_37.run(getitem_3, buf192, buf193, 25690112, grid=grid(25690112), stream=stream0)
        del buf192
        del getitem_3
        buf194 = buf184; del buf184  # reuse
        buf196 = buf182; del buf182  # reuse
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_38.run(relu, buf193, convolution, unsqueeze_310, buf194, buf196, 32768, 784, grid=grid(32768), stream=stream0)
        buf195 = buf185; del buf185  # reuse
        buf199 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_28.run(buf194, buf195, buf199, 64, 512, grid=grid(64), stream=stream0)
        del buf194
        buf197 = buf183; del buf183  # reuse
        buf198 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_29.run(buf196, squeeze_1, buf197, buf198, 64, 512, grid=grid(64), stream=stream0)
        del buf196
        buf200 = buf193; del buf193  # reuse
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(buf200, relu, convolution, unsqueeze_310, buf197, squeeze_1, buf195, primals_2, 25690112, grid=grid(25690112), stream=stream0)
        del buf195
        del buf197
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_310
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf201 = aten.convolution_backward.default(buf200, primals_123, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf200
        del primals_1
        del primals_123
        buf202 = buf201[1]
        del buf201
    return (buf202, buf198, buf199, buf191, buf186, buf187, buf181, buf176, buf177, buf170, buf165, buf166, buf160, buf155, buf156, buf149, buf144, buf145, buf139, buf135, buf128, buf132, buf127, buf128, buf121, buf116, buf117, buf111, buf106, buf107, buf100, buf95, buf96, buf90, buf86, buf79, buf83, buf78, buf79, buf72, buf67, buf68, buf62, buf57, buf58, buf51, buf46, buf47, buf41, buf37, buf29, buf33, buf28, buf29, buf22, buf17, buf18, buf12, buf7, buf8, reinterpret_tensor(buf1, (1000, 512), (512, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float16)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float16)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    primals_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_123 = rand_strided((32, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float16)
    convolution = rand_strided((32, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float16)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((32, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float16)
    getitem_2 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    getitem_3 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int8)
    convolution_1 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    convolution_2 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    convolution_3 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    squeeze_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    convolution_4 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((32, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float16)
    convolution_5 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    squeeze_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    convolution_6 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    squeeze_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    squeeze_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    convolution_8 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    squeeze_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    convolution_9 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    squeeze_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((32, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float16)
    convolution_10 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    convolution_11 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    squeeze_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    squeeze_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    convolution_13 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    squeeze_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    convolution_14 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    squeeze_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((32, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float16)
    convolution_15 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    squeeze_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    convolution_16 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    squeeze_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    squeeze_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    convolution_18 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    squeeze_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    convolution_19 = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float16)
    squeeze_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    permute_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    le = rand_strided((32, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_82 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_94 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_106 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_118 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_142 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_166 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, 1000), (1000, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_123, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, view, permute_1, le, unsqueeze_82, unsqueeze_94, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, unsqueeze_154, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
