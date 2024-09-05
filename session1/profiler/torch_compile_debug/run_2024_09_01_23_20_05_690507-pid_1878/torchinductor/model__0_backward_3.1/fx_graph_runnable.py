
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True




isolate_fails_code_str = None



# torch version: 2.4.0+cu121
# torch cuda version: 12.1
# torch git version: e4ee3be4063b7c430974252fdf7db42273388d86


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2017 NVIDIA Corporation 
# Built on Fri_Sep__1_21:08:03_CDT_2017 
# Cuda compilation tools, release 9.0, V9.0.176 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 2080 Ti : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_123, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, view, permute_1, le, unsqueeze_82, unsqueeze_94, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, unsqueeze_154, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, tangents_1):
        mm = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        permute_3 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1 = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_4 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        view_2 = torch.ops.aten.view.default(mm, [32, 512, 1, 1]);  mm = None
        expand = torch.ops.aten.expand.default(view_2, [32, 512, 7, 7]);  view_2 = None
        div = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, div);  le = div = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_20 = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_82);  convolution_19 = unsqueeze_82 = None
        mul_140 = torch.ops.aten.mul.Tensor(where, sub_20)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_140, [0, 2, 3]);  mul_140 = None
        mul_141 = torch.ops.aten.mul.Tensor(sum_2, 0.0006377551020408163)
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(mul_141, 0);  mul_141 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 3);  unsqueeze_84 = None
        mul_142 = torch.ops.aten.mul.Tensor(sum_3, 0.0006377551020408163)
        mul_143 = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_144 = torch.ops.aten.mul.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(mul_144, 0);  mul_144 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, 2);  unsqueeze_86 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(unsqueeze_87, 3);  unsqueeze_87 = None
        mul_145 = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(mul_145, 0);  mul_145 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        mul_146 = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_88);  sub_20 = unsqueeze_88 = None
        sub_22 = torch.ops.aten.sub.Tensor(where, mul_146);  mul_146 = None
        sub_23 = torch.ops.aten.sub.Tensor(sub_22, unsqueeze_85);  sub_22 = unsqueeze_85 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_91);  sub_23 = unsqueeze_91 = None
        mul_148 = torch.ops.aten.mul.Tensor(sum_3, squeeze_58);  sum_3 = squeeze_58 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_147, relu_15, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_147 = primals_58 = None
        getitem_42 = convolution_backward[0]
        getitem_43 = convolution_backward[1];  convolution_backward = None
        le_1 = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_1 = torch.ops.aten.where.self(le_1, full_default, getitem_42);  le_1 = getitem_42 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_24 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_94);  convolution_18 = unsqueeze_94 = None
        mul_149 = torch.ops.aten.mul.Tensor(where_1, sub_24)
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_149, [0, 2, 3]);  mul_149 = None
        mul_150 = torch.ops.aten.mul.Tensor(sum_4, 0.0006377551020408163)
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(mul_150, 0);  mul_150 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        mul_151 = torch.ops.aten.mul.Tensor(sum_5, 0.0006377551020408163)
        mul_152 = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_153 = torch.ops.aten.mul.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(mul_153, 0);  mul_153 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        mul_154 = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(mul_154, 0);  mul_154 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
        mul_155 = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_100);  sub_24 = unsqueeze_100 = None
        sub_26 = torch.ops.aten.sub.Tensor(where_1, mul_155);  where_1 = mul_155 = None
        sub_27 = torch.ops.aten.sub.Tensor(sub_26, unsqueeze_97);  sub_26 = unsqueeze_97 = None
        mul_156 = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_103);  sub_27 = unsqueeze_103 = None
        mul_157 = torch.ops.aten.mul.Tensor(sum_5, squeeze_55);  sum_5 = squeeze_55 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_156, relu_14, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_156 = primals_55 = None
        getitem_45 = convolution_backward_1[0]
        getitem_46 = convolution_backward_1[1];  convolution_backward_1 = None
        add_108 = torch.ops.aten.add.Tensor(where, getitem_45);  where = getitem_45 = None
        le_2 = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_2 = torch.ops.aten.where.self(le_2, full_default, add_108);  le_2 = add_108 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_28 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_106);  convolution_17 = unsqueeze_106 = None
        mul_158 = torch.ops.aten.mul.Tensor(where_2, sub_28)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_158, [0, 2, 3]);  mul_158 = None
        mul_159 = torch.ops.aten.mul.Tensor(sum_6, 0.0006377551020408163)
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(mul_159, 0);  mul_159 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
        mul_160 = torch.ops.aten.mul.Tensor(sum_7, 0.0006377551020408163)
        mul_161 = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_162 = torch.ops.aten.mul.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(mul_162, 0);  mul_162 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
        mul_163 = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(mul_163, 0);  mul_163 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
        mul_164 = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_112);  sub_28 = unsqueeze_112 = None
        sub_30 = torch.ops.aten.sub.Tensor(where_2, mul_164);  mul_164 = None
        sub_31 = torch.ops.aten.sub.Tensor(sub_30, unsqueeze_109);  sub_30 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_115);  sub_31 = unsqueeze_115 = None
        mul_166 = torch.ops.aten.mul.Tensor(sum_7, squeeze_52);  sum_7 = squeeze_52 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_165, relu_12, primals_52, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_165 = primals_52 = None
        getitem_48 = convolution_backward_2[0]
        getitem_49 = convolution_backward_2[1];  convolution_backward_2 = None
        sub_32 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_118);  convolution_16 = unsqueeze_118 = None
        mul_167 = torch.ops.aten.mul.Tensor(where_2, sub_32)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_167, [0, 2, 3]);  mul_167 = None
        mul_169 = torch.ops.aten.mul.Tensor(sum_9, 0.0006377551020408163)
        mul_170 = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_171 = torch.ops.aten.mul.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(mul_171, 0);  mul_171 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
        mul_172 = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
        mul_173 = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_124);  sub_32 = unsqueeze_124 = None
        sub_34 = torch.ops.aten.sub.Tensor(where_2, mul_173);  where_2 = mul_173 = None
        sub_35 = torch.ops.aten.sub.Tensor(sub_34, unsqueeze_109);  sub_34 = unsqueeze_109 = None
        mul_174 = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_127);  sub_35 = unsqueeze_127 = None
        mul_175 = torch.ops.aten.mul.Tensor(sum_9, squeeze_49);  sum_9 = squeeze_49 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_174, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_174 = primals_49 = None
        getitem_51 = convolution_backward_3[0]
        getitem_52 = convolution_backward_3[1];  convolution_backward_3 = None
        le_3 = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_3 = torch.ops.aten.where.self(le_3, full_default, getitem_51);  le_3 = getitem_51 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_36 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_130);  convolution_15 = unsqueeze_130 = None
        mul_176 = torch.ops.aten.mul.Tensor(where_3, sub_36)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_176, [0, 2, 3]);  mul_176 = None
        mul_177 = torch.ops.aten.mul.Tensor(sum_10, 0.0006377551020408163)
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
        mul_178 = torch.ops.aten.mul.Tensor(sum_11, 0.0006377551020408163)
        mul_179 = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_180 = torch.ops.aten.mul.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(mul_180, 0);  mul_180 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
        mul_181 = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
        mul_182 = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_136);  sub_36 = unsqueeze_136 = None
        sub_38 = torch.ops.aten.sub.Tensor(where_3, mul_182);  where_3 = mul_182 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, unsqueeze_133);  sub_38 = unsqueeze_133 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_139);  sub_39 = unsqueeze_139 = None
        mul_184 = torch.ops.aten.mul.Tensor(sum_11, squeeze_46);  sum_11 = squeeze_46 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_183, relu_12, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_183 = primals_46 = None
        getitem_54 = convolution_backward_4[0]
        getitem_55 = convolution_backward_4[1];  convolution_backward_4 = None
        add_109 = torch.ops.aten.add.Tensor(getitem_48, getitem_54);  getitem_48 = getitem_54 = None
        le_4 = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_4 = torch.ops.aten.where.self(le_4, full_default, add_109);  le_4 = add_109 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_40 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_142);  convolution_14 = unsqueeze_142 = None
        mul_185 = torch.ops.aten.mul.Tensor(where_4, sub_40)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_185, [0, 2, 3]);  mul_185 = None
        mul_186 = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(mul_186, 0);  mul_186 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        mul_187 = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        mul_188 = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_189 = torch.ops.aten.mul.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(mul_189, 0);  mul_189 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
        mul_190 = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
        mul_191 = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_148);  sub_40 = unsqueeze_148 = None
        sub_42 = torch.ops.aten.sub.Tensor(where_4, mul_191);  mul_191 = None
        sub_43 = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_145);  sub_42 = unsqueeze_145 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_151);  sub_43 = unsqueeze_151 = None
        mul_193 = torch.ops.aten.mul.Tensor(sum_13, squeeze_43);  sum_13 = squeeze_43 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_192, relu_11, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_192 = primals_43 = None
        getitem_57 = convolution_backward_5[0]
        getitem_58 = convolution_backward_5[1];  convolution_backward_5 = None
        le_5 = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_5 = torch.ops.aten.where.self(le_5, full_default, getitem_57);  le_5 = getitem_57 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_44 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_154);  convolution_13 = unsqueeze_154 = None
        mul_194 = torch.ops.aten.mul.Tensor(where_5, sub_44)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_194, [0, 2, 3]);  mul_194 = None
        mul_195 = torch.ops.aten.mul.Tensor(sum_14, 0.00015943877551020407)
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(mul_195, 0);  mul_195 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
        mul_196 = torch.ops.aten.mul.Tensor(sum_15, 0.00015943877551020407)
        mul_197 = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_198 = torch.ops.aten.mul.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(mul_198, 0);  mul_198 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        mul_199 = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
        mul_200 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_160);  sub_44 = unsqueeze_160 = None
        sub_46 = torch.ops.aten.sub.Tensor(where_5, mul_200);  where_5 = mul_200 = None
        sub_47 = torch.ops.aten.sub.Tensor(sub_46, unsqueeze_157);  sub_46 = unsqueeze_157 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_163);  sub_47 = unsqueeze_163 = None
        mul_202 = torch.ops.aten.mul.Tensor(sum_15, squeeze_40);  sum_15 = squeeze_40 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_201, relu_10, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_201 = primals_40 = None
        getitem_60 = convolution_backward_6[0]
        getitem_61 = convolution_backward_6[1];  convolution_backward_6 = None
        add_110 = torch.ops.aten.add.Tensor(where_4, getitem_60);  where_4 = getitem_60 = None
        le_6 = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_6 = torch.ops.aten.where.self(le_6, full_default, add_110);  le_6 = add_110 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_48 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_166);  convolution_12 = unsqueeze_166 = None
        mul_203 = torch.ops.aten.mul.Tensor(where_6, sub_48)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_203, [0, 2, 3]);  mul_203 = None
        mul_204 = torch.ops.aten.mul.Tensor(sum_16, 0.00015943877551020407)
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(mul_204, 0);  mul_204 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
        mul_205 = torch.ops.aten.mul.Tensor(sum_17, 0.00015943877551020407)
        mul_206 = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_207 = torch.ops.aten.mul.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(mul_207, 0);  mul_207 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
        mul_208 = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
        mul_209 = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_172);  sub_48 = unsqueeze_172 = None
        sub_50 = torch.ops.aten.sub.Tensor(where_6, mul_209);  mul_209 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, unsqueeze_169);  sub_50 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_175);  sub_51 = unsqueeze_175 = None
        mul_211 = torch.ops.aten.mul.Tensor(sum_17, squeeze_37);  sum_17 = squeeze_37 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_210, relu_8, primals_37, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = primals_37 = None
        getitem_63 = convolution_backward_7[0]
        getitem_64 = convolution_backward_7[1];  convolution_backward_7 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_178);  convolution_11 = unsqueeze_178 = None
        mul_212 = torch.ops.aten.mul.Tensor(where_6, sub_52)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
        mul_214 = torch.ops.aten.mul.Tensor(sum_19, 0.00015943877551020407)
        mul_215 = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_216 = torch.ops.aten.mul.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(mul_216, 0);  mul_216 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
        mul_217 = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        mul_218 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_184);  sub_52 = unsqueeze_184 = None
        sub_54 = torch.ops.aten.sub.Tensor(where_6, mul_218);  where_6 = mul_218 = None
        sub_55 = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_169);  sub_54 = unsqueeze_169 = None
        mul_219 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_187);  sub_55 = unsqueeze_187 = None
        mul_220 = torch.ops.aten.mul.Tensor(sum_19, squeeze_34);  sum_19 = squeeze_34 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_219, relu_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_219 = primals_34 = None
        getitem_66 = convolution_backward_8[0]
        getitem_67 = convolution_backward_8[1];  convolution_backward_8 = None
        le_7 = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_7 = torch.ops.aten.where.self(le_7, full_default, getitem_66);  le_7 = getitem_66 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_56 = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_190);  convolution_10 = unsqueeze_190 = None
        mul_221 = torch.ops.aten.mul.Tensor(where_7, sub_56)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_221, [0, 2, 3]);  mul_221 = None
        mul_222 = torch.ops.aten.mul.Tensor(sum_20, 0.00015943877551020407)
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(mul_222, 0);  mul_222 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
        mul_223 = torch.ops.aten.mul.Tensor(sum_21, 0.00015943877551020407)
        mul_224 = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_225 = torch.ops.aten.mul.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
        mul_226 = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        mul_227 = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_196);  sub_56 = unsqueeze_196 = None
        sub_58 = torch.ops.aten.sub.Tensor(where_7, mul_227);  where_7 = mul_227 = None
        sub_59 = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_193);  sub_58 = unsqueeze_193 = None
        mul_228 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_199);  sub_59 = unsqueeze_199 = None
        mul_229 = torch.ops.aten.mul.Tensor(sum_21, squeeze_31);  sum_21 = squeeze_31 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_228, relu_8, primals_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_228 = primals_31 = None
        getitem_69 = convolution_backward_9[0]
        getitem_70 = convolution_backward_9[1];  convolution_backward_9 = None
        add_111 = torch.ops.aten.add.Tensor(getitem_63, getitem_69);  getitem_63 = getitem_69 = None
        le_8 = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_8 = torch.ops.aten.where.self(le_8, full_default, add_111);  le_8 = add_111 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_60 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_202);  convolution_9 = unsqueeze_202 = None
        mul_230 = torch.ops.aten.mul.Tensor(where_8, sub_60)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_230, [0, 2, 3]);  mul_230 = None
        mul_231 = torch.ops.aten.mul.Tensor(sum_22, 3.985969387755102e-05)
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(mul_231, 0);  mul_231 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
        mul_232 = torch.ops.aten.mul.Tensor(sum_23, 3.985969387755102e-05)
        mul_233 = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_234 = torch.ops.aten.mul.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        mul_235 = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
        mul_236 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_208);  sub_60 = unsqueeze_208 = None
        sub_62 = torch.ops.aten.sub.Tensor(where_8, mul_236);  mul_236 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_205);  sub_62 = unsqueeze_205 = None
        mul_237 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_211);  sub_63 = unsqueeze_211 = None
        mul_238 = torch.ops.aten.mul.Tensor(sum_23, squeeze_28);  sum_23 = squeeze_28 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_237, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_237 = primals_28 = None
        getitem_72 = convolution_backward_10[0]
        getitem_73 = convolution_backward_10[1];  convolution_backward_10 = None
        le_9 = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_9 = torch.ops.aten.where.self(le_9, full_default, getitem_72);  le_9 = getitem_72 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_64 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_214);  convolution_8 = unsqueeze_214 = None
        mul_239 = torch.ops.aten.mul.Tensor(where_9, sub_64)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2, 3]);  mul_239 = None
        mul_240 = torch.ops.aten.mul.Tensor(sum_24, 3.985969387755102e-05)
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_241 = torch.ops.aten.mul.Tensor(sum_25, 3.985969387755102e-05)
        mul_242 = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_243 = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_244 = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_245 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_220);  sub_64 = unsqueeze_220 = None
        sub_66 = torch.ops.aten.sub.Tensor(where_9, mul_245);  where_9 = mul_245 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_217);  sub_66 = unsqueeze_217 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_223);  sub_67 = unsqueeze_223 = None
        mul_247 = torch.ops.aten.mul.Tensor(sum_25, squeeze_25);  sum_25 = squeeze_25 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_246, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_246 = primals_25 = None
        getitem_75 = convolution_backward_11[0]
        getitem_76 = convolution_backward_11[1];  convolution_backward_11 = None
        add_112 = torch.ops.aten.add.Tensor(where_8, getitem_75);  where_8 = getitem_75 = None
        le_10 = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_10 = torch.ops.aten.where.self(le_10, full_default, add_112);  le_10 = add_112 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_68 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_226);  convolution_7 = unsqueeze_226 = None
        mul_248 = torch.ops.aten.mul.Tensor(where_10, sub_68)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_248, [0, 2, 3]);  mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(sum_26, 3.985969387755102e-05)
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_250 = torch.ops.aten.mul.Tensor(sum_27, 3.985969387755102e-05)
        mul_251 = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_252 = torch.ops.aten.mul.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_253 = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_232);  sub_68 = unsqueeze_232 = None
        sub_70 = torch.ops.aten.sub.Tensor(where_10, mul_254);  mul_254 = None
        sub_71 = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_229);  sub_70 = None
        mul_255 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_235);  sub_71 = unsqueeze_235 = None
        mul_256 = torch.ops.aten.mul.Tensor(sum_27, squeeze_22);  sum_27 = squeeze_22 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_255, relu_4, primals_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_255 = primals_22 = None
        getitem_78 = convolution_backward_12[0]
        getitem_79 = convolution_backward_12[1];  convolution_backward_12 = None
        sub_72 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_238);  convolution_6 = unsqueeze_238 = None
        mul_257 = torch.ops.aten.mul.Tensor(where_10, sub_72)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
        mul_259 = torch.ops.aten.mul.Tensor(sum_29, 3.985969387755102e-05)
        mul_260 = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_261 = torch.ops.aten.mul.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_262 = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_263 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_244);  sub_72 = unsqueeze_244 = None
        sub_74 = torch.ops.aten.sub.Tensor(where_10, mul_263);  where_10 = mul_263 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_229);  sub_74 = unsqueeze_229 = None
        mul_264 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_247);  sub_75 = unsqueeze_247 = None
        mul_265 = torch.ops.aten.mul.Tensor(sum_29, squeeze_19);  sum_29 = squeeze_19 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_264, relu_5, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_264 = primals_19 = None
        getitem_81 = convolution_backward_13[0]
        getitem_82 = convolution_backward_13[1];  convolution_backward_13 = None
        le_11 = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_11 = torch.ops.aten.where.self(le_11, full_default, getitem_81);  le_11 = getitem_81 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_76 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_250);  convolution_5 = unsqueeze_250 = None
        mul_266 = torch.ops.aten.mul.Tensor(where_11, sub_76)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3]);  mul_266 = None
        mul_267 = torch.ops.aten.mul.Tensor(sum_30, 3.985969387755102e-05)
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(mul_267, 0);  mul_267 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_268 = torch.ops.aten.mul.Tensor(sum_31, 3.985969387755102e-05)
        mul_269 = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_270 = torch.ops.aten.mul.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(mul_270, 0);  mul_270 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_271 = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_256);  sub_76 = unsqueeze_256 = None
        sub_78 = torch.ops.aten.sub.Tensor(where_11, mul_272);  where_11 = mul_272 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_253);  sub_78 = unsqueeze_253 = None
        mul_273 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_259);  sub_79 = unsqueeze_259 = None
        mul_274 = torch.ops.aten.mul.Tensor(sum_31, squeeze_16);  sum_31 = squeeze_16 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_273, relu_4, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_273 = primals_16 = None
        getitem_84 = convolution_backward_14[0]
        getitem_85 = convolution_backward_14[1];  convolution_backward_14 = None
        add_113 = torch.ops.aten.add.Tensor(getitem_78, getitem_84);  getitem_78 = getitem_84 = None
        le_12 = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_12 = torch.ops.aten.where.self(le_12, full_default, add_113);  le_12 = add_113 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_80 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_262);  convolution_4 = unsqueeze_262 = None
        mul_275 = torch.ops.aten.mul.Tensor(where_12, sub_80)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_275, [0, 2, 3]);  mul_275 = None
        mul_276 = torch.ops.aten.mul.Tensor(sum_32, 9.964923469387754e-06)
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_277 = torch.ops.aten.mul.Tensor(sum_33, 9.964923469387754e-06)
        mul_278 = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_279 = torch.ops.aten.mul.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_279, 0);  mul_279 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_280 = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(mul_280, 0);  mul_280 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_281 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_268);  sub_80 = unsqueeze_268 = None
        sub_82 = torch.ops.aten.sub.Tensor(where_12, mul_281);  mul_281 = None
        sub_83 = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_265);  sub_82 = unsqueeze_265 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_271);  sub_83 = unsqueeze_271 = None
        mul_283 = torch.ops.aten.mul.Tensor(sum_33, squeeze_13);  sum_33 = squeeze_13 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_282, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_282 = primals_13 = None
        getitem_87 = convolution_backward_15[0]
        getitem_88 = convolution_backward_15[1];  convolution_backward_15 = None
        le_13 = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_13 = torch.ops.aten.where.self(le_13, full_default, getitem_87);  le_13 = getitem_87 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_84 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_274);  convolution_3 = unsqueeze_274 = None
        mul_284 = torch.ops.aten.mul.Tensor(where_13, sub_84)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_284, [0, 2, 3]);  mul_284 = None
        mul_285 = torch.ops.aten.mul.Tensor(sum_34, 9.964923469387754e-06)
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_286 = torch.ops.aten.mul.Tensor(sum_35, 9.964923469387754e-06)
        mul_287 = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_288 = torch.ops.aten.mul.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_289 = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(mul_289, 0);  mul_289 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_280);  sub_84 = unsqueeze_280 = None
        sub_86 = torch.ops.aten.sub.Tensor(where_13, mul_290);  where_13 = mul_290 = None
        sub_87 = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_277);  sub_86 = unsqueeze_277 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_283);  sub_87 = unsqueeze_283 = None
        mul_292 = torch.ops.aten.mul.Tensor(sum_35, squeeze_10);  sum_35 = squeeze_10 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_291, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_291 = primals_10 = None
        getitem_90 = convolution_backward_16[0]
        getitem_91 = convolution_backward_16[1];  convolution_backward_16 = None
        add_114 = torch.ops.aten.add.Tensor(where_12, getitem_90);  where_12 = getitem_90 = None
        le_14 = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_14 = torch.ops.aten.where.self(le_14, full_default, add_114);  le_14 = add_114 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_88 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_286);  convolution_2 = unsqueeze_286 = None
        mul_293 = torch.ops.aten.mul.Tensor(where_14, sub_88)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(sum_36, 9.964923469387754e-06)
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_295 = torch.ops.aten.mul.Tensor(sum_37, 9.964923469387754e-06)
        mul_296 = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_297 = torch.ops.aten.mul.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_298 = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_299 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_292);  sub_88 = unsqueeze_292 = None
        sub_90 = torch.ops.aten.sub.Tensor(where_14, mul_299);  mul_299 = None
        sub_91 = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_289);  sub_90 = unsqueeze_289 = None
        mul_300 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_295);  sub_91 = unsqueeze_295 = None
        mul_301 = torch.ops.aten.mul.Tensor(sum_37, squeeze_7);  sum_37 = squeeze_7 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_300, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_300 = primals_7 = None
        getitem_93 = convolution_backward_17[0]
        getitem_94 = convolution_backward_17[1];  convolution_backward_17 = None
        le_15 = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_15 = torch.ops.aten.where.self(le_15, full_default, getitem_93);  le_15 = getitem_93 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_92 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_298);  convolution_1 = unsqueeze_298 = None
        mul_302 = torch.ops.aten.mul.Tensor(where_15, sub_92)
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3]);  mul_302 = None
        mul_303 = torch.ops.aten.mul.Tensor(sum_38, 9.964923469387754e-06)
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_304 = torch.ops.aten.mul.Tensor(sum_39, 9.964923469387754e-06)
        mul_305 = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_306 = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_307 = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_304);  sub_92 = unsqueeze_304 = None
        sub_94 = torch.ops.aten.sub.Tensor(where_15, mul_308);  where_15 = mul_308 = None
        sub_95 = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_301);  sub_94 = unsqueeze_301 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_307);  sub_95 = unsqueeze_307 = None
        mul_310 = torch.ops.aten.mul.Tensor(sum_39, squeeze_4);  sum_39 = squeeze_4 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_309, getitem_2, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_309 = getitem_2 = primals_4 = None
        getitem_96 = convolution_backward_18[0]
        getitem_97 = convolution_backward_18[1];  convolution_backward_18 = None
        add_115 = torch.ops.aten.add.Tensor(where_14, getitem_96);  where_14 = getitem_96 = None
        _low_memory_max_pool2d_offsets_to_indices = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_3, 3, 112, [2, 2], [1, 1]);  getitem_3 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(add_115, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices);  add_115 = _low_memory_max_pool2d_offsets_to_indices = None
        le_16 = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_16 = torch.ops.aten.where.self(le_16, full_default, max_pool2d_with_indices_backward);  le_16 = full_default = max_pool2d_with_indices_backward = None
        sum_40 = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_96 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_310);  convolution = unsqueeze_310 = None
        mul_311 = torch.ops.aten.mul.Tensor(where_16, sub_96)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_311, [0, 2, 3]);  mul_311 = None
        mul_312 = torch.ops.aten.mul.Tensor(sum_40, 2.4912308673469386e-06)
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(mul_312, 0);  mul_312 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(sum_41, 2.4912308673469386e-06)
        mul_314 = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_315 = torch.ops.aten.mul.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_316 = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_317 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_316);  sub_96 = unsqueeze_316 = None
        sub_98 = torch.ops.aten.sub.Tensor(where_16, mul_317);  where_16 = mul_317 = None
        sub_99 = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_313);  sub_98 = unsqueeze_313 = None
        mul_318 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_319);  sub_99 = unsqueeze_319 = None
        mul_319 = torch.ops.aten.mul.Tensor(sum_41, squeeze_1);  sum_41 = squeeze_1 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_318, primals_123, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_318 = primals_123 = primals_1 = None
        getitem_100 = convolution_backward_19[1];  convolution_backward_19 = None
        return [getitem_100, mul_319, sum_40, getitem_97, mul_310, sum_38, getitem_94, mul_301, sum_36, getitem_91, mul_292, sum_34, getitem_88, mul_283, sum_32, getitem_85, mul_274, sum_30, getitem_82, mul_265, sum_26, getitem_79, mul_256, sum_26, getitem_76, mul_247, sum_24, getitem_73, mul_238, sum_22, getitem_70, mul_229, sum_20, getitem_67, mul_220, sum_16, getitem_64, mul_211, sum_16, getitem_61, mul_202, sum_14, getitem_58, mul_193, sum_12, getitem_55, mul_184, sum_10, getitem_52, mul_175, sum_6, getitem_49, mul_166, sum_6, getitem_46, mul_157, sum_4, getitem_43, mul_148, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 7, 7), (147, 1, 21, 3), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64,), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # primals_5
    buf4 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_7
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # primals_8
    buf6 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_10
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # primals_11
    buf8 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_13
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # primals_14
    buf10 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_16
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # primals_17
    buf12 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128, 128, 3, 3), (1152, 1, 384, 128), is_leaf=True)  # primals_19
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # primals_20
    buf14 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128, 64, 1, 1), is_leaf=True)  # primals_22
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # primals_23
    buf16 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128, 128, 3, 3), (1152, 1, 384, 128), is_leaf=True)  # primals_25
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128,), is_leaf=True)  # primals_26
    buf18 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128, 128, 3, 3), (1152, 1, 384, 128), is_leaf=True)  # primals_28
    buf19 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128,), is_leaf=True)  # primals_29
    buf20 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf20, (256, 128, 3, 3), (1152, 1, 384, 128), is_leaf=True)  # primals_31
    buf21 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256,), is_leaf=True)  # primals_32
    buf22 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_34
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # primals_35
    buf24 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256, 128, 1, 1), is_leaf=True)  # primals_37
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), is_leaf=True)  # primals_38
    buf26 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_40
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256,), is_leaf=True)  # primals_41
    buf28 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_43
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # primals_44
    buf30 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_46
    buf31 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512,), is_leaf=True)  # primals_47
    buf32 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf32, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_49
    buf33 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512,), is_leaf=True)  # primals_50
    buf34 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512, 256, 1, 1), is_leaf=True)  # primals_52
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # primals_53
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_55
    buf37 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512,), is_leaf=True)  # primals_56
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_58
    buf39 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512,), is_leaf=True)  # primals_59
    buf40 = reader.storage(None, 19267584, device=device(type='cuda', index=0))
    reader.tensor(buf40, (32, 3, 224, 224), (150528, 1, 672, 3), is_leaf=True)  # primals_123
    buf41 = reader.storage(None, 102760448, device=device(type='cuda', index=0))
    reader.tensor(buf41, (32, 64, 112, 112), (802816, 1, 7168, 64), is_leaf=True)  # convolution
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # squeeze_1
    buf43 = reader.storage(None, 102760448, device=device(type='cuda', index=0))
    reader.tensor(buf43, (32, 64, 112, 112), (802816, 1, 7168, 64), is_leaf=True)  # relu
    buf44 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf44, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # getitem_2
    buf45 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf45, (32, 64, 56, 56), (200704, 1, 3584, 64), dtype=torch.int8, is_leaf=True)  # getitem_3
    buf46 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf46, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # convolution_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # squeeze_4
    buf48 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf48, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # relu_1
    buf49 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf49, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # convolution_2
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # squeeze_7
    buf51 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf51, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # relu_2
    buf52 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf52, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # convolution_3
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # squeeze_10
    buf54 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf54, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # relu_3
    buf55 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf55, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # convolution_4
    buf56 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64,), is_leaf=True)  # squeeze_13
    buf57 = reader.storage(None, 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf57, (32, 64, 56, 56), (200704, 1, 3584, 64), is_leaf=True)  # relu_4
    buf58 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf58, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # convolution_5
    buf59 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf59, (128,), is_leaf=True)  # squeeze_16
    buf60 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf60, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # relu_5
    buf61 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf61, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # convolution_6
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # squeeze_19
    buf63 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf63, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # convolution_7
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # squeeze_22
    buf65 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf65, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # relu_6
    buf66 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf66, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # convolution_8
    buf67 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128,), is_leaf=True)  # squeeze_25
    buf68 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf68, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # relu_7
    buf69 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf69, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # convolution_9
    buf70 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128,), is_leaf=True)  # squeeze_28
    buf71 = reader.storage(None, 12845056, device=device(type='cuda', index=0))
    reader.tensor(buf71, (32, 128, 28, 28), (100352, 1, 3584, 128), is_leaf=True)  # relu_8
    buf72 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf72, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # convolution_10
    buf73 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256,), is_leaf=True)  # squeeze_31
    buf74 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf74, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # relu_9
    buf75 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf75, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # convolution_11
    buf76 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf76, (256,), is_leaf=True)  # squeeze_34
    buf77 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf77, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # convolution_12
    buf78 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf78, (256,), is_leaf=True)  # squeeze_37
    buf79 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf79, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # relu_10
    buf80 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf80, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # convolution_13
    buf81 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256,), is_leaf=True)  # squeeze_40
    buf82 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf82, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # relu_11
    buf83 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf83, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # convolution_14
    buf84 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf84, (256,), is_leaf=True)  # squeeze_43
    buf85 = reader.storage(None, 6422528, device=device(type='cuda', index=0))
    reader.tensor(buf85, (32, 256, 14, 14), (50176, 1, 3584, 256), is_leaf=True)  # relu_12
    buf86 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf86, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # convolution_15
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # squeeze_46
    buf88 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf88, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # relu_13
    buf89 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf89, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # convolution_16
    buf90 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512,), is_leaf=True)  # squeeze_49
    buf91 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf91, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # convolution_17
    buf92 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512,), is_leaf=True)  # squeeze_52
    buf93 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf93, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # relu_14
    buf94 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf94, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # convolution_18
    buf95 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512,), is_leaf=True)  # squeeze_55
    buf96 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf96, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # relu_15
    buf97 = reader.storage(None, 3211264, device=device(type='cuda', index=0))
    reader.tensor(buf97, (32, 512, 7, 7), (25088, 1, 3584, 512), is_leaf=True)  # convolution_19
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # squeeze_58
    buf99 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf99, (32, 512), is_leaf=True)  # view
    buf100 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1000, 512), is_leaf=True)  # permute_1
    buf101 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf101, (32, 512, 7, 7), (25088, 1, 3584, 512), dtype=torch.bool, is_leaf=True)  # le
    buf102 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1, 512, 1, 1), is_leaf=True)  # unsqueeze_82
    buf103 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1, 512, 1, 1), is_leaf=True)  # unsqueeze_94
    buf104 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf104, (1, 512, 1, 1), is_leaf=True)  # unsqueeze_106
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1, 512, 1, 1), is_leaf=True)  # unsqueeze_118
    buf106 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1, 512, 1, 1), is_leaf=True)  # unsqueeze_130
    buf107 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1, 256, 1, 1), is_leaf=True)  # unsqueeze_142
    buf108 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1, 256, 1, 1), is_leaf=True)  # unsqueeze_154
    buf109 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1, 256, 1, 1), is_leaf=True)  # unsqueeze_166
    buf110 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1, 256, 1, 1), is_leaf=True)  # unsqueeze_178
    buf111 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf111, (1, 256, 1, 1), is_leaf=True)  # unsqueeze_190
    buf112 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf112, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_202
    buf113 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf113, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_214
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_226
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_238
    buf116 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1, 128, 1, 1), is_leaf=True)  # unsqueeze_250
    buf117 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_262
    buf118 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_274
    buf119 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_286
    buf120 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_298
    buf121 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1, 64, 1, 1), is_leaf=True)  # unsqueeze_310
    buf122 = reader.storage(None, 128000, device=device(type='cuda', index=0))
    reader.tensor(buf122, (32, 1000), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)