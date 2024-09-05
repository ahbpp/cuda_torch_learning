
## Plan for today
# colors in markdown: 
- [x] Check torch compile for simple model (just linear layer) and more complex (classification or detection) models (Compile also fused kernels) []
- [x] check how TORCH_LOGS work (we can check compile outpus (triton kernels)) 
- [x] check torch profiler  
- [ ] ncu profiler
- [x] check torch compile options and and fuse options
- [ ]  Summarize everything about torch compile and link docs and tutorials (check cuda_mode for the same)
- [ ] check if training with compile is actually faster




This part is based on Lecture 1 from CUDA MODE:
* [REPO](https://github.com/cuda-mode/lectures/tree/main/lecture_001)
* [YouTube](https://www.youtube.com/watch?v=LuhJEEJQgUM)


NCU: https://developer.nvidia.com/tools-overview/nsight-compute/get-started
## Torch compile

* [FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html#torch-compiler-graph-breaks)
* [How are you (torch.compile) speeding up my code?](https://pytorch.org/docs/stable/torch.compiler_faq.html#how-are-you-speeding-up-my-code)
* [GPU optimization workshop](https://www.youtube.com/live/v_q2JTIqE20)

Torch compile with `inductor` (default) backend firstly fuse all ops and  then generates [Triton](https://openai.com/index/triton/) kernels. 


Torch compile doesn't give any perfomance speed up for 2080TI and fp32, but it looks like it speeds up for fp16.   

Probable reasons:
1. torch.compile models inference is compute bounded task on 2080TI, but it is memory bounded for A100 and etc
2.  Triton is optimized for high but not for 2080TI
   
You can get Triton kernels from torch.compile by adding options: 
```
{'trace.output_code': True, 'trace.enabled': True}
```
Compiled resnet cod from TORCH_LOGS: `session1/compile/torch_compile_debug/run_2024_09_02_17_44_38_946469-pid_11747/torchinductor/model__0_forward_1.0/output_code.py` 
 
or running code with `TORCH_LOGS` env var
```
TORCH_LOGS="output_code" python compile_resnet.py
```



* 

## Torch Profile

* [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#pytorch-profiler)
* [Profiling to understand torch.compile performance](https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html)
* [Triton](https://triton-lang.org/main/index.html)



