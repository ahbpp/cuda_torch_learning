import torch
import os
import torchvision.models as models

import torch._dynamo as dynamo
from torch.profiler import profile, record_function, ProfilerActivity

from pprint import pprint

from utils import time_pytorch_function

# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#pytorch-profiler
# https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
DEVICE = "cuda"
DTYPE = torch.float32
output_dir = "tb"
NUM_RUNS = 32
COMPILE_MODE = "default" # or reduce-overhead



def fwd_bwd(model, inp):
    out = model(inp)
    out.sum().backward()

def main():
    model = models.resnet18().to(DEVICE).to(DTYPE)
    inputs = [torch.randn((32, 3, 224, 224), device=DEVICE, dtype=DTYPE)
              for _ in range(NUM_RUNS)]
    
    # # warm up
    # fwd_bwd(model, inputs[0])
    # fwd_bwd(model, inputs[0])
    # with torch.profiler.profile() as prof:
    #     with record_function("eager_mode"):
    #         for i in range(1, NUM_RUNS):
    #             fwd_bwd(model, inputs[i])
    #             prof.step()
    # prof.export_chrome_trace(f"{output_dir}/trace_eager_fp16.json")




    model_c = torch.compile(model, dynamic=False,
                            # mode=COMPILE_MODE, 
                            options={ 'dynamic_scale_rblock': False, 
                                # 'triton.max_tiles': 32, 
                                    #  'cuda.arch': 'sm_75'
                                     }
                            )
    print("Compilation done")
    # warm up
    fwd_bwd(model_c, inputs[0])
    fwd_bwd(model_c, inputs[0])
    with torch.profiler.profile() as prof:
        with record_function("compile_mode_ro"):
            for i in range(1, NUM_RUNS):
                fwd_bwd(model_c, inputs[i])
                prof.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace(f"{output_dir}/compile_{COMPILE_MODE}_fp32.json")




    # 
    

    # print("Model device: ", next(model.parameters()).device)
    # print("Input device: ", inputs.device)

    # time_pytorch_function(model, inputs)
    

    # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], 
    #             #  with_stack=True,
    #               profile_memory=True,
    #              ) as prof:
    #     model(inputs)
    #     # with record_function("model_inference"):
            
        
            
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    # # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # # os.makedirs(output_dir, exist_ok=True)
    # prof.export_chrome_trace(f"{output_dir}/trace.json")


if __name__ == "__main__":
    main()
