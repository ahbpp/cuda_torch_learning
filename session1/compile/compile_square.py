import torch
import torch.nn as nn   
import torchvision.models as models

DEVICE = "cuda"
DTYPE = torch.float32

def fwd_bwd(model, inp):
    out = model(inp)
    # out.sum().backward()

def main():
    input = torch.randn((32, 10), device=DEVICE, dtype=DTYPE)
    sq_c = torch.compile(torch.square, options={'trace.output_code': True, 'trace.enabled': True})
    print("Compilation done")
    sq_c(input)
    

if __name__ == "__main__":
    main()
