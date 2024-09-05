import torch
import torch.nn as nn   
import torchvision.models as models

DEVICE = "cuda"
DTYPE = torch.float32

def fwd_bwd(model, inp):
    out = model(inp)
    # out.sum().backward()

def main():
    model = nn.Sequential(nn.Linear(10, 10),
                          nn.ReLU(),
                          nn.Linear(10, 10)).to(DEVICE).to(DTYPE)
    inputs = [torch.randn((32, 10), device=DEVICE, dtype=DTYPE)
              for _ in range(2)]
    model_c = torch.compile(model,
                            options={}
                            )
    print("Compilation done")
    model_c(inputs[0])
    

if __name__ == "__main__":
    main()
