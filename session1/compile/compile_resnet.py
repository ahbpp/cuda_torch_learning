import torch
import torchvision.models as models

DEVICE = "cuda"
DTYPE = torch.float32

def fwd_bwd(model, inp):
    out = model(inp)
    # out.sum().backward()

def main():
    model = models.resnet18().to(DEVICE).to(DTYPE)
    inputs = [torch.randn((32, 3, 224, 224), device=DEVICE, dtype=DTYPE)
              for _ in range(2)]
    model_c = torch.compile(model,
                            options={'trace.output_code': True,
                                     'trace.enabled': True},
                            )
    print("Compilation done")
    fwd_bwd(model_c, inputs[0])
    

if __name__ == "__main__":
    main()
