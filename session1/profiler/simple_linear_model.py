import torch
import torch.nn as nn

from utils import time_pytorch_function

DEVICE = "cuda"


class SimpleLinearModel(nn.Module):
    def __init__(self, num_features):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        return self.linear(x)
    

def main():
    num_features = 1024
    model = SimpleLinearModel(num_features).to(DEVICE)
    input = torch.randn(10000, num_features).to(DEVICE)

    inference_time = time_pytorch_function(model, input)
    print(f"Inference time: {inference_time} ms")

    # # Now profile each function using pytorch profiler
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()


