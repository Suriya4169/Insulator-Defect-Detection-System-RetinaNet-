import torch

def check_cuda():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        x = torch.rand(5, 3).cuda()
        print(f"Test Tensor on GPU:\n{x}")
    else:
        print("CUDA is NOT available. Training will be slow on CPU.")

if __name__ == "__main__":
    check_cuda()
