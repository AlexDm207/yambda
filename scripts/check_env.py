import time, torch

print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, VRAM: {p.total_memory/1e9:.1f} GB")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        y = x @ x
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"fp16 matmul: {50 * 2 * 4096**3 / dt / 1e12:.1f} TFLOPS")
