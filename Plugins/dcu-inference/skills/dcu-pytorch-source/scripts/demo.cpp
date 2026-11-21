#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== LibTorch DCU Verification Demo ===" << std::endl;

    // 1. Print PyTorch version
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

    // 2. Check HIP/CUDA device availability
    bool has_hip = torch::cuda::is_available();
    std::cout << "HIP/CUDA available: " << (has_hip ? "YES" : "NO") << std::endl;

    if (!has_hip) {
        std::cerr << "ERROR: No DCU device found!" << std::endl;
        return 1;
    }

    // 3. Print device count
    int device_count = (int)torch::cuda::device_count();
    std::cout << "Device count: " << device_count << std::endl;

    // 4. Create tensors on DCU and do a simple matmul
    torch::Device device(torch::kCUDA, 0);
    std::cout << "\nUsing device: " << device << std::endl;

    // Create random float tensors directly on device
    auto a = torch::rand({512, 512}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto b = torch::rand({512, 512}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    // Matrix multiplication: c = a @ b
    auto c = torch::matmul(a, b);

    // Synchronize to ensure computation is done
    torch::cuda::synchronize();

    std::cout << "Input  a: dtype=" << a.dtype() << " shape=" << a.sizes()
              << " device=" << a.device() << std::endl;
    std::cout << "Input  b: dtype=" << b.dtype() << " shape=" << b.sizes()
              << " device=" << b.device() << std::endl;
    std::cout << "Output c: dtype=" << c.dtype() << " shape=" << c.sizes()
              << " device=" << c.device() << std::endl;

    float sum = c.sum().item<float>();
    std::cout << "Result sum: " << std::fixed << std::setprecision(4) << sum << std::endl;

    // 5. Cross-validate against CPU
    auto a_cpu = a.to(torch::kCPU);
    auto b_cpu = b.to(torch::kCPU);
    auto c_cpu = torch::matmul(a_cpu, b_cpu);
    float sum_cpu = c_cpu.sum().item<float>();
    float diff = std::abs(sum - sum_cpu);

    std::cout << "CPU   sum: " << std::fixed << std::setprecision(4) << sum_cpu << std::endl;
    std::cout << "Difference: " << std::scientific << diff << std::endl;

    if (diff < 1e-2) {
        std::cout << "\nPASS: DCU libtorch works correctly!" << std::endl;
        return 0;
    } else {
        std::cerr << "\nFAIL: DCU vs CPU mismatch!" << std::endl;
        return 1;
    }
}
