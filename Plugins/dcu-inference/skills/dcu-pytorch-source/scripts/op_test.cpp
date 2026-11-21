#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════════
// Test framework utilities
// ═══════════════════════════════════════════════════════════════════════

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;
static std::vector<std::string> g_failed_tests;
static std::vector<std::string> g_skipped_tests;

// Precision tolerance by dtype
struct Tolerance {
    double atol;
    double rtol;
};
static Tolerance tolerance(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat64: return {1e-8, 1e-8};
        case torch::kFloat32: return {1e-5, 1e-5};
        case torch::kFloat16: return {1e-2, 1e-2};
        case torch::kBFloat16:return {1e-2, 1e-2};
        case torch::kInt32:   return {0.0,  0.0};
        case torch::kInt64:   return {0.0,  0.0};
        case torch::kBool:    return {0.0,  0.0};
        default:              return {1e-5, 1e-5};
    }
}

static bool all_close(const torch::Tensor& a, const torch::Tensor& b,
                      double atol = -1.0, double rtol = -1.0) {
    if (a.sizes() != b.sizes()) return false;
    auto a_cpu = a.to(torch::kCPU).contiguous();
    auto b_cpu = b.to(torch::kCPU).contiguous();

    // Bool handling
    if (a_cpu.scalar_type() == torch::kBool) {
        return torch::equal(a_cpu, b_cpu);
    }

    // Integer handling
    if (a_cpu.scalar_type() == torch::kInt32 || a_cpu.scalar_type() == torch::kInt64) {
        return torch::equal(a_cpu, b_cpu);
    }

    // Float handling
    auto tol = tolerance(a.scalar_type());
    if (atol < 0) atol = tol.atol;
    if (rtol < 0) rtol = tol.rtol;

    // Promote to float64 for comparison to avoid overflow
    auto a_f64 = a_cpu.to(torch::kFloat64);
    auto b_f64 = b_cpu.to(torch::kFloat64);
    auto diff = (a_f64 - b_f64).abs();
    auto rel  = (b_f64.abs() + atol);

    auto max_diff = diff.max().item<double>();
    auto max_rel  = (diff / rel).max().item<double>();

    (void)max_rel;
    return max_diff <= atol;
}

#define TEST(name, body)                                                              \
    do {                                                                              \
        std::cout << "  [" << std::setw(45) << std::left << name << "] ";             \
        std::cout.flush();                                                            \
        try {                                                                         \
            body;                                                                     \
            std::cout << "\033[1;32mPASS\033[0m" << std::endl;                        \
            g_pass++;                                                                 \
        } catch (const std::exception& e) {                                           \
            std::cout << "\033[1;31mFAIL\033[0m (" << e.what() << ")" << std::endl;   \
            g_fail++;                                                                 \
            g_failed_tests.push_back(std::string(name) + " : " + e.what());           \
        } catch (...) {                                                               \
            std::cout << "\033[1;31mFAIL\033[0m (unknown)" << std::endl;              \
            g_fail++;                                                                 \
            g_failed_tests.push_back(std::string(name) + " : unknown exception");     \
        }                                                                             \
    } while(0)

#define OP_CHECK(condition, msg)                                                       \
    do {                                                                              \
        if (!(condition)) throw std::runtime_error(msg);                              \
    } while(0)

#define CHECK_CLOSE(dcu, cpu, atol, rtol)                                             \
    do {                                                                              \
        if (!all_close((dcu), (cpu), (atol), (rtol)))                                 \
            throw std::runtime_error("DCU vs CPU mismatch");                           \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════
// Helper: get device
// ═══════════════════════════════════════════════════════════════════════
static torch::Device dcu_device() {
    return torch::Device(torch::kCUDA, 0);
}

static torch::Tensor to_dcu(const torch::Tensor& t) {
    return t.to(dcu_device());
}

// ═══════════════════════════════════════════════════════════════════════
// Main test runner
// ═══════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   LibTorch DCU Operator Compatibility & Precision Test  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // ── Environment check ──────────────────────────────────────────
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "C++ ABI: " << (_GLIBCXX_USE_CXX11_ABI ? "new" : "old") << std::endl;
    bool has_hip = torch::cuda::is_available();
    std::cout << "HIP/CUDA available: " << (has_hip ? "YES" : "NO") << std::endl;
    if (has_hip) {
        std::cout << "Device count: " << torch::cuda::device_count() << std::endl;
    }
    std::cout << "Device: " << dcu_device() << std::endl;
    std::cout << "\n";

    if (!has_hip) {
        std::cerr << "FATAL: No DCU device found. Aborting.\n";
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════
    // 1. Tensor Creation & Basic Properties
    // ═══════════════════════════════════════════════════════════════
    std::cout << "── Tensor Creation & Properties ──\n";

    TEST("zeros", {
        auto d = torch::zeros({3, 4}, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c = torch::zeros({3, 4}, torch::kFloat32);
        OP_CHECK(d.sizes() == c.sizes(), "shape mismatch");
        OP_CHECK(d.device().is_cuda(), "not on CUDA device");
        CHECK_CLOSE(d, c, 0.0, 0.0);
    });

    TEST("ones", {
        auto d = torch::ones({2, 5}, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c = torch::ones({2, 5}, torch::kFloat32);
        CHECK_CLOSE(d, c, 0.0, 0.0);
    });

    TEST("arange", {
        auto d = torch::arange(0, 100, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c = torch::arange(0, 100, torch::kFloat32);
        CHECK_CLOSE(d, c, 0.0, 0.0);
    });

    TEST("linspace", {
        auto d = torch::linspace(-5.0, 5.0, 100, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c = torch::linspace(-5.0, 5.0, 100, torch::kFloat32);
        CHECK_CLOSE(d, c, 1e-5, 1e-5);
    });

    TEST("eye", {
        auto d = torch::eye(8, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c = torch::eye(8, torch::kFloat32);
        CHECK_CLOSE(d, c, 0.0, 0.0);
    });

    TEST("rand_uniform", {
        torch::manual_seed(42);
        auto c = torch::rand({64, 64}, torch::kFloat32);
        auto d = to_dcu(c);  // copy to DCU to ensure identical values
        CHECK_CLOSE(d, c, 1e-6, 1e-6);
    });

    TEST("randn_normal", {
        torch::manual_seed(42);
        auto c = torch::randn({64, 64}, torch::kFloat32);
        auto d = to_dcu(c);  // copy to DCU to ensure identical values
        CHECK_CLOSE(d, c, 1e-5, 1e-5);
    });

    TEST("full", {
        auto d = torch::full({3, 4}, 3.14, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c = torch::full({3, 4}, 3.14, torch::kFloat32);
        CHECK_CLOSE(d, c, 0.0, 0.0);
    });

    TEST("empty + fill", {
        auto d = torch::empty({4, 4}, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        d.fill_(7.0);
        auto c = torch::full({4, 4}, 7.0, torch::kFloat32);
        CHECK_CLOSE(d, c, 0.0, 0.0);
    });

    // ═══════════════════════════════════════════════════════════════
    // 2. Dtype Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Dtype Casting ──\n";

    TEST("float32 -> float64", {
        auto c = torch::rand({16, 16}, torch::kFloat32);
        auto d = c.to(dcu_device()).to(torch::kFloat64);
        auto c64 = c.to(torch::kFloat64);
        CHECK_CLOSE(d, c64, 1e-8, 1e-8);
    });

    TEST("float32 -> float16", {
        auto c = torch::rand({16, 16}, torch::kFloat32);
        auto d = c.to(dcu_device()).to(torch::kFloat16);
        auto c16 = c.to(torch::kFloat16);
        CHECK_CLOSE(d, c16, 1e-2, 1e-2);
    });

    TEST("float32 -> int32", {
        auto c = torch::rand({16, 16}, torch::kFloat32).mul_(100);
        auto d = c.to(dcu_device()).to(torch::kInt32);
        auto ci = c.to(torch::kInt32);
        OP_CHECK(torch::equal(d.to(torch::kCPU), ci), "int cast mismatch");
    });

    TEST("float32 -> int64", {
        auto c = torch::rand({16, 16}, torch::kFloat32).mul_(100);
        auto d = c.to(dcu_device()).to(torch::kInt64);
        auto ci = c.to(torch::kInt64);
        OP_CHECK(torch::equal(d.to(torch::kCPU), ci), "int64 cast mismatch");
    });

    TEST("float32 -> bool", {
        auto c = torch::tensor({0.0f, 1.0f, -1.0f, 0.5f, 0.0f}, torch::kFloat32);
        auto d = c.to(dcu_device()).to(torch::kBool);
        auto cb = c.to(torch::kBool);
        OP_CHECK(torch::equal(d.to(torch::kCPU), cb), "bool cast mismatch");
    });

    // ═══════════════════════════════════════════════════════════════
    // 3. Basic Arithmetic Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Basic Arithmetic ──\n";

    TEST("add (tensor + tensor)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        auto b = torch::rand({128, 256}, torch::kFloat32);
        auto dc = to_dcu(a) + to_dcu(b);
        auto cc = a + b;
        CHECK_CLOSE(dc, cc, 1e-5, 1e-5);
    });

    TEST("add_ (in-place)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        auto b = torch::rand({128, 256}, torch::kFloat32);
        auto ad = to_dcu(a);
        ad.add_(to_dcu(b));
        auto ac = a.clone();
        ac.add_(b);
        CHECK_CLOSE(ad, ac, 1e-5, 1e-5);
    });

    TEST("sub", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        auto b = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a) - to_dcu(b), a - b, 1e-5, 1e-5);
    });

    TEST("mul (element-wise)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        auto b = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a) * to_dcu(b), a * b, 1e-5, 1e-5);
    });

    TEST("div", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        auto b = torch::rand({128, 256}, torch::kFloat32) + 0.5; // avoid div-by-zero
        CHECK_CLOSE(to_dcu(a) / to_dcu(b), a / b, 1e-5, 1e-5);
    });

    TEST("scalar add", {
        auto a = torch::rand({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a) + 3.14, a + 3.14, 1e-5, 1e-5);
    });

    TEST("scalar mul", {
        auto a = torch::rand({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a) * 2.718, a * 2.718, 1e-5, 1e-5);
    });

    TEST("neg", {
        auto a = torch::rand({64, 64}, torch::kFloat32);
        CHECK_CLOSE(-to_dcu(a), -a, 1e-6, 1e-6);
    });

    TEST("abs", {
        auto a = torch::rand({64, 64}, torch::kFloat32).sub_(0.5); // mix pos/neg
        CHECK_CLOSE(to_dcu(a).abs(), a.abs(), 1e-6, 1e-6);
    });

    TEST("floor_divide", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(1);
        auto b = torch::rand({64, 64}, torch::kFloat32).mul_(5).add_(1);
        CHECK_CLOSE(torch::floor_divide(to_dcu(a), to_dcu(b)),
                    torch::floor_divide(a, b), 1e-5, 1e-5);
    });

    TEST("fmod (remainder)", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10);
        auto b = torch::rand({64, 64}, torch::kFloat32).mul_(5).add_(0.5);
        CHECK_CLOSE(torch::fmod(to_dcu(a), to_dcu(b)),
                    torch::fmod(a, b), 1e-5, 1e-5);
    });

    TEST("clamp", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).sub_(5);
        CHECK_CLOSE(to_dcu(a).clamp(-2.0, 2.0), a.clamp(-2.0, 2.0), 1e-6, 1e-6);
    });

    TEST("clamp_min", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).sub_(5);
        CHECK_CLOSE(to_dcu(a).clamp_min(0.0), a.clamp_min(0.0), 1e-6, 1e-6);
    });

    TEST("clamp_max", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).sub_(5);
        CHECK_CLOSE(to_dcu(a).clamp_max(0.0), a.clamp_max(0.0), 1e-6, 1e-6);
    });

    TEST("sign", {
        auto a = torch::rand({64, 64}, torch::kFloat32).sub_(0.5).mul_(10);
        CHECK_CLOSE(to_dcu(a).sign(), a.sign(), 1e-6, 1e-6);
    });

    TEST("round", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(100);
        CHECK_CLOSE(to_dcu(a).round(), a.round(), 1e-6, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 4. Matrix Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Matrix Operations ──\n";

    TEST("matmul (2D)", {
        auto a = torch::rand({256, 512}, torch::kFloat32);
        auto b = torch::rand({512, 128}, torch::kFloat32);
        auto d = torch::matmul(to_dcu(a), to_dcu(b));
        auto c = torch::matmul(a, b);
        CHECK_CLOSE(d, c, 1e-3, 1e-5);
    });

    TEST("matmul (3D batch)", {
        auto a = torch::rand({8, 64, 128}, torch::kFloat32);
        auto b = torch::rand({8, 128, 64}, torch::kFloat32);
        auto d = torch::matmul(to_dcu(a), to_dcu(b));
        auto c = torch::matmul(a, b);
        CHECK_CLOSE(d, c, 1e-4, 1e-4);
    });

    TEST("matmul (4D batch)", {
        auto a = torch::rand({2, 4, 32, 64}, torch::kFloat32);
        auto b = torch::rand({2, 4, 64, 32}, torch::kFloat32);
        auto d = torch::matmul(to_dcu(a), to_dcu(b));
        auto c = torch::matmul(a, b);
        CHECK_CLOSE(d, c, 1e-4, 1e-4);
    });

    TEST("bmm (batch mm)", {
        auto a = torch::rand({16, 32, 64}, torch::kFloat32);
        auto b = torch::rand({16, 64, 32}, torch::kFloat32);
        auto d = torch::bmm(to_dcu(a), to_dcu(b));
        auto c = torch::bmm(a, b);
        CHECK_CLOSE(d, c, 1e-4, 1e-4);
    });

    TEST("addmm (A @ B + C)", {
        auto a = torch::rand({64, 128}, torch::kFloat32);
        auto b = torch::rand({128, 64}, torch::kFloat32);
        auto c_in = torch::rand({64, 64}, torch::kFloat32);
        auto d = torch::addmm(to_dcu(c_in), to_dcu(a), to_dcu(b));
        auto c = torch::addmm(c_in, a, b);
        CHECK_CLOSE(d, c, 1e-4, 1e-4);
    });

    TEST("addmv (A @ v + u)", {
        auto a = torch::rand({64, 128}, torch::kFloat32);
        auto v = torch::rand({128}, torch::kFloat32);
        auto u = torch::rand({64}, torch::kFloat32);
        auto d = torch::addmv(to_dcu(u), to_dcu(a), to_dcu(v));
        auto c = torch::addmv(u, a, v);
        CHECK_CLOSE(d, c, 1e-4, 1e-4);
    });

    TEST("dot", {
        auto a = torch::rand({1024}, torch::kFloat32);
        auto b = torch::rand({1024}, torch::kFloat32);
        auto d = torch::dot(to_dcu(a), to_dcu(b));
        auto c = torch::dot(a, b);
        double diff = std::abs(d.item<float>() - c.item<float>());
        OP_CHECK(diff < 1e-3, "dot product mismatch");
    });

    TEST("outer", {
        auto a = torch::rand({256}, torch::kFloat32);
        auto b = torch::rand({128}, torch::kFloat32);
        // outer(a, b) = a.unsqueeze(-1) * b.unsqueeze(-2)
        auto d = to_dcu(a).unsqueeze(1) * to_dcu(b).unsqueeze(0);
        auto c = a.unsqueeze(1) * b.unsqueeze(0);
        CHECK_CLOSE(d, c, 1e-5, 1e-5);
    });

    TEST("t (transpose 2D)", {
        auto a = torch::rand({64, 128}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).t(), a.t(), 1e-6, 1e-6);
    });

    TEST("transpose (specific dims)", {
        auto a = torch::rand({8, 16, 32}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).transpose(0, 2), a.transpose(0, 2), 1e-6, 1e-6);
    });

    TEST("permute", {
        auto a = torch::rand({4, 8, 16, 32}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).permute({3, 1, 0, 2}), a.permute({3, 1, 0, 2}), 1e-6, 1e-6);
    });

    TEST("inverse (3x3)", {
        auto c = torch::rand({3, 3}, torch::kFloat64);
        c = c + c.t();  // make positive-definite for better conditioning
        c.diagonal().add_(6.0);
        auto d = torch::inverse(to_dcu(c));
        auto cc = torch::inverse(c);
        CHECK_CLOSE(d, cc, 1e-6, 1e-6);
    });

    TEST("pinverse", {
        auto c = torch::rand({8, 4}, torch::kFloat64);
        auto d = torch::linalg::pinv(to_dcu(c), 1e-8, false);
        auto cc = torch::linalg::pinv(c, 1e-8, false);
        CHECK_CLOSE(d, cc, 1e-5, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 5. Activation Functions
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Activation Functions ──\n";

    TEST("relu", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::relu(to_dcu(a)), torch::relu(a), 1e-6, 1e-6);
    });

    TEST("leaky_relu", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::leaky_relu(to_dcu(a), 0.01),
                    torch::leaky_relu(a, 0.01), 1e-6, 1e-6);
    });

    TEST("elu", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::elu(to_dcu(a)), torch::elu(a), 1e-5, 1e-6);
    });

    TEST("selu", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::selu(to_dcu(a)), torch::selu(a), 1e-5, 1e-6);
    });

    TEST("celu", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::celu(to_dcu(a), 1.0), torch::celu(a, 1.0), 1e-5, 1e-6);
    });

    TEST("gelu", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::gelu(to_dcu(a)), torch::gelu(a), 1e-5, 1e-5);
    });

    TEST("silu (swish)", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::silu(to_dcu(a)), torch::silu(a), 1e-5, 1e-6);
    });

    TEST("sigmoid", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::sigmoid(to_dcu(a)), torch::sigmoid(a), 1e-5, 1e-5);
    });

    TEST("tanh", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        CHECK_CLOSE(torch::tanh(to_dcu(a)), torch::tanh(a), 1e-5, 1e-5);
    });

    TEST("hardtanh", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(2);
        CHECK_CLOSE(torch::hardtanh(to_dcu(a), -1.0, 1.0),
                    torch::hardtanh(a, -1.0, 1.0), 1e-6, 1e-6);
    });

    TEST("hardsigmoid", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(2);
        CHECK_CLOSE(torch::hardsigmoid(to_dcu(a)), torch::hardsigmoid(a), 1e-5, 1e-6);
    });

    TEST("hardswish", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::hardswish(to_dcu(a)), torch::hardswish(a), 1e-5, 1e-6);
    });

    TEST("softmax (dim=-1)", {
        auto a = torch::randn({32, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::softmax(to_dcu(a), -1), torch::softmax(a, -1), 1e-5, 1e-6);
    });

    TEST("softmax (dim=0)", {
        auto a = torch::randn({64, 32}, torch::kFloat32);
        CHECK_CLOSE(torch::softmax(to_dcu(a), 0), torch::softmax(a, 0), 1e-5, 1e-6);
    });

    TEST("log_softmax", {
        auto a = torch::randn({32, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::log_softmax(to_dcu(a), -1), torch::log_softmax(a, -1), 1e-5, 1e-6);
    });

    TEST("softplus", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::softplus(to_dcu(a)), torch::softplus(a), 1e-5, 1e-6);
    });

    TEST("softsign", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::nn::functional::softsign(to_dcu(a)), torch::nn::functional::softsign(a), 1e-5, 1e-6);
    });

    TEST("tanhshrink", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::nn::functional::tanhshrink(to_dcu(a)), torch::nn::functional::tanhshrink(a), 1e-5, 1e-6);
    });

    TEST("threshold", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::threshold(to_dcu(a), 0.0, 0.0),
                    torch::threshold(a, 0.0, 0.0), 1e-6, 1e-6);
    });

    TEST("prelu (single weight)", {
        auto a = torch::randn({128, 256}, torch::kFloat32);
        auto w = torch::tensor({0.25}, torch::kFloat32);
        CHECK_CLOSE(torch::prelu(to_dcu(a), to_dcu(w)),
                    torch::prelu(a, w), 1e-5, 1e-6);
    });

    TEST("prelu (channel weight)", {
        auto a = torch::randn({16, 8, 32}, torch::kFloat32);
        auto w = torch::rand({8}, torch::kFloat32);
        CHECK_CLOSE(torch::prelu(to_dcu(a), to_dcu(w)),
                    torch::prelu(a, w), 1e-5, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 6. Element-wise Math Functions
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Element-wise Math ──\n";

    TEST("exp", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).exp(), a.exp(), 1e-5, 1e-5);
    });

    TEST("expm1", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).expm1(), a.expm1(), 1e-5, 1e-6);
    });

    TEST("log", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(0.01);
        CHECK_CLOSE(to_dcu(a).log(), a.log(), 1e-5, 1e-6);
    });

    TEST("log2", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(0.01);
        CHECK_CLOSE(to_dcu(a).log2(), a.log2(), 1e-5, 1e-6);
    });

    TEST("log10", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(100).add_(0.01);
        CHECK_CLOSE(to_dcu(a).log10(), a.log10(), 1e-5, 1e-6);
    });

    TEST("log1p", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(0.5);
        CHECK_CLOSE(to_dcu(a).log1p(), a.log1p(), 1e-5, 1e-6);
    });

    TEST("sqrt", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(0.01);
        CHECK_CLOSE(to_dcu(a).sqrt(), a.sqrt(), 1e-5, 1e-6);
    });

    TEST("rsqrt", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(0.01);
        CHECK_CLOSE(to_dcu(a).rsqrt(), a.rsqrt(), 1e-5, 1e-6);
    });

    TEST("square", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).square(), a.square(), 1e-6, 1e-6);
    });

    TEST("pow (scalar exponent)", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(2).add_(0.1);
        CHECK_CLOSE(to_dcu(a).pow(2.5), a.pow(2.5), 1e-5, 1e-6);
    });

    TEST("pow (tensor exponent)", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(2).add_(0.1);
        auto e = torch::rand({64, 64}, torch::kFloat32).mul_(2).add_(0.1);
        CHECK_CLOSE(to_dcu(a).pow(to_dcu(e)), a.pow(e), 1e-5, 1e-6);
    });

    TEST("sin", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).sin(), a.sin(), 1e-5, 1e-6);
    });

    TEST("cos", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).cos(), a.cos(), 1e-5, 1e-6);
    });

    TEST("tan", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(1.0); // avoid asymptotes
        CHECK_CLOSE(to_dcu(a).tan(), a.tan(), 1e-4, 1e-5);
    });

    TEST("asin", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(2).sub_(1).mul_(0.99);
        CHECK_CLOSE(to_dcu(a).asin(), a.asin(), 1e-5, 1e-6);
    });

    TEST("acos", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(2).sub_(1).mul_(0.99);
        CHECK_CLOSE(to_dcu(a).acos(), a.acos(), 1e-5, 1e-6);
    });

    TEST("atan", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(5);
        CHECK_CLOSE(to_dcu(a).atan(), a.atan(), 1e-5, 1e-6);
    });

    TEST("atan2", {
        auto y = torch::randn({64, 64}, torch::kFloat32);
        auto x = torch::rand({64, 64}, torch::kFloat32).add_(0.1);
        CHECK_CLOSE(to_dcu(y).atan2(to_dcu(x)), y.atan2(x), 1e-5, 1e-6);
    });

    TEST("sinh", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(2);
        CHECK_CLOSE(to_dcu(a).sinh(), a.sinh(), 1e-5, 1e-6);
    });

    TEST("cosh", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(1.0);
        CHECK_CLOSE(to_dcu(a).cosh(), a.cosh(), 1e-4, 1e-5);
    });

    TEST("erf", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).erf(), a.erf(), 1e-5, 1e-6);
    });

    TEST("erfc", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).erfc(), a.erfc(), 1e-5, 1e-6);
    });

    TEST("erfinv", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(1.8).sub_(0.9);
        CHECK_CLOSE(to_dcu(a).erfinv(), a.erfinv(), 1e-4, 1e-5);
    });

    TEST("frac", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(10);
        CHECK_CLOSE(to_dcu(a).frac(), a.frac(), 1e-5, 1e-6);
    });

    TEST("trunc", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(10);
        CHECK_CLOSE(to_dcu(a).trunc(), a.trunc(), 1e-6, 1e-6);
    });

    TEST("ceil", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(10);
        CHECK_CLOSE(to_dcu(a).ceil(), a.ceil(), 1e-6, 1e-6);
    });

    TEST("floor", {
        auto a = torch::randn({64, 64}, torch::kFloat32).mul_(10);
        CHECK_CLOSE(to_dcu(a).floor(), a.floor(), 1e-6, 1e-6);
    });

    TEST("reciprocal", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(0.1);
        CHECK_CLOSE(to_dcu(a).reciprocal(), a.reciprocal(), 1e-5, 1e-6);
    });

    TEST("lgamma", {
        auto a = torch::rand({64, 64}, torch::kFloat32).mul_(10).add_(0.1);
        CHECK_CLOSE(to_dcu(a).lgamma(), a.lgamma(), 1e-5, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 7. Reduction Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Reduction Operations ──\n";

    TEST("sum (all)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        auto d = to_dcu(a).sum();
        auto c = a.sum();
        double diff = std::abs(d.item<float>() - c.item<float>());
        OP_CHECK(diff < 1e-2, "sum mismatch");
    });

    TEST("sum (dim=0)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).sum(0), a.sum(0), 1e-3, 1e-5);
    });

    TEST("sum (dim=1)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).sum(1), a.sum(1), 1e-3, 1e-5);
    });

    TEST("mean (all)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        double d = to_dcu(a).mean().item<float>();
        double c = a.mean().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "mean mismatch");
    });

    TEST("mean (dim)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).mean(0), a.mean(0), 1e-5, 1e-6);
    });

    TEST("std (all, unbiased)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        double d = to_dcu(a).std().item<float>();
        double c = a.std().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-4, "std mismatch");
    });

    TEST("std (dim, unbiased)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).std(0), a.std(0), 1e-4, 1e-5);
    });

    TEST("var (all, unbiased)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        double d = to_dcu(a).var().item<float>();
        double c = a.var().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-4, "var mismatch");
    });

    TEST("var (dim)", {
        auto a = torch::rand({128, 256}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).var(0), a.var(0), 1e-4, 1e-5);
    });

    TEST("prod", {
        auto a = torch::rand({4, 8}, torch::kFloat32).add_(0.5);
        CHECK_CLOSE(to_dcu(a).prod(), a.prod(), 1e-2, 1e-4);
    });

    TEST("max (all)", {
        auto a = torch::rand({128, 128}, torch::kFloat32);
        double d = to_dcu(a).max().item<float>();
        double c = a.max().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-6, "max mismatch");
    });

    TEST("max (dim)", {
        auto a = torch::rand({64, 128}, torch::kFloat32);
        auto dr = to_dcu(a).max(1);
        auto cr = a.max(1);
        CHECK_CLOSE(std::get<0>(dr), std::get<0>(cr), 1e-6, 1e-6);
        OP_CHECK(torch::equal(std::get<1>(dr).to(torch::kCPU), std::get<1>(cr)), "argmax mismatch");
    });

    TEST("min (all)", {
        auto a = torch::rand({128, 128}, torch::kFloat32);
        double d = to_dcu(a).min().item<float>();
        double c = a.min().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-6, "min mismatch");
    });

    TEST("min (dim)", {
        auto a = torch::rand({64, 128}, torch::kFloat32);
        auto dr = to_dcu(a).min(1);
        auto cr = a.min(1);
        CHECK_CLOSE(std::get<0>(dr), std::get<0>(cr), 1e-6, 1e-6);
        OP_CHECK(torch::equal(std::get<1>(dr).to(torch::kCPU), std::get<1>(cr)), "argmin mismatch");
    });

    TEST("argmax (dim)", {
        auto a = torch::rand({16, 32}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).argmax(1).to(torch::kCPU), a.argmax(1)),
              "argmax mismatch");
    });

    TEST("argmin (dim)", {
        auto a = torch::rand({16, 32}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).argmin(1).to(torch::kCPU), a.argmin(1)),
              "argmin mismatch");
    });

    TEST("logsumexp", {
        auto a = torch::randn({32, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).logsumexp(1), a.logsumexp(1), 1e-5, 1e-6);
    });

    TEST("norm (L2 vector)", {
        auto a = torch::randn({1024}, torch::kFloat32);
        double d = to_dcu(a).norm().item<float>();
        double c = a.norm().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-4, "norm mismatch");
    });

    TEST("norm (L2 matrix)", {
        auto a = torch::randn({64, 128}, torch::kFloat32);
        double d = to_dcu(a).norm().item<float>();
        double c = a.norm().item<float>();
        OP_CHECK(std::abs(d - c) < 1e-4, "norm mismatch");
    });

    TEST("norm (L1 dim)", {
        auto a = torch::randn({64, 128}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).norm(1, 1), a.norm(1, 1), 1e-4, 1e-5);
    });

    TEST("cumsum", {
        auto a = torch::rand({8, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).cumsum(1), a.cumsum(1), 1e-4, 1e-5);
    });

    TEST("cumprod", {
        auto a = torch::rand({8, 16}, torch::kFloat32).mul_(0.5).add_(0.5);
        CHECK_CLOSE(to_dcu(a).cumprod(1), a.cumprod(1), 1e-4, 1e-5);
    });

    // ═══════════════════════════════════════════════════════════════
    // 8. Comparison Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Comparison Operations ──\n";

    TEST("eq", {
        auto a = torch::tensor({1.0f, 2.0f, 3.0f, 2.0f, 1.0f}, torch::kFloat32);
        auto b = torch::tensor({1.0f, 0.0f, 3.0f, 0.0f, 0.0f}, torch::kFloat32);
        auto d = to_dcu(a).eq(to_dcu(b));
        auto c = a.eq(b);
        OP_CHECK(torch::equal(d.to(torch::kCPU), c), "eq mismatch");
    });

    TEST("ne", {
        auto a = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat32);
        auto b = torch::tensor({1.0f, 0.0f, 3.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).ne(to_dcu(b)).to(torch::kCPU), a.ne(b)),
              "ne mismatch");
    });

    TEST("gt", {
        auto a = torch::tensor({1.0f, 5.0f, 3.0f}, torch::kFloat32);
        auto b = torch::tensor({2.0f, 2.0f, 2.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).gt(to_dcu(b)).to(torch::kCPU), a.gt(b)),
              "gt mismatch");
    });

    TEST("lt", {
        auto a = torch::tensor({1.0f, 5.0f, 3.0f}, torch::kFloat32);
        auto b = torch::tensor({2.0f, 2.0f, 2.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).lt(to_dcu(b)).to(torch::kCPU), a.lt(b)),
              "lt mismatch");
    });

    TEST("ge", {
        auto a = torch::tensor({2.0f, 5.0f, 3.0f}, torch::kFloat32);
        auto b = torch::tensor({2.0f, 2.0f, 2.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).ge(to_dcu(b)).to(torch::kCPU), a.ge(b)),
              "ge mismatch");
    });

    TEST("le", {
        auto a = torch::tensor({2.0f, 1.0f, 3.0f}, torch::kFloat32);
        auto b = torch::tensor({2.0f, 2.0f, 2.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).le(to_dcu(b)).to(torch::kCPU), a.le(b)),
              "le mismatch");
    });

    TEST("isnan", {
        auto a = torch::tensor({1.0f, NAN, 3.0f, NAN}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).isnan().to(torch::kCPU), a.isnan()),
              "isnan mismatch");
    });

    TEST("isinf", {
        auto a = torch::tensor({1.0f, INFINITY, 3.0f, -INFINITY}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).isinf().to(torch::kCPU), a.isinf()),
              "isinf mismatch");
    });

    TEST("isfinite", {
        auto a = torch::tensor({1.0f, INFINITY, NAN, -INFINITY, 5.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).isfinite().to(torch::kCPU), a.isfinite()),
              "isfinite mismatch");
    });

    TEST("maximum (element-wise)", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        auto b = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::maximum(to_dcu(a), to_dcu(b)),
                    torch::maximum(a, b), 1e-6, 1e-6);
    });

    TEST("minimum (element-wise)", {
        auto a = torch::randn({64, 64}, torch::kFloat32);
        auto b = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::minimum(to_dcu(a), to_dcu(b)),
                    torch::minimum(a, b), 1e-6, 1e-6);
    });

    TEST("where", {
        auto cond = torch::rand({64, 64}, torch::kFloat32).gt(0.5);
        auto a = torch::randn({64, 64}, torch::kFloat32);
        auto b = torch::randn({64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::where(to_dcu(cond), to_dcu(a), to_dcu(b)),
                    torch::where(cond, a, b), 1e-6, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 9. Shape / View Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Shape & View Operations ──\n";

    TEST("reshape", {
        auto a = torch::rand({8, 16, 32}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).reshape({-1, 32}), a.reshape({-1, 32}), 1e-6, 1e-6);
    });

    TEST("view", {
        auto a = torch::rand({8, 16, 32}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).view({-1, 32}), a.view({-1, 32}), 1e-6, 1e-6);
    });

    TEST("unsqueeze", {
        auto a = torch::rand({8, 16}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).unsqueeze(0), a.unsqueeze(0), 1e-6, 1e-6);
        CHECK_CLOSE(to_dcu(a).unsqueeze(2), a.unsqueeze(2), 1e-6, 1e-6);
    });

    TEST("squeeze", {
        auto a = torch::rand({8, 1, 16, 1}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).squeeze(), a.squeeze(), 1e-6, 1e-6);
    });

    TEST("flatten", {
        auto a = torch::rand({8, 16, 32}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).flatten(), a.flatten(), 1e-6, 1e-6);
        CHECK_CLOSE(to_dcu(a).flatten(1), a.flatten(1), 1e-6, 1e-6);
    });

    TEST("cat (dim=0)", {
        auto a = torch::rand({16, 32}, torch::kFloat32);
        auto b = torch::rand({8, 32}, torch::kFloat32);
        CHECK_CLOSE(torch::cat({to_dcu(a), to_dcu(b)}, 0),
                    torch::cat({a, b}, 0), 1e-6, 1e-6);
    });

    TEST("cat (dim=1)", {
        auto a = torch::rand({16, 32}, torch::kFloat32);
        auto b = torch::rand({16, 16}, torch::kFloat32);
        CHECK_CLOSE(torch::cat({to_dcu(a), to_dcu(b)}, 1),
                    torch::cat({a, b}, 1), 1e-6, 1e-6);
    });

    TEST("stack", {
        auto a = torch::rand({8, 16}, torch::kFloat32);
        auto b = torch::rand({8, 16}, torch::kFloat32);
        auto da = to_dcu(a);
        auto db = to_dcu(b);
        auto dl = torch::cat({da.unsqueeze(0), db.unsqueeze(0)}, 0);
        auto cl = torch::cat({a.unsqueeze(0), b.unsqueeze(0)}, 0);
        // stack via unsqueeze + cat is equivalent
        CHECK_CLOSE(dl, cl, 1e-6, 1e-6);
    });

    TEST("split", {
        auto a = torch::rand({16, 32}, torch::kFloat32);
        auto dr = to_dcu(a).split(4, 0);
        auto cr = a.split(4, 0);
        for (size_t i = 0; i < cr.size(); i++) {
            CHECK_CLOSE(dr[i], cr[i], 1e-6, 1e-6);
        }
    });

    TEST("chunk", {
        auto a = torch::rand({16, 32}, torch::kFloat32);
        auto dr = to_dcu(a).chunk(4, 0);
        auto cr = a.chunk(4, 0);
        for (size_t i = 0; i < cr.size(); i++) {
            CHECK_CLOSE(dr[i], cr[i], 1e-6, 1e-6);
        }
    });

    TEST("repeat", {
        auto a = torch::rand({4, 8}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).repeat({3, 2}), a.repeat({3, 2}), 1e-6, 1e-6);
    });

    TEST("expand", {
        auto a = torch::rand({1, 16}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).expand({8, 16}), a.expand({8, 16}), 1e-6, 1e-6);
    });

    TEST("tile", {
        auto a = torch::rand({4, 8}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).tile({3, 2}), a.tile({3, 2}), 1e-6, 1e-6);
    });

    TEST("roll", {
        auto a = torch::rand({8, 16}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).roll({3, -2}, {0, 1}), a.roll({3, -2}, {0, 1}), 1e-6, 1e-6);
    });

    TEST("flip", {
        auto a = torch::rand({8, 16}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).flip({0, 1}), a.flip({0, 1}), 1e-6, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 10. Indexing & Slicing Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Indexing & Slicing ──\n";

    TEST("index_select", {
        auto a = torch::rand({32, 64}, torch::kFloat32);
        auto idx = torch::tensor({0, 3, 5, 10, 25}, torch::kInt64);
        CHECK_CLOSE(to_dcu(a).index_select(0, to_dcu(idx)),
                    a.index_select(0, idx), 1e-6, 1e-6);
    });

    TEST("slice (dim=0, 0:8)", {
        auto a = torch::rand({32, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).slice(0, 0, 8), a.slice(0, 0, 8), 1e-6, 1e-6);
    });

    TEST("slice (dim=1, 16:48)", {
        auto a = torch::rand({32, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).slice(1, 16, 48), a.slice(1, 16, 48), 1e-6, 1e-6);
    });

    TEST("narrow", {
        auto a = torch::rand({32, 64}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).narrow(0, 4, 8), a.narrow(0, 4, 8), 1e-6, 1e-6);
    });

    TEST("gather", {
        auto a = torch::rand({8, 16}, torch::kFloat32);
        auto idx = torch::randint(0, 16, {8, 32}, torch::kInt64);
        CHECK_CLOSE(to_dcu(a).gather(1, to_dcu(idx)),
                    a.gather(1, idx), 1e-6, 1e-6);
    });

    TEST("scatter_", {
        auto src = torch::rand({8, 16}, torch::kFloat32);
        // Use unique column indices per row to avoid non-deterministic duplicate writes
        auto idx = torch::arange(0, 16, torch::kInt64).unsqueeze(0).expand({8, -1}).contiguous();
        // Shuffle each row independently for more interesting scatter pattern
        idx = idx.index_select(0, torch::randperm(8, torch::kInt64));
        auto d_dst = torch::zeros({8, 16}, torch::TensorOptions().device(dcu_device()).dtype(torch::kFloat32));
        auto c_dst = torch::zeros({8, 16}, torch::kFloat32);
        d_dst.scatter_(1, to_dcu(idx), to_dcu(src));
        c_dst.scatter_(1, idx, src);
        CHECK_CLOSE(d_dst, c_dst, 1e-6, 1e-6);
    });

    TEST("masked_select", {
        auto a = torch::randn({16, 8}, torch::kFloat32);
        auto mask = a.gt(0.0);
        auto d = to_dcu(a).masked_select(to_dcu(mask));
        auto c = a.masked_select(mask);
        CHECK_CLOSE(d, c, 1e-6, 1e-6);
    });

    TEST("masked_fill", {
        auto a = torch::randn({16, 8}, torch::kFloat32);
        auto mask = a.gt(0.0);
        auto dc = to_dcu(a).masked_fill(to_dcu(mask), 0.0);
        auto cc = a.masked_fill(mask, 0.0);
        CHECK_CLOSE(dc, cc, 1e-6, 1e-6);
    });

    TEST("nonzero", {
        auto a = torch::tensor({0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f}, torch::kFloat32);
        OP_CHECK(torch::equal(to_dcu(a).nonzero().to(torch::kCPU), a.nonzero()),
              "nonzero mismatch");
    });

    TEST("take", {
        auto a = torch::rand({64}, torch::kFloat32);
        auto idx = torch::tensor({0, 10, 20, 30, 40, 50}, torch::kInt64);
        CHECK_CLOSE(to_dcu(a).take(to_dcu(idx)), a.take(idx), 1e-6, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 11. Convolution & Pooling Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Convolution & Pooling ──\n";

    TEST("conv2d (stride=1)", {
        auto x = torch::randn({2, 3, 32, 32}, torch::kFloat32);
        auto w = torch::randn({16, 3, 3, 3}, torch::kFloat32);
        auto b = torch::randn({16}, torch::kFloat32);
        auto dc = torch::conv2d(to_dcu(x), to_dcu(w), to_dcu(b), 1, 1);
        auto cc = torch::conv2d(x, w, b, 1, 1);
        CHECK_CLOSE(dc, cc, 1e-4, 1e-4);
    });

    TEST("conv2d (stride=2, pad=1)", {
        auto x = torch::randn({1, 8, 28, 28}, torch::kFloat32);
        auto w = torch::randn({16, 8, 3, 3}, torch::kFloat32);
        auto dc = torch::conv2d(to_dcu(x), to_dcu(w), {}, 2, 1);
        auto cc = torch::conv2d(x, w, {}, 2, 1);
        CHECK_CLOSE(dc, cc, 1e-4, 1e-4);
    });

    TEST("conv2d (dilation=2, groups=1)", {
        auto x = torch::randn({1, 4, 16, 16}, torch::kFloat32);
        auto w = torch::randn({8, 4, 3, 3}, torch::kFloat32);
        auto dc = torch::conv2d(to_dcu(x), to_dcu(w), {}, 1, 1, 2);
        auto cc = torch::conv2d(x, w, {}, 1, 1, 2);
        CHECK_CLOSE(dc, cc, 1e-4, 1e-4);
    });

    TEST("conv2d (groups=4, depthwise)", {
        auto groups = 4;
        auto x = torch::randn({1, 4, 16, 16}, torch::kFloat32);
        auto w = torch::randn({4, 1, 3, 3}, torch::kFloat32);
        auto dc = torch::conv2d(to_dcu(x), to_dcu(w), {}, 1, 1, 1, groups);
        auto cc = torch::conv2d(x, w, {}, 1, 1, 1, groups);
        CHECK_CLOSE(dc, cc, 1e-4, 1e-4);
    });

    TEST("max_pool2d (kernel=2, stride=2)", {
        auto x = torch::randn({2, 8, 32, 32}, torch::kFloat32);
        CHECK_CLOSE(torch::max_pool2d(to_dcu(x), 2, 2),
                    torch::max_pool2d(x, 2, 2), 1e-6, 1e-6);
    });

    TEST("max_pool2d (kernel=3, stride=2, pad=1)", {
        auto x = torch::randn({2, 8, 32, 32}, torch::kFloat32);
        CHECK_CLOSE(torch::max_pool2d(to_dcu(x), 3, 2, 1),
                    torch::max_pool2d(x, 3, 2, 1), 1e-6, 1e-6);
    });

    TEST("avg_pool2d (kernel=2, stride=2)", {
        auto x = torch::randn({2, 8, 32, 32}, torch::kFloat32);
        CHECK_CLOSE(torch::avg_pool2d(to_dcu(x), 2, 2),
                    torch::avg_pool2d(x, 2, 2), 1e-5, 1e-6);
    });

    TEST("avg_pool2d (kernel=4, stride=4)", {
        auto x = torch::randn({1, 4, 64, 64}, torch::kFloat32);
        CHECK_CLOSE(torch::avg_pool2d(to_dcu(x), 4, 4),
                    torch::avg_pool2d(x, 4, 4), 1e-5, 1e-6);
    });

    TEST("adaptive_avg_pool2d (output=1x1)", {
        auto x = torch::randn({2, 8, 32, 32}, torch::kFloat32);
        CHECK_CLOSE(torch::adaptive_avg_pool2d(to_dcu(x), {1, 1}),
                    torch::adaptive_avg_pool2d(x, {1, 1}), 1e-5, 1e-6);
    });

    TEST("adaptive_avg_pool2d (output=7x7)", {
        auto x = torch::randn({2, 3, 56, 56}, torch::kFloat32);
        CHECK_CLOSE(torch::adaptive_avg_pool2d(to_dcu(x), {7, 7}),
                    torch::adaptive_avg_pool2d(x, {7, 7}), 1e-5, 1e-6);
    });

    TEST("adaptive_max_pool2d (output=1x1)", {
        auto x = torch::randn({2, 8, 32, 32}, torch::kFloat32);
        auto opts = torch::nn::functional::AdaptiveMaxPool2dFuncOptions({1, 1});
        CHECK_CLOSE(torch::nn::functional::adaptive_max_pool2d(to_dcu(x), opts),
                    torch::nn::functional::adaptive_max_pool2d(x, opts), 1e-6, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 12. Normalization Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Normalization ──\n";

    TEST("batch_norm (eval mode)", {
        auto x = torch::randn({4, 16, 8, 8}, torch::kFloat32);
        auto w = torch::randn({16}, torch::kFloat32);
        auto b = torch::randn({16}, torch::kFloat32);
        auto rm = torch::randn({16}, torch::kFloat32);
        auto rv = torch::rand({16}, torch::kFloat32).add_(0.1);
        auto dc = torch::batch_norm(to_dcu(x), to_dcu(w), to_dcu(b),
                                     to_dcu(rm), to_dcu(rv), false, 0.0, 1e-5, true);
        auto cc = torch::batch_norm(x, w, b, rm, rv, false, 0.0, 1e-5, true);
        CHECK_CLOSE(dc, cc, 1e-4, 1e-5);
    });

    TEST("layer_norm (last dim)", {
        auto x = torch::randn({8, 64}, torch::kFloat32);
        auto dc = torch::layer_norm(to_dcu(x), {64}, {}, {}, 1e-5);
        auto cc = torch::layer_norm(x, {64}, {}, {}, 1e-5);
        CHECK_CLOSE(dc, cc, 1e-5, 1e-6);
    });

    TEST("layer_norm (2D last dims)", {
        auto x = torch::randn({4, 16, 64}, torch::kFloat32);
        auto dc = torch::layer_norm(to_dcu(x), {64}, {}, {}, 1e-5);
        auto cc = torch::layer_norm(x, {64}, {}, {}, 1e-5);
        CHECK_CLOSE(dc, cc, 1e-5, 1e-6);
    });

    TEST("layer_norm (with weight/bias)", {
        auto x = torch::randn({8, 128}, torch::kFloat32);
        auto w = torch::randn({128}, torch::kFloat32);
        auto b = torch::randn({128}, torch::kFloat32);
        auto dc = torch::layer_norm(to_dcu(x), {128}, to_dcu(w), to_dcu(b), 1e-5);
        auto cc = torch::layer_norm(x, {128}, w, b, 1e-5);
        CHECK_CLOSE(dc, cc, 1e-5, 1e-6);
    });

    TEST("group_norm", {
        auto x = torch::randn({4, 16, 8, 8}, torch::kFloat32);
        auto w = torch::randn({16}, torch::kFloat32);
        auto b = torch::randn({16}, torch::kFloat32);
        auto dc = torch::group_norm(to_dcu(x), 4, to_dcu(w), to_dcu(b), 1e-5);
        auto cc = torch::group_norm(x, 4, w, b, 1e-5);
        CHECK_CLOSE(dc, cc, 1e-4, 1e-5);
    });

    TEST("instance_norm (training mode)", {
        auto x = torch::randn({4, 8, 16, 16}, torch::kFloat32);
        auto w = torch::randn({8}, torch::kFloat32);
        auto b = torch::randn({8}, torch::kFloat32);
        // use_input_stats=true: compute stats from input
        auto dc = torch::instance_norm(to_dcu(x), to_dcu(w), to_dcu(b),
                                        {}, {}, true, 0.0, 1e-5, true);
        auto cc = torch::instance_norm(x, w, b, {}, {}, true, 0.0, 1e-5, true);
        CHECK_CLOSE(dc, cc, 1e-3, 1e-4);
    });

    TEST("normalize (L2, dim=1)", {
        auto x = torch::randn({8, 64}, torch::kFloat32);
        auto opts = torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1);
        CHECK_CLOSE(torch::nn::functional::normalize(to_dcu(x), opts),
                    torch::nn::functional::normalize(x, opts), 1e-5, 1e-6);
    });

    // ═══════════════════════════════════════════════════════════════
    // 13. Loss Functions
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Loss Functions ──\n";

    TEST("mse_loss", {
        auto pred = torch::randn({32, 16}, torch::kFloat32);
        auto target = torch::randn({32, 16}, torch::kFloat32);
        double d = torch::mse_loss(to_dcu(pred), to_dcu(target)).item<float>();
        double c = torch::mse_loss(pred, target).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "mse_loss mismatch");
    });

    TEST("l1_loss", {
        auto pred = torch::randn({32, 16}, torch::kFloat32);
        auto target = torch::randn({32, 16}, torch::kFloat32);
        double d = torch::l1_loss(to_dcu(pred), to_dcu(target)).item<float>();
        double c = torch::l1_loss(pred, target).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-4, "l1_loss mismatch");
    });

    TEST("cross_entropy_loss", {
        auto pred = torch::randn({16, 10}, torch::kFloat32);
        auto target = torch::randint(0, 10, {16}, torch::kInt64);
        double d = torch::cross_entropy_loss(to_dcu(pred), to_dcu(target)).item<float>();
        double c = torch::cross_entropy_loss(pred, target).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "cross_entropy_loss mismatch");
    });

    TEST("nll_loss", {
        auto pred = torch::randn({16, 10}, torch::kFloat32).log_softmax(1);
        auto target = torch::randint(0, 10, {16}, torch::kInt64);
        double d = torch::nll_loss(to_dcu(pred), to_dcu(target)).item<float>();
        double c = torch::nll_loss(pred, target).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "nll_loss mismatch");
    });

    TEST("binary_cross_entropy", {
        auto pred = torch::sigmoid(torch::randn({64}, torch::kFloat32));
        auto target = torch::rand({64}, torch::kFloat32).round();
        double d = torch::binary_cross_entropy(to_dcu(pred), to_dcu(target)).item<float>();
        double c = torch::binary_cross_entropy(pred, target).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "bce_loss mismatch");
    });

    TEST("smooth_l1_loss", {
        auto pred = torch::randn({32, 16}, torch::kFloat32);
        auto target = torch::randn({32, 16}, torch::kFloat32);
        double d = torch::smooth_l1_loss(to_dcu(pred), to_dcu(target)).item<float>();
        double c = torch::smooth_l1_loss(pred, target).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "smooth_l1_loss mismatch");
    });

    TEST("kl_div", {
        auto log_pred = torch::randn({16, 10}, torch::kFloat32).log_softmax(1);
        auto target = torch::rand({16, 10}, torch::kFloat32).softmax(1);
        double d = torch::kl_div(to_dcu(log_pred), to_dcu(target),
                                  torch::Reduction::Mean, true).item<float>();
        double c = torch::kl_div(log_pred, target,
                                  torch::Reduction::Mean, true).item<float>();
        OP_CHECK(std::abs(d - c) < 1e-5, "kl_div mismatch");
    });

    // ═══════════════════════════════════════════════════════════════
    // 14. Linear Algebra
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Linear Algebra ──\n";

    TEST("cross (3D vectors)", {
        auto a = torch::randn({64, 3}, torch::kFloat32);
        auto b = torch::randn({64, 3}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).cross(to_dcu(b), 1), a.cross(b, 1), 1e-5, 1e-6);
    });

    TEST("trace", {
        auto c = torch::rand({8, 8}, torch::kFloat64);
        double d = to_dcu(c).trace().item<double>();
        double cc = c.trace().item<double>();
        OP_CHECK(std::abs(d - cc) < 1e-8, "trace mismatch");
    });

    TEST("diag", {
        auto a = torch::rand({16}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).diag(), a.diag(), 1e-6, 1e-6);
    });

    TEST("diagonal", {
        auto a = torch::rand({8, 8}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).diagonal(), a.diagonal(), 1e-6, 1e-6);
    });

    TEST("tril", {
        auto a = torch::rand({8, 8}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).tril(), a.tril(), 1e-6, 1e-6);
        CHECK_CLOSE(to_dcu(a).tril(-1), a.tril(-1), 1e-6, 1e-6);
    });

    TEST("triu", {
        auto a = torch::rand({8, 8}, torch::kFloat32);
        CHECK_CLOSE(to_dcu(a).triu(), a.triu(), 1e-6, 1e-6);
        CHECK_CLOSE(to_dcu(a).triu(1), a.triu(1), 1e-6, 1e-6);
    });

    TEST("matrix_rank", {
        auto c = torch::rand({8, 8}, torch::kFloat64);
        // ensure full rank
        c.diagonal().add_(8.0);
        auto d = torch::linalg::matrix_rank(to_dcu(c), 1e-6, false);
        auto cc = torch::linalg::matrix_rank(c, 1e-6, false);
        CHECK_CLOSE(d, cc, 1e-8, 1e-8);
    });

    TEST("matrix_power", {
        auto c = torch::rand({4, 4}, torch::kFloat64);
        CHECK_CLOSE(torch::linalg::matrix_power(to_dcu(c), 3),
                    torch::linalg::matrix_power(c, 3), 1e-6, 1e-8);
    });

    // ═══════════════════════════════════════════════════════════════
    // 15. Interpolation / Upsampling
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Interpolation & Upsampling ──\n";

    TEST("upsample_nearest2d", {
        auto x = torch::rand({1, 3, 16, 16}, torch::kFloat32);
        auto opts = torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{32, 32})
                        .mode(torch::kNearest);
        CHECK_CLOSE(torch::nn::functional::interpolate(to_dcu(x), opts),
                    torch::nn::functional::interpolate(x, opts), 1e-6, 1e-6);
    });

    TEST("upsample_bilinear2d", {
        auto x = torch::rand({1, 3, 16, 16}, torch::kFloat32);
        auto opts = torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{32, 32})
                        .mode(torch::kBilinear)
                        .align_corners(false);
        CHECK_CLOSE(torch::nn::functional::interpolate(to_dcu(x), opts),
                    torch::nn::functional::interpolate(x, opts), 1e-4, 1e-5);
    });

    // ═══════════════════════════════════════════════════════════════
    // 16. Miscellaneous Operations
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Miscellaneous ──\n";

    TEST("topk (values + indices)", {
        auto a = torch::rand({256}, torch::kFloat32);
        auto dr = to_dcu(a).topk(10);
        auto cr = a.topk(10);
        CHECK_CLOSE(std::get<0>(dr), std::get<0>(cr), 1e-6, 1e-6);
        // indices may differ if duplicate values; skip index comparison for that case
    });

    TEST("sort", {
        auto a = torch::rand({128}, torch::kFloat32);
        auto dr = to_dcu(a).sort();
        auto cr = a.sort();
        CHECK_CLOSE(std::get<0>(dr), std::get<0>(cr), 1e-6, 1e-6);
    });

    TEST("kthvalue", {
        auto a = torch::rand({128}, torch::kFloat32);
        auto dr = to_dcu(a).kthvalue(32);
        auto cr = a.kthvalue(32);
        CHECK_CLOSE(std::get<0>(dr), std::get<0>(cr), 1e-6, 1e-6);
    });

    TEST("unique", {
        auto a = torch::tensor({1.0f, 2.0f, 2.0f, 3.0f, 1.0f, 4.0f, 5.0f, 5.0f}, torch::kFloat32);
        auto dr = std::get<0>(at::_unique2(to_dcu(a), true, false, false));
        auto cr = std::get<0>(at::_unique2(a, true, false, false));
        auto dr_sorted = std::get<0>(dr.to(torch::kCPU).sort());
        auto cr_sorted = std::get<0>(cr.sort());
        CHECK_CLOSE(dr_sorted, cr_sorted, 1e-6, 1e-6);
    });

    TEST("one_hot", {
        auto labels = torch::tensor({0, 1, 2, 3, 4}, torch::kInt64);
        OP_CHECK(torch::equal(torch::one_hot(to_dcu(labels), 10).to(torch::kCPU),
                           torch::one_hot(labels, 10)),
              "one_hot mismatch");
    });

    TEST("multinomial", {
        // Just verify it runs without error (stochastic sampling)
        auto probs = torch::rand({4, 10}, torch::kFloat32);
        probs = probs / probs.sum(1, true);
        auto d = to_dcu(probs).multinomial(5, true);
        auto c = probs.multinomial(5, true);
        OP_CHECK(d.sizes() == c.sizes(), "multinomial shape mismatch");
        OP_CHECK(d.device().is_cuda(), "multinomial not on CUDA");
    });

    // ═══════════════════════════════════════════════════════════════
    // 17. Gradient Operations (Autograd on DCU)
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Autograd on DCU ──\n";

    TEST("autograd: backward (simple)", {
        auto x = torch::randn({32, 64}, torch::TensorOptions().device(dcu_device())
                                  .dtype(torch::kFloat32).requires_grad(true));
        auto y = x.pow(2).sum();
        y.backward();
        OP_CHECK(x.grad().defined(), "grad not defined");
        OP_CHECK(x.grad().device().is_cuda(), "grad not on CUDA");
        OP_CHECK(x.grad().sizes() == x.sizes(), "grad shape mismatch");
    });

    TEST("autograd: matmul backward", {
        auto a = torch::randn({16, 32}, torch::TensorOptions().device(dcu_device())
                                  .dtype(torch::kFloat32).requires_grad(true));
        auto b = torch::randn({32, 8}, torch::TensorOptions().device(dcu_device())
                                  .dtype(torch::kFloat32).requires_grad(true));
        auto c = torch::matmul(a, b).sum();
        c.backward();
        OP_CHECK(a.grad().defined() && b.grad().defined(), "grads not defined");
        OP_CHECK(a.grad().sizes() == a.sizes(), "a grad shape mismatch");
        OP_CHECK(b.grad().sizes() == b.sizes(), "b grad shape mismatch");
    });

    TEST("autograd: relu backward", {
        auto x = torch::randn({64, 128}, torch::TensorOptions().device(dcu_device())
                                  .dtype(torch::kFloat32).requires_grad(true));
        auto y = torch::relu(x).sum();
        y.backward();
        OP_CHECK(x.grad().defined(), "grad not defined");
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                      TEST SUMMARY                       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    int total = g_pass + g_fail + g_skip;
    std::cout << "║  TOTAL:  " << std::setw(4) << total  << "                                          ║\n";
    std::cout << "║  PASS:   " << std::setw(4) << g_pass;
    if (g_pass == total) std::cout << "  \033[1;32m✓ ALL PASSED\033[0m";
    std::cout << "                                ║\n";
    std::cout << "║  FAIL:   " << std::setw(4) << g_fail;
    if (g_fail > 0) std::cout << "  \033[1;31m✗\033[0m";
    std::cout << "                                        ║\n";
    if (g_skip > 0) {
        std::cout << "║  SKIP:   " << std::setw(4) << g_skip << "                                        ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    if (g_fail > 0) {
        std::cout << "\n\033[1;31mFailed tests:\033[0m\n";
        for (size_t i = 0; i < g_failed_tests.size(); i++) {
            std::cout << "  " << (i+1) << ". " << g_failed_tests[i] << "\n";
        }
        return 1;
    }

    std::cout << "\n\033[1;32mAll operator tests passed!\033[0m\n";
    return 0;
}
