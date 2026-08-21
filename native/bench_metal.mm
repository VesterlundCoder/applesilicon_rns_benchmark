#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "common.hpp"

#include <cmath>
#include <cstdlib>
#include <sstream>

using namespace rdbench;

#ifndef BENCH_METAL_SOURCE_DIR
#define BENCH_METAL_SOURCE_DIR "."
#endif

struct Args {
    std::string corpus;
    std::string backend = "rns_gpu";
    std::string dump;
    std::string metal = std::string(BENCH_METAL_SOURCE_DIR) + "/bench_kernels.metal";
    uint32_t ops = 0;
    uint32_t K = 64;
    uint32_t batch = 1;
    int repeats = 5;
    int warmups = 1;
};

struct RNSCfg { uint32_t nOps, K, batch; };
struct DF64Cfg { uint32_t nOps, batch; };
struct F2 { float x, y; };

static Args parse_args(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error(
        "usage: bench_metal CORPUS [--backend rns_gpu|df64_gpu] [--ops N] [--k K] "
        "[--batch B] [--repeats N] [--warmups N] [--dump PATH] [--metal PATH]");
    Args a; a.corpus = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + s);
            return argv[i];
        };
        if (s == "--backend") a.backend = need();
        else if (s == "--ops") a.ops = static_cast<uint32_t>(std::stoul(need()));
        else if (s == "--k") a.K = static_cast<uint32_t>(std::stoul(need()));
        else if (s == "--batch") a.batch = static_cast<uint32_t>(std::stoul(need()));
        else if (s == "--repeats") a.repeats = std::stoi(need());
        else if (s == "--warmups") a.warmups = std::stoi(need());
        else if (s == "--dump") a.dump = need();
        else if (s == "--metal") a.metal = need();
        else throw std::runtime_error("unknown argument: " + s);
    }
    if (!a.K || !a.batch || a.repeats < 1 || a.warmups < 0) throw std::runtime_error("invalid numeric argument");
    return a;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open Metal source: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) throw std::runtime_error("missing Metal function: " + std::string([name UTF8String]));
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) throw std::runtime_error([[err localizedDescription] UTF8String]);
    return pso;
}

static void dispatch1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(w, 256));
    tg = (tg / w) * w;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

struct Timing {
    double gpu = 0.0;
    double wall = 0.0;
};

static Timing commit_wait(id<MTLCommandBuffer> cb) {
    auto t0 = std::chrono::steady_clock::now();
    [cb commit];
    [cb waitUntilCompleted];
    auto t1 = std::chrono::steady_clock::now();
    if (cb.status == MTLCommandBufferStatusError)
        throw std::runtime_error([[cb.error localizedDescription] UTF8String]);
    Timing t;
    t.wall = std::chrono::duration<double>(t1 - t0).count();
    if (cb.GPUEndTime > cb.GPUStartTime) t.gpu = cb.GPUEndTime - cb.GPUStartTime;
    return t;
}

static void emit_json(const Args& a, const Corpus& c, id<MTLDevice> dev, double setup_s,
                      const std::vector<double>& gpu, const std::vector<double>& wall,
                      const std::string& extra) {
    std::cout << std::setprecision(12)
              << "{\"backend\":\"" << json_escape(a.backend) << "\""
              << ",\"operator_count\":" << c.nops
              << ",\"dim\":6"
              << ",\"device\":\"" << json_escape([[dev name] UTF8String]) << "\""
              << ",\"setup_s\":" << setup_s
              << ",\"gpu_median_s\":" << median(gpu)
              << ",\"gpu_min_s\":" << *std::min_element(gpu.begin(), gpu.end())
              << ",\"gpu_mad_s\":" << mad(gpu)
              << ",\"wall_median_s\":" << median(wall)
              << ",\"batch\":" << a.batch
              << ",\"repeats\":" << a.repeats
              << ",\"warmups\":" << a.warmups;
    if (!extra.empty()) std::cout << "," << extra;
    std::cout << "}\n";
}

static void run_rns(const Args& a, const Corpus& c, id<MTLDevice> dev,
                    id<MTLCommandQueue> queue, id<MTLLibrary> lib) {
    auto pso = make_pipeline(dev, lib, @"rns_chain6");
    auto primes = generate_primes30(a.K);
    std::vector<MontPrime> meta; meta.reserve(a.K);
    for (uint32_t p : primes) meta.push_back(make_mont_prime(p));

    auto s0 = std::chrono::steady_clock::now();
    const size_t perPrime = static_cast<size_t>(c.nops) * E;
    std::vector<uint32_t> enc(static_cast<size_t>(a.K) * perPrime);
    for (uint32_t k = 0; k < a.K; ++k) {
        const MontPrime& pm = meta[k];
        uint32_t* dst = enc.data() + static_cast<size_t>(k) * perPrime;
        for (size_t i = 0; i < perPrime; ++i) dst[i] = to_mont32(c.ops[i], pm);
    }
    id<MTLBuffer> bOps = [dev newBufferWithBytes:enc.data() length:enc.size() * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> bPrimes = [dev newBufferWithBytes:meta.data() length:meta.size() * sizeof(MontPrime)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut = [dev newBufferWithLength:static_cast<size_t>(a.K) * a.batch * E * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];
    if (!bOps || !bPrimes || !bOut) throw std::runtime_error("Metal buffer allocation failed");
    RNSCfg cfg{c.nops, a.K, a.batch};
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();

    auto one = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        [ce setBuffer:bOps offset:0 atIndex:0];
        [ce setBuffer:bPrimes offset:0 atIndex:1];
        [ce setBytes:&cfg length:sizeof(cfg) atIndex:2];
        [ce setBuffer:bOut offset:0 atIndex:3];
        dispatch1D(ce, pso, static_cast<NSUInteger>(a.K) * a.batch);
        [ce endEncoding];
        return commit_wait(cb);
    };
    for (int i = 0; i < a.warmups; ++i) (void)one();
    std::vector<double> gpu, wall;
    for (int r = 0; r < a.repeats; ++r) {
        Timing t = one(); gpu.push_back(t.gpu); wall.push_back(t.wall);
    }

    auto* out = reinterpret_cast<uint32_t*>(bOut.contents);
    std::vector<uint32_t> r0(static_cast<size_t>(a.K) * E);
    for (uint32_t k = 0; k < a.K; ++k)
        std::copy(out + (static_cast<size_t>(k) * a.batch) * E,
                  out + (static_cast<size_t>(k) * a.batch) * E + E,
                  r0.begin() + static_cast<size_t>(k) * E);
    if (!a.dump.empty()) write_rns_dump(a.dump, primes, r0);

    double traj_per_gpu_s = a.batch / median(gpu);
    std::string extra = "\"K\":" + std::to_string(a.K) +
                        ",\"prime_bits\":30,\"modular_method\":\"Montgomery32\"," +
                        "\"trajectories_per_gpu_s\":" + std::to_string(traj_per_gpu_s) +
                        ",\"batch_replay\":true";
#ifdef HAVE_BOOST
    double crt_s = benchmark_crt_boost(primes, r0);
    extra += ",\"crt_host_median_s\":" + std::to_string(crt_s) +
             ",\"exact_result_wall_plus_crt_median_s\":" + std::to_string(median(wall) + crt_s) +
             ",\"crt_backend\":\"Boost.Multiprecision\"";
#endif
    emit_json(a, c, dev, setup_s, gpu, wall, extra);
}

static void run_df64(const Args& a, const Corpus& c, id<MTLDevice> dev,
                     id<MTLCommandQueue> queue, id<MTLLibrary> lib) {
    auto pso = make_pipeline(dev, lib, @"df64_chain6");
    auto s0 = std::chrono::steady_clock::now();
    std::vector<F2> ops(c.ops.size());
    for (size_t i = 0; i < c.ops.size(); ++i) {
        long double x = static_cast<long double>(c.ops[i]);
        float hi = static_cast<float>(x);
        float lo = static_cast<float>(x - static_cast<long double>(hi));
        ops[i] = {hi, lo};
    }
    id<MTLBuffer> bOps = [dev newBufferWithBytes:ops.data() length:ops.size() * sizeof(F2)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut = [dev newBufferWithLength:static_cast<size_t>(a.batch) * E * sizeof(F2)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> bExp = [dev newBufferWithLength:static_cast<size_t>(a.batch) * sizeof(int32_t)
                                           options:MTLResourceStorageModeShared];
    if (!bOps || !bOut || !bExp) throw std::runtime_error("Metal buffer allocation failed");
    DF64Cfg cfg{c.nops, a.batch};
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();

    auto one = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        [ce setBuffer:bOps offset:0 atIndex:0];
        [ce setBytes:&cfg length:sizeof(cfg) atIndex:1];
        [ce setBuffer:bOut offset:0 atIndex:2];
        [ce setBuffer:bExp offset:0 atIndex:3];
        dispatch1D(ce, pso, a.batch);
        [ce endEncoding];
        return commit_wait(cb);
    };
    for (int i = 0; i < a.warmups; ++i) (void)one();
    std::vector<double> gpu, wall;
    for (int r = 0; r < a.repeats; ++r) {
        Timing t = one(); gpu.push_back(t.gpu); wall.push_back(t.wall);
    }

    int32_t exp0 = reinterpret_cast<int32_t*>(bExp.contents)[0];
    if (!a.dump.empty()) {
        auto* out = reinterpret_cast<F2*>(bOut.contents);
        std::array<double, E> vals{};
        for (int e = 0; e < E; ++e) vals[e] = static_cast<double>(out[e].x) + static_cast<double>(out[e].y);
        write_f64_dump(a.dump, vals, exp0);
    }
    double traj_per_gpu_s = a.batch / median(gpu);
    emit_json(a, c, dev, setup_s, gpu, wall,
              "\"normalization\":\"power-of-two after every operator\",\"exp_sum\":" + std::to_string(exp0) +
              ",\"trajectories_per_gpu_s\":" + std::to_string(traj_per_gpu_s));
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Args a = parse_args(argc, argv);
            Corpus c = load_corpus(a.corpus, a.ops);
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (!dev) throw std::runtime_error("No Metal device");
            id<MTLCommandQueue> queue = [dev newCommandQueue];

            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:read_text(a.metal).c_str()];
            MTLCompileOptions* opts = [MTLCompileOptions new];
            opts.fastMathEnabled = NO; // df64 relies on IEEE behavior and explicit fma
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
            if (!lib) throw std::runtime_error([[err localizedDescription] UTF8String]);

            if (a.backend == "rns_gpu") run_rns(a, c, dev, queue, lib);
            else if (a.backend == "df64_gpu") run_df64(a, c, dev, queue, lib);
            else throw std::runtime_error("unknown backend: " + a.backend);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 2;
        }
    }
}
