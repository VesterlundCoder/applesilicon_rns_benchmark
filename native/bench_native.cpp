#include "common.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>

#ifdef HAVE_GMP
#include <gmpxx.h>
#endif

#ifdef HAVE_BOOST
#include <boost/multiprecision/cpp_int.hpp>
#endif

#ifdef HAVE_FLINT
extern "C" {
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
}
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

using namespace rdbench;

// Optimizer sink: prevents the compiler from eliding batched chains whose
// results are otherwise unused in the throughput measurement.
static std::atomic<unsigned long> g_sink{0};

struct Args {
    std::string corpus;
    std::string backend = "gmp";
    std::string dump;
    uint32_t ops = 0;
    int repeats = 3;
    int warmups = 1;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    uint32_t K = 64;
    uint32_t batch = 1;  // >1 selects trajectory-parallel throughput mode
};

static Args parse_args(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error(
        "usage: bench_native CORPUS [--backend boost|gmp|flint|accelerate|rns_cpu] "
        "[--ops N] [--repeats N] [--warmups N] [--threads N] [--k K] [--dump PATH]");
    Args a; a.corpus = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + s);
            return argv[i];
        };
        if (s == "--backend") a.backend = need();
        else if (s == "--ops") a.ops = static_cast<uint32_t>(std::stoul(need()));
        else if (s == "--repeats") a.repeats = std::stoi(need());
        else if (s == "--warmups") a.warmups = std::stoi(need());
        else if (s == "--threads") a.threads = std::stoi(need());
        else if (s == "--batch") a.batch = static_cast<uint32_t>(std::stoul(need()));
        else if (s == "--k") a.K = static_cast<uint32_t>(std::stoul(need()));
        else if (s == "--dump") a.dump = need();
        else throw std::runtime_error("unknown argument: " + s);
    }
    if (a.repeats < 1 || a.warmups < 0 || a.threads < 1 || a.K < 1 || a.batch < 1)
        throw std::runtime_error("invalid numeric argument");
    return a;
}

// Run `batch` independent chains across `threads` workers (a worker grabs the
// next chain index atomically) and return the wall time for the whole batch.
// This measures trajectory-level throughput on the full CPU, the fair CPU
// counterpart to the GPU batch-replay throughput.
template <class F>
static double time_pool(uint32_t batch, int threads, F per_chain) {
    std::atomic<uint32_t> next{0};
    std::vector<std::thread> ws;
    ws.reserve(threads);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < threads; ++i) ws.emplace_back([&] {
        uint32_t idx;
        while ((idx = next.fetch_add(1)) < batch) per_chain(idx);
    });
    for (auto& w : ws) w.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void emit_throughput_json(const Args& a, uint32_t nops,
                                 const std::vector<double>& samples,
                                 const std::string& extra) {
    double med = median(samples);
    std::cout << std::setprecision(12)
              << "{\"backend\":\"" << json_escape(a.backend) << "\""
              << ",\"mode\":\"cpu_throughput\""
              << ",\"operator_count\":" << nops
              << ",\"dim\":6"
              << ",\"batch\":" << a.batch
              << ",\"threads\":" << a.threads
              << ",\"batch_median_s\":" << med
              << ",\"batch_min_s\":" << *std::min_element(samples.begin(), samples.end())
              << ",\"batch_mad_s\":" << mad(samples)
              << ",\"throughput_chains_per_s\":" << (a.batch / med)
              << ",\"repeats\":" << a.repeats
              << ",\"warmups\":" << a.warmups;
    if (!extra.empty()) std::cout << "," << extra;
    std::cout << "}\n";
}

static void emit_timing_json(const Args& a, uint32_t nops, double setup_s,
                             const std::vector<double>& samples,
                             const std::string& extra = "") {
    std::cout << std::setprecision(12)
              << "{\"backend\":\"" << json_escape(a.backend) << "\""
              << ",\"operator_count\":" << nops
              << ",\"dim\":6"
              << ",\"setup_s\":" << setup_s
              << ",\"median_s\":" << median(samples)
              << ",\"min_s\":" << *std::min_element(samples.begin(), samples.end())
              << ",\"max_s\":" << *std::max_element(samples.begin(), samples.end())
              << ",\"mad_s\":" << mad(samples)
              << ",\"repeats\":" << a.repeats
              << ",\"warmups\":" << a.warmups;
    if (!extra.empty()) std::cout << "," << extra;
    std::cout << "}\n";
}

#ifdef HAVE_BOOST
using boost::multiprecision::cpp_int;

static std::array<cpp_int, E> boost_chain(const std::vector<int64_t>& mats, uint32_t nops) {
    std::array<cpp_int, E> P{}, C{};
    for (int i = 0; i < DIM; ++i) P[i * DIM + i] = 1;
    for (uint32_t t = 0; t < nops; ++t) {
        for (auto& x : C) x = 0;
        const int64_t* A = mats.data() + static_cast<size_t>(t) * E;
        for (int i = 0; i < DIM; ++i) {
            for (int k = 0; k < DIM; ++k) {
                const cpp_int& pik = P[i * DIM + k];
                if (pik == 0) continue;
                for (int j = 0; j < DIM; ++j)
                    C[i * DIM + j] += pik * A[k * DIM + j];
            }
        }
        P.swap(C);
    }
    return P;
}

static void run_boost(const Args& a, const Corpus& c) {
    double setup_s = 0.0;
    for (int i = 0; i < a.warmups; ++i) (void)boost_chain(c.ops, c.nops);
    std::vector<double> times;
    std::array<cpp_int, E> result;
    for (int r = 0; r < a.repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        result = boost_chain(c.ops, c.nops);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    size_t bits = 0;
    for (auto& x : result) {
        cpp_int y = x < 0 ? -x : x;
        if (y != 0) bits = std::max(bits, static_cast<size_t>(boost::multiprecision::msb(y) + 1));
    }
    if (!a.dump.empty()) {
        std::ofstream f(a.dump);
        for (auto& x : result) f << x << "\n";
    }
    emit_timing_json(a, c.nops, setup_s, times,
                     "\"implementation\":\"Boost.Multiprecision cpp_int -O3\",\"max_abs_bits\":" + std::to_string(bits));
}
#endif

#ifdef HAVE_GMP
static std::array<mpz_class, E> gmp_chain(const std::vector<mpz_class>& mats, uint32_t nops) {
    std::array<mpz_class, E> P, C;
    for (int i = 0; i < DIM; ++i) P[i * DIM + i] = 1;
    for (uint32_t t = 0; t < nops; ++t) {
        for (auto& x : C) x = 0;
        const mpz_class* A = mats.data() + static_cast<size_t>(t) * E;
        for (int i = 0; i < DIM; ++i) {
            for (int k = 0; k < DIM; ++k) {
                const mpz_class& pik = P[i * DIM + k];
                if (pik == 0) continue;
                for (int j = 0; j < DIM; ++j)
                    mpz_addmul(C[i * DIM + j].get_mpz_t(), pik.get_mpz_t(), A[k * DIM + j].get_mpz_t());
            }
        }
        P.swap(C);
    }
    return P;
}

static void run_gmp(const Args& a, const Corpus& c) {
    auto s0 = std::chrono::steady_clock::now();
    std::vector<mpz_class> mats(c.ops.size());
    // int64_t is `long long` on macOS; gmpxx has no unambiguous long-long operator=,
    // so cast to `long` (64-bit under LP64, hence lossless) to select the exact overload.
    for (size_t i = 0; i < c.ops.size(); ++i) mats[i] = static_cast<long>(c.ops[i]);
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();

    for (int i = 0; i < a.warmups; ++i) (void)gmp_chain(mats, c.nops);
    std::vector<double> times;
    std::array<mpz_class, E> result;
    for (int r = 0; r < a.repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        result = gmp_chain(mats, c.nops);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    size_t bits = 0;
    for (auto& x : result) bits = std::max(bits, mpz_sizeinbase(x.get_mpz_t(), 2));
    if (!a.dump.empty()) {
        std::ofstream f(a.dump);
        for (auto& x : result) f << x.get_str() << "\n";
    }
    emit_timing_json(a, c.nops, setup_s, times, "\"max_abs_bits\":" + std::to_string(bits));
}
#endif

#ifdef HAVE_FLINT
struct FlintMats {
    fmpz_mat_struct* p = nullptr;
    size_t n = 0;
    explicit FlintMats(size_t n_) : n(n_) {
        p = static_cast<fmpz_mat_struct*>(flint_malloc(n * sizeof(fmpz_mat_struct)));
        if (!p) throw std::bad_alloc();
        for (size_t i = 0; i < n; ++i) fmpz_mat_init(p + i, DIM, DIM);
    }
    ~FlintMats() {
        if (p) {
            for (size_t i = 0; i < n; ++i) fmpz_mat_clear(p + i);
            flint_free(p);
        }
    }
};

static void run_flint(const Args& a, const Corpus& c) {
    flint_set_num_threads(a.threads);
    auto s0 = std::chrono::steady_clock::now();
    FlintMats mats(c.nops);
    for (uint32_t t = 0; t < c.nops; ++t)
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                fmpz_set_si(fmpz_mat_entry(mats.p + t, i, j),
                            static_cast<slong>(c.ops[static_cast<size_t>(t) * E + i * DIM + j]));
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();

    fmpz_mat_t P, C;
    fmpz_mat_init(P, DIM, DIM); fmpz_mat_init(C, DIM, DIM);
    auto compute = [&]() {
        fmpz_mat_one(P);
        for (uint32_t t = 0; t < c.nops; ++t) {
            fmpz_mat_mul(C, P, mats.p + t);
            fmpz_mat_swap(P, C);
        }
    };
    for (int i = 0; i < a.warmups; ++i) compute();
    std::vector<double> times;
    for (int r = 0; r < a.repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now(); compute(); auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    flint_bitcnt_t bits = 0;
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
            bits = std::max(bits, fmpz_bits(fmpz_mat_entry(P, i, j)));
    if (!a.dump.empty()) {
        std::ofstream f(a.dump);
        char* s = nullptr;
        for (int i = 0; i < DIM; ++i) for (int j = 0; j < DIM; ++j) {
            s = fmpz_get_str(nullptr, 10, fmpz_mat_entry(P, i, j));
            f << s << "\n";
            flint_free(s);
        }
    }
    fmpz_mat_clear(P); fmpz_mat_clear(C);
    emit_timing_json(a, c.nops, setup_s, times,
                     "\"threads\":" + std::to_string(a.threads) + ",\"max_abs_bits\":" + std::to_string(bits));
}
#endif

#ifdef __APPLE__
static std::pair<std::array<double, E>, int64_t> accelerate_chain(const std::vector<double>& mats, uint32_t nops) {
    std::array<double, E> P{}, C{};
    for (int i = 0; i < DIM; ++i) P[i * DIM + i] = 1.0;
    int64_t exp_sum = 0;
    for (uint32_t t = 0; t < nops; ++t) {
        const double* A = mats.data() + static_cast<size_t>(t) * E;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    DIM, DIM, DIM, 1.0, P.data(), DIM, A, DIM, 0.0, C.data(), DIM);
        double mx = 0.0;
        for (double x : C) mx = std::max(mx, std::abs(x));
        if (mx > 0.0 && std::isfinite(mx)) {
            int ex = 0; std::frexp(mx, &ex);
            for (double& x : C) x = std::ldexp(x, -ex);
            exp_sum += ex;
        }
        P.swap(C);
    }
    return {P, exp_sum};
}

static void run_accelerate(const Args& a, const Corpus& c) {
    auto s0 = std::chrono::steady_clock::now();
    std::vector<double> mats(c.ops.size());
    for (size_t i = 0; i < c.ops.size(); ++i) mats[i] = static_cast<double>(c.ops[i]);
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
    for (int i = 0; i < a.warmups; ++i) (void)accelerate_chain(mats, c.nops);
    std::vector<double> times;
    std::pair<std::array<double, E>, int64_t> result;
    for (int r = 0; r < a.repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now(); result = accelerate_chain(mats, c.nops); auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    if (!a.dump.empty()) write_f64_dump(a.dump, result.first, result.second);
    emit_timing_json(a, c.nops, setup_s, times,
                     "\"normalization\":\"power-of-two after every operator\",\"exp_sum\":" + std::to_string(result.second));
}
#endif

static void mul_mod6_mont(const uint32_t* P, const uint32_t* A, uint32_t* C, const MontPrime& pm) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            uint32_t acc = 0;
            for (int k = 0; k < DIM; ++k)
                acc = add_mod32(acc, mont_mul32(P[i * DIM + k], A[k * DIM + j], pm), pm.p);
            C[i * DIM + j] = acc;
        }
    }
}

static void run_rns_cpu(const Args& a, const Corpus& c) {
    auto primes = generate_primes30(a.K);
    std::vector<MontPrime> meta; meta.reserve(a.K);
    for (uint32_t p : primes) meta.push_back(make_mont_prime(p));
    auto s0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> enc(static_cast<size_t>(a.K) * c.nops * E);
    for (uint32_t k = 0; k < a.K; ++k) {
        const MontPrime& pm = meta[k];
        uint32_t* dst = enc.data() + static_cast<size_t>(k) * c.nops * E;
        for (size_t i = 0; i < static_cast<size_t>(c.nops) * E; ++i) dst[i] = to_mont32(c.ops[i], pm);
    }
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
    std::vector<uint32_t> residues(static_cast<size_t>(a.K) * E);

    auto compute = [&]() {
        std::atomic<uint32_t> next{0};
        int nt = std::min<int>(a.threads, a.K);
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (int ti = 0; ti < nt; ++ti) workers.emplace_back([&] {
            std::array<uint32_t, E> P{}, C{};
            while (true) {
                uint32_t k = next.fetch_add(1);
                if (k >= a.K) break;
                const MontPrime& pm = meta[k];
                P.fill(0); for (int i = 0; i < DIM; ++i) P[i * DIM + i] = pm.rmod;
                const uint32_t* base = enc.data() + static_cast<size_t>(k) * c.nops * E;
                for (uint32_t t = 0; t < c.nops; ++t) {
                    mul_mod6_mont(P.data(), base + static_cast<size_t>(t) * E, C.data(), pm);
                    P.swap(C);
                }
                auto outIt = residues.begin() + static_cast<size_t>(k) * E;
                for (int e = 0; e < E; ++e) outIt[e] = mont_mul32(P[e], 1u, pm);
            }
        });
        for (auto& th : workers) th.join();
    };

    for (int i = 0; i < a.warmups; ++i) compute();
    std::vector<double> times;
    for (int r = 0; r < a.repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now(); compute(); auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    if (!a.dump.empty()) write_rns_dump(a.dump, primes, residues);
    std::string extra = "\"threads\":" + std::to_string(a.threads) + ",\"K\":" + std::to_string(a.K) +
                        ",\"prime_bits\":30,\"modular_method\":\"Montgomery32\"";
#ifdef HAVE_BOOST
    double crt_s = benchmark_crt_boost(primes, residues);
    extra += ",\"crt_host_median_s\":" + std::to_string(crt_s) +
             ",\"exact_result_median_s\":" + std::to_string(median(times) + crt_s) +
             ",\"crt_backend\":\"Boost.Multiprecision\"";
#endif
    emit_timing_json(a, c.nops, setup_s, times, extra);
}

#ifdef HAVE_GMP
static void run_gmp_throughput(const Args& a, const Corpus& c) {
    std::vector<mpz_class> mats(c.ops.size());
    for (size_t i = 0; i < c.ops.size(); ++i) mats[i] = static_cast<long>(c.ops[i]);
    auto per = [&](uint32_t) {
        auto r = gmp_chain(mats, c.nops);
        g_sink.fetch_add(mpz_get_ui(r[0].get_mpz_t()), std::memory_order_relaxed);
    };
    for (int i = 0; i < a.warmups; ++i) (void)time_pool(a.batch, a.threads, per);
    std::vector<double> ts;
    for (int r = 0; r < a.repeats; ++r) ts.push_back(time_pool(a.batch, a.threads, per));
    emit_throughput_json(a, c.nops, ts,
                         "\"implementation\":\"GMP mpz_class per-chain, trajectory-parallel\"");
}
#endif

#ifdef HAVE_FLINT
static void run_flint_throughput(const Args& a, const Corpus& c) {
    flint_set_num_threads(1);  // parallelism is across chains, not within one
    FlintMats mats(c.nops);
    for (uint32_t t = 0; t < c.nops; ++t)
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                fmpz_set_si(fmpz_mat_entry(mats.p + t, i, j),
                            static_cast<slong>(c.ops[static_cast<size_t>(t) * E + i * DIM + j]));
    auto per = [&](uint32_t) {
        fmpz_mat_t P, C;
        fmpz_mat_init(P, DIM, DIM); fmpz_mat_init(C, DIM, DIM);
        fmpz_mat_one(P);
        for (uint32_t t = 0; t < c.nops; ++t) { fmpz_mat_mul(C, P, mats.p + t); fmpz_mat_swap(P, C); }
        g_sink.fetch_add(fmpz_get_ui(fmpz_mat_entry(P, 0, 0)), std::memory_order_relaxed);
        fmpz_mat_clear(P); fmpz_mat_clear(C);
    };
    for (int i = 0; i < a.warmups; ++i) (void)time_pool(a.batch, a.threads, per);
    std::vector<double> ts;
    for (int r = 0; r < a.repeats; ++r) ts.push_back(time_pool(a.batch, a.threads, per));
    emit_throughput_json(a, c.nops, ts,
                         "\"implementation\":\"FLINT fmpz_mat_mul per-chain, trajectory-parallel\"");
}
#endif

#ifdef __APPLE__
static void run_accelerate_throughput(const Args& a, const Corpus& c) {
    std::vector<double> mats(c.ops.size());
    for (size_t i = 0; i < c.ops.size(); ++i) mats[i] = static_cast<double>(c.ops[i]);
    auto per = [&](uint32_t) {
        auto r = accelerate_chain(mats, c.nops);
        g_sink.fetch_add(static_cast<unsigned long>(std::llabs(static_cast<long long>(r.first[0]))),
                         std::memory_order_relaxed);
    };
    for (int i = 0; i < a.warmups; ++i) (void)time_pool(a.batch, a.threads, per);
    std::vector<double> ts;
    for (int r = 0; r < a.repeats; ++r) ts.push_back(time_pool(a.batch, a.threads, per));
    emit_throughput_json(a, c.nops, ts,
                         "\"implementation\":\"Accelerate dgemm per-chain, trajectory-parallel (approx)\"");
}
#endif

#ifdef HAVE_BOOST
// End-to-end EXACT throughput ceiling for the RNS path: the modular kernel is
// faster than reconstruction, so for a workload that needs the exact integer of
// every trajectory the host CRT step is the bottleneck. We compute the residues
// of the real chain once, then time `batch` parallel centered-CRT
// reconstructions across `threads` workers. This is the fair "time-to-exact
// -result" counterpart to the GPU modular kernel throughput.
static void run_crt_throughput(const Args& a, const Corpus& c) {
    auto primes = generate_primes30(a.K);
    std::vector<MontPrime> meta; meta.reserve(a.K);
    for (uint32_t p : primes) meta.push_back(make_mont_prime(p));
    std::vector<uint32_t> residues(static_cast<size_t>(a.K) * E);
    for (uint32_t k = 0; k < a.K; ++k) {
        const MontPrime& pm = meta[k];
        std::array<uint32_t, E> P{}, C{}, Am{};
        P.fill(0); for (int i = 0; i < DIM; ++i) P[i * DIM + i] = pm.rmod;
        for (uint32_t t = 0; t < c.nops; ++t) {
            for (int e = 0; e < E; ++e) Am[e] = to_mont32(c.ops[static_cast<size_t>(t) * E + e], pm);
            mul_mod6_mont(P.data(), Am.data(), C.data(), pm);
            P.swap(C);
        }
        for (int e = 0; e < E; ++e) residues[static_cast<size_t>(k) * E + e] = mont_mul32(P[e], 1u, pm);
    }
    auto per = [&](uint32_t) {
        auto v = crt_centered_boost(primes, residues);
        unsigned long s = 0;  // observe all entries to defeat dead-code elimination
        for (int e = 0; e < E; ++e) s += v[e].is_zero() ? 0u : 1u;
        g_sink.fetch_add(s, std::memory_order_relaxed);
    };
    for (int i = 0; i < a.warmups; ++i) (void)time_pool(a.batch, a.threads, per);
    std::vector<double> ts;
    for (int r = 0; r < a.repeats; ++r) ts.push_back(time_pool(a.batch, a.threads, per));
    emit_throughput_json(a, c.nops, ts,
                         "\"K\":" + std::to_string(a.K) +
                         ",\"implementation\":\"centered CRT (Boost) per-matrix, trajectory-parallel\","
                         "\"note\":\"end-to-end exact reconstruction throughput ceiling for the RNS path\"");
}
#endif

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);
        Corpus c = load_corpus(a.corpus, a.ops);
        if (a.batch > 1) {
            // Trajectory-parallel CPU throughput mode.
#ifdef HAVE_GMP
            if (a.backend == "gmp") { run_gmp_throughput(a, c); return 0; }
#endif
#ifdef HAVE_FLINT
            if (a.backend == "flint") { run_flint_throughput(a, c); return 0; }
#endif
#ifdef __APPLE__
            if (a.backend == "accelerate") { run_accelerate_throughput(a, c); return 0; }
#endif
#ifdef HAVE_BOOST
            if (a.backend == "crt") { run_crt_throughput(a, c); return 0; }
#endif
            throw std::runtime_error("throughput mode (--batch>1) supports backends: gmp, flint, accelerate, crt");
        }
        if (a.backend == "rns_cpu") run_rns_cpu(a, c);
#ifdef HAVE_BOOST
        else if (a.backend == "boost") run_boost(a, c);
#endif
#ifdef HAVE_GMP
        else if (a.backend == "gmp") run_gmp(a, c);
#endif
#ifdef HAVE_FLINT
        else if (a.backend == "flint") run_flint(a, c);
#endif
#ifdef __APPLE__
        else if (a.backend == "accelerate") run_accelerate(a, c);
#endif
        else throw std::runtime_error("backend unavailable or unknown: " + a.backend);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
