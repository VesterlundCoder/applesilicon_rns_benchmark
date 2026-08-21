#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_BOOST
#include <boost/multiprecision/cpp_int.hpp>
#endif

namespace rdbench {
constexpr int DIM = 6;
constexpr int E = 36;
constexpr char MAGIC[8] = {'R','D','C','M','F','6','I','1'};

struct Corpus {
    uint32_t nops = 0;
    std::vector<int64_t> ops; // [nops][36]
};

inline Corpus load_corpus(const std::string& path, uint32_t limit = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open corpus: " + path);
    char magic[8];
    uint32_t dim = 0, nops = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.read(reinterpret_cast<char*>(&nops), 4);
    if (!f || std::memcmp(magic, MAGIC, 8) != 0 || dim != DIM)
        throw std::runtime_error("bad corpus header");
    if (limit && limit < nops) nops = limit;
    Corpus c;
    c.nops = nops;
    c.ops.resize(static_cast<size_t>(nops) * E);
    f.read(reinterpret_cast<char*>(c.ops.data()), static_cast<std::streamsize>(c.ops.size() * sizeof(int64_t)));
    if (!f) throw std::runtime_error("truncated corpus");
    return c;
}

inline std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    return o;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n & 1 ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

inline double mad(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = median(v);
    std::vector<double> d;
    d.reserve(v.size());
    for (double x : v) d.push_back(std::abs(x - m));
    return median(std::move(d));
}

inline bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t p : {2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    auto modpow = [n](uint64_t a, uint64_t e) {
        uint64_t r = 1, x = a % n;
        while (e) {
            if (e & 1) r = (r * x) % n;
            x = (x * x) % n;
            e >>= 1;
        }
        return r;
    };
    uint32_t d = n - 1, s = 0;
    while ((d & 1u) == 0) { d >>= 1; ++s; }
    // More than sufficient for 32-bit n.
    for (uint32_t a : {2u, 3u, 5u, 7u, 11u}) {
        if (a >= n) continue;
        uint64_t x = modpow(a, d);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (uint32_t r = 1; r < s; ++r) {
            x = (x * x) % n;
            if (x == n - 1) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}

inline std::vector<uint32_t> generate_primes30(size_t K) {
    std::vector<uint32_t> p;
    p.reserve(K);
    uint32_t x = (1u << 30) - 35u;
    if ((x & 1u) == 0) --x;
    while (p.size() < K && x > (1u << 29)) {
        if (is_prime_u32(x)) p.push_back(x);
        x -= 2;
    }
    if (p.size() != K) throw std::runtime_error("failed to generate enough 30-bit primes");
    return p;
}

inline uint32_t mod_i64(int64_t x, uint32_t p) {
    int64_t r = x % static_cast<int64_t>(p);
    if (r < 0) r += p;
    return static_cast<uint32_t>(r);
}

struct MontPrime {
    uint32_t p;
    uint32_t pinv;  // -p^{-1} mod 2^32
    uint32_t rmod;  // 2^32 mod p
    uint32_t pad;
};

inline uint32_t mont_pinv32(uint32_t p) {
    // Newton iteration for inverse modulo 2^32, valid for odd p.
    uint32_t inv = 1u;
    for (int i = 0; i < 5; ++i) inv *= (2u - p * inv);
    return 0u - inv;
}

inline MontPrime make_mont_prime(uint32_t p) {
    if ((p & 1u) == 0u) throw std::runtime_error("Montgomery modulus must be odd");
    return MontPrime{p, mont_pinv32(p), static_cast<uint32_t>((uint64_t(1) << 32) % p), 0u};
}

inline uint32_t mont_mul32(uint32_t a, uint32_t b, const MontPrime& pm) {
    uint64_t t = static_cast<uint64_t>(a) * b;
    uint32_t m = static_cast<uint32_t>(t) * pm.pinv;
    uint64_t u = (t + static_cast<uint64_t>(m) * pm.p) >> 32;
    uint32_t r = static_cast<uint32_t>(u);
    return r >= pm.p ? r - pm.p : r;
}

inline uint32_t add_mod32(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t s = static_cast<uint64_t>(a) + b;
    return static_cast<uint32_t>(s >= p ? s - p : s);
}

inline uint32_t to_mont32(int64_t x, const MontPrime& pm) {
    return static_cast<uint32_t>((static_cast<uint64_t>(mod_i64(x, pm.p)) * pm.rmod) % pm.p);
}


#ifdef HAVE_BOOST
inline uint32_t inv_mod_u32(uint32_t a, uint32_t p) {
    int64_t t = 0, newt = 1;
    int64_t r = p, newr = a;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (r != 1) throw std::runtime_error("non-invertible CRT modulus");
    t %= static_cast<int64_t>(p);
    if (t < 0) t += p;
    return static_cast<uint32_t>(t);
}

inline std::array<boost::multiprecision::cpp_int, E>
crt_centered_boost(const std::vector<uint32_t>& primes,
                   const std::vector<uint32_t>& residues) {
    using boost::multiprecision::cpp_int;
    if (residues.size() != primes.size() * E)
        throw std::runtime_error("CRT residue shape mismatch");
    std::array<cpp_int, E> x{};
    cpp_int M = 1;
    for (size_t k = 0; k < primes.size(); ++k) {
        uint32_t p = primes[k];
        uint32_t Mmod = static_cast<uint32_t>(M % p);
        uint32_t inv = inv_mod_u32(Mmod, p);
        for (int e = 0; e < E; ++e) {
            uint32_t xmod = static_cast<uint32_t>(x[e] % p);
            uint32_t r = residues[k * E + e];
            uint32_t delta = r >= xmod ? r - xmod : static_cast<uint32_t>((uint64_t)r + p - xmod);
            uint32_t t = static_cast<uint32_t>((uint64_t)delta * inv % p);
            x[e] += M * t;
        }
        M *= p;
    }
    cpp_int half = M >> 1;
    for (auto& v : x) if (v > half) v -= M;
    return x;
}

inline double benchmark_crt_boost(const std::vector<uint32_t>& primes,
                                  const std::vector<uint32_t>& residues,
                                  int repeats = 3) {
    std::vector<double> t;
    t.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        auto v = crt_centered_boost(primes, residues);
        // Observe ALL entries so the optimizer cannot dead-code-eliminate the
        // reconstruction of x[1..E-1] (only reading v[0] under-measures ~3x).
        size_t sink_local = 0;
        for (int e = 0; e < E; ++e) sink_local += v[e].is_zero() ? 0u : 1u;
        volatile size_t sink = sink_local;
        (void)sink;
        auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    return median(t);
}
#endif

inline void write_rns_dump(const std::string& path, const std::vector<uint32_t>& primes,
                           const std::vector<uint32_t>& residues) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const char magic[8] = {'R','N','S','R','6','I','0','1'};
    uint32_t K = static_cast<uint32_t>(primes.size()), dim = DIM;
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(&K), 4);
    f.write(reinterpret_cast<const char*>(&dim), 4);
    f.write(reinterpret_cast<const char*>(primes.data()), static_cast<std::streamsize>(K * sizeof(uint32_t)));
    f.write(reinterpret_cast<const char*>(residues.data()), static_cast<std::streamsize>(residues.size() * sizeof(uint32_t)));
}

inline void write_f64_dump(const std::string& path, const std::array<double, E>& vals, int64_t exp_sum) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const char magic[8] = {'F','6','4','R','6','I','0','1'};
    uint32_t dim = DIM;
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(&dim), 4);
    f.write(reinterpret_cast<const char*>(&exp_sum), 8);
    f.write(reinterpret_cast<const char*>(vals.data()), E * sizeof(double));
}

} // namespace rdbench
