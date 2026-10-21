#include "common.hpp"
#include <random>
#include <limits>

int main() {
  try {
    std::mt19937_64 rng(42);
    for(uint32_t p:{3u,101u,1073741789u,2147483647u}) {
      auto meta=rdbench::make_mont_prime(p);
      for(int t=0;t<1000;++t) {
        const uint32_t a=rng()%p,b=rng()%p;
        const auto ma=rdbench::to_mont32(a,meta),mb=rdbench::to_mont32(b,meta);
        const auto got=rdbench::mont_mul32(rdbench::mont_mul32(ma,mb,meta),1,meta);
        if(got!=(uint64_t)a*b%p) throw std::runtime_error("Montgomery differs from integer remainder");
      }
      for(int64_t x:{int64_t(-101),std::numeric_limits<int64_t>::min()}) {
        auto got=rdbench::mont_mul32(rdbench::to_mont32(x,meta),1,meta);
        int64_t expected=x%(int64_t)p;if(expected<0)expected+=p;
        if(got!=expected) throw std::runtime_error("Signed Montgomery encoding failed");
      }
    }
    for(uint32_t p:{0u,1u,2u,105u,2147483659u}) {
      bool rejected=false;
      try{rdbench::make_mont_prime(p);}catch(const std::runtime_error&){rejected=true;}
      if(!rejected) throw std::runtime_error("Unsupported modulus was accepted");
    }
    std::cout<<"4000 Montgomery products and signed-input cases match integer oracles\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
