# Strict portable benchmark verification

RNS dump verification checks nonempty distinct prime moduli below 2^31,
canonical residues, dimensions and binary lengths before reconstructing.
Signed CRT uses the proper midpoint for odd and even modulus products.
A NaN or infinity in a floating dump now causes CLI failure with null error
metrics instead of a successful-looking numerical report.

The Montgomery constructor rejects zero, one, even/composite and out-of-range
moduli. A portable C++ fixture checks 4,000 products against integer remainder,
plus signed INT64_MIN encoding. It stays active in Release builds.

```bash
python -m unittest discover -s tests -p test_verification.py -v
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

Local Debug and Release validation: **1 CTest** and **6 Python tests**. The
new verifier also rechecks all 108 final entries in the three stored M5 corpora:
K=319 (9570 bits), K=306 (9180 bits), K=15 (450 bits); every entry matches its
exact reference with sufficient capacity. This rechecks existing dumps, not
newly measured Metal performance. Original measurements are unchanged.

Fresh Metal qualification and a reproducible cross-device benchmark suite are
still required before extending performance claims to CUDA/ROCm or submitting
a consolidated RNS research-software paper.
