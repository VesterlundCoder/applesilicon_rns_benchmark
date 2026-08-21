cmake_minimum_required(VERSION 3.20)
project(Apple6F5Bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Homebrew discovery (Apple Silicon) ---------------------------------------
# Apple clang does not search Homebrew's prefix by default, and Boost is header
# detected via check_include_file_cxx which ignores CMAKE_PREFIX_PATH. Wire the
# Homebrew prefix into the include/link/pkg-config search paths so GMP, FLINT and
# Boost are found on a stock `brew install boost gmp flint` setup.
if(APPLE)
  execute_process(COMMAND brew --prefix
                  OUTPUT_VARIABLE HOMEBREW_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(HOMEBREW_PREFIX AND EXISTS "${HOMEBREW_PREFIX}/include")
    message(STATUS "Homebrew prefix: ${HOMEBREW_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
    include_directories(SYSTEM "${HOMEBREW_PREFIX}/include")
    link_directories("${HOMEBREW_PREFIX}/lib")
    # Make check_include_file_cxx and find_* see the Homebrew tree.
    set(CMAKE_REQUIRED_INCLUDES "${HOMEBREW_PREFIX}/include")
    list(APPEND CMAKE_INCLUDE_PATH "${HOMEBREW_PREFIX}/include")
    list(APPEND CMAKE_LIBRARY_PATH "${HOMEBREW_PREFIX}/lib")
    if(DEFINED ENV{PKG_CONFIG_PATH})
      set(ENV{PKG_CONFIG_PATH} "${HOMEBREW_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    else()
      set(ENV{PKG_CONFIG_PATH} "${HOMEBREW_PREFIX}/lib/pkgconfig")
    endif()
  endif()
endif()
# -----------------------------------------------------------------------------

add_executable(bench_native bench_native.cpp)
target_compile_options(bench_native PRIVATE -O3 -DNDEBUG)

include(CheckIncludeFileCXX)
check_include_file_cxx("boost/multiprecision/cpp_int.hpp" HAVE_BOOST_HEADER)
if(HAVE_BOOST_HEADER)
  target_compile_definitions(bench_native PRIVATE HAVE_BOOST=1)
endif()

find_package(PkgConfig QUIET)

# GMP / GMPXX
if(PkgConfig_FOUND)
  pkg_check_modules(GMPXX QUIET IMPORTED_TARGET gmpxx)
  pkg_check_modules(GMP QUIET IMPORTED_TARGET gmp)
endif()
if(TARGET PkgConfig::GMPXX AND TARGET PkgConfig::GMP)
  target_link_libraries(bench_native PRIVATE PkgConfig::GMPXX PkgConfig::GMP)
  target_compile_definitions(bench_native PRIVATE HAVE_GMP=1)
else()
  find_path(GMPXX_INCLUDE_DIR gmpxx.h)
  find_library(GMPXX_LIBRARY gmpxx)
  find_library(GMP_LIBRARY gmp)
  if(GMPXX_INCLUDE_DIR AND GMPXX_LIBRARY AND GMP_LIBRARY)
    target_include_directories(bench_native PRIVATE ${GMPXX_INCLUDE_DIR})
    target_link_libraries(bench_native PRIVATE ${GMPXX_LIBRARY} ${GMP_LIBRARY})
    target_compile_definitions(bench_native PRIVATE HAVE_GMP=1)
  endif()
endif()

# FLINT
if(PkgConfig_FOUND)
  pkg_check_modules(FLINT QUIET IMPORTED_TARGET flint)
endif()
if(TARGET PkgConfig::FLINT)
  target_link_libraries(bench_native PRIVATE PkgConfig::FLINT)
  target_compile_definitions(bench_native PRIVATE HAVE_FLINT=1)
else()
  find_path(FLINT_INCLUDE_DIR flint/flint.h)
  find_library(FLINT_LIBRARY flint)
  if(FLINT_INCLUDE_DIR AND FLINT_LIBRARY)
    target_include_directories(bench_native PRIVATE ${FLINT_INCLUDE_DIR})
    target_link_libraries(bench_native PRIVATE ${FLINT_LIBRARY})
    target_compile_definitions(bench_native PRIVATE HAVE_FLINT=1)
  endif()
endif()

if(APPLE)
  target_link_libraries(bench_native PRIVATE "-framework Accelerate")

  enable_language(OBJCXX)
  add_executable(bench_metal bench_metal.mm)
  target_compile_options(bench_metal PRIVATE -O3 -fobjc-arc -DNDEBUG)
  if(HAVE_BOOST_HEADER)
    target_compile_definitions(bench_metal PRIVATE HAVE_BOOST=1)
  endif()
  target_compile_definitions(bench_metal PRIVATE BENCH_METAL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(bench_metal PRIVATE "-framework Foundation" "-framework Metal")
endif()

message(STATUS "Boost backend: ${HAVE_BOOST_HEADER}")
message(STATUS "GMPXX library: ${GMPXX_LIBRARY}")
message(STATUS "FLINT library: ${FLINT_LIBRARY}")
