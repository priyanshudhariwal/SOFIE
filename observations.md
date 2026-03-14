# SOFIE ALPAKA GPU Implementation - Observations

## Exercise 3: Architecture Exploration & Improvements

### SOFIE Architecture Understanding

#### Code Generation Model
SOFIE generates C++ headers from ONNX models rather than interpreting them at runtime. This allows for compile-time optimization and minimal runtime overhead. The generated headers contain a `Session` class that can be instantiated to run inference on either CPU or GPU.

#### Three-Method GPU Pattern
GPU operators follow a consistent pattern with three methods that generate different parts of the GPU code:

1. **`Generate_GPU_Kernel_ALPAKA()`** - Returns a string containing the kernel struct definition. This struct has a templated `operator()` that contains the actual computational logic.

2. **`Generate_GPU_Kernel_Definitions_ALPAKA()`** - Returns a string that instantiates the kernel object. This creates the kernel instance that will be launched.

3. **`Generate_GPU_ALPAKA()`** - Returns a string with the kernel launch code. This configures the work division (grid size, block size) and launches the kernel with `alpaka::exec()`.

This separation allows SOFIE to generate modular, readable GPU code where kernels are defined once and can be reused multiple times in the same model.

#### ALPAKA Portability Layer
ALPAKA provides abstraction over CUDA, HIP, SYCL, and CPU backends. The same operator code can target different hardware without modification. The backend is selected at compile time using CMake options (`ALPAKA_BACKEND`), making the code portable across different GPU vendors and even CPU for testing.

#### Code Generation Pipeline
The build system uses CMake to:
1. Scan all `.onnx` files in `input_models/` directory
2. Generate `emitFromONNXAlpaka` executable that calls `RModel::GenerateGPU_ALPAKA()` for each model
3. Run the executable to produce `*_FromONNX_GPU_ALPAKA.hxx` headers in the build directory
4. Compile test executables that `#include` these generated headers
5. Run tests that instantiate `Session<alpaka::TagGpuCudaRt>` and call `infer()`

### Identified Improvements

#### 1. Missing `override` Keywords

**Location**: Multiple operator files in `src/SOFIE_core/inc/SOFIE/`

**Issue**: During build, the compiler generates warnings indicating that GPU methods override base class virtual functions but lack the `override` keyword:

```
/Users/priyanshu/SOFIE/src/SOFIE_core/inc/SOFIE/ROperator_BasicBinary.hxx:476:16: warning: 'Generate_GPU_ALPAKA' overrides a member function but is not marked 'override' [-Winconsistent-missing-override]
  476 |    std::string Generate_GPU_ALPAKA(std::string OpName) {
      |                ^
/Users/priyanshu/SOFIE/src/SOFIE_core/inc/SOFIE/ROperator.hxx:52:24: note: overridden virtual function is here
   52 |    virtual std::string Generate_GPU_ALPAKA(std::string OpName){ return "";} //expect unique opName for each operator within the same RModel
      |                        ^
```

**Affected Operators**:
- ROperator_BasicBinary.hxx (lines 389, 476)
- ROperator_BasicUnary.hxx (similar pattern)

**Why This Matters**:
- Catches typos in method names at compile time (e.g., if someone writes `Generate_GPU_Alpaka` instead of `Generate_GPU_ALPAKA`)
- Ensures signature compatibility with base class - compiler will error if base class signature changes
- Prevents silent failures where a method is meant to override but doesn't due to signature mismatch
- Modern C++ best practice (C++11 standard feature)
- Makes intent explicit to future maintainers

**Recommendation**: Add `override` keyword to all three GPU methods in operators that implement them:
```cpp
std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override { ... }
std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override { ... }
std::string Generate_GPU_ALPAKA(std::string OpName) override { ... }
```

This is a systematic change that can be applied across all GPU operator implementations.

#### 2. No Performance Benchmarking Infrastructure

**Observation**: Tests verify correctness (output matches expected values) but don't measure performance. There's no infrastructure to:
- Benchmark inference time
- Compare CPU vs GPU performance
- Track performance regressions over time
- Identify slow operators

**Impact**: Without performance metrics, it's unclear whether GPU implementations actually provide speedup, or if changes introduce performance regressions.

**Recommendation**: Add optional benchmark mode to tests (e.g., `--benchmark` flag) that:
- Runs inference multiple times and reports average/median time
- Measures CPU↔GPU transfer overhead separately from computation
- Outputs results in machine-readable format (JSON/CSV) for tracking trends

This would make performance optimization a data-driven process.



## Exercise 4: Implementation Work

### Operators Implemented

#### 1. Tanh Operator

**Files Modified**: `src/SOFIE_core/inc/SOFIE/ROperator_Tanh.hxx`

**Implementation Approach**:
Studied `ROperator_Sigmoid.hxx` as the primary reference since Tanh is also a unary elementwise activation function. The implementation follows the three-method pattern:

1. **Kernel Definition**: Simple elementwise application of `tanh()` function:
   ```cpp
   out[idx] = tanh(data[idx]);
   ```

2. **Kernel Instantiation**: Creates a `tanhKernel` object

3. **Kernel Launch**: Configures work division for 1D tensor and launches with `alpaka::exec()`

**Key Implementation Details**:
- Uses 1D work division: `Vec::all(static_cast<Idx>(1))` for elements per thread
- Configures grid size based on total tensor length
- Uses `alpaka::getValidWorkDiv()` to ensure configuration is valid for target device
- Passes input and output device buffers via `alpaka::getPtrNative()`

**Testing**:
- Added GPU test in `TestCustomModelsFromONNXForAlpakaCuda.cxx` (lines 145-194)
- Used identical input values as existing CPU test for direct comparison:
  ```cpp
  std::vector<float> input({
      -0.3896, -0.3521,  0.0363,  1.0962, ...
  });
  ```
- Verified output against `Tanh_ExpectedOutput::outputs` from `Tanh.ref.hxx`
- Tolerance: `1e-3f` (default for floating-point comparisons)


#### 2. Where Operator (Bonus - Exercise 5)

**Files Modified**: `src/SOFIE_core/inc/SOFIE/ROperator_Where.hxx`

**Implementation Approach**:
Where is significantly more complex than Tanh because it:
- Takes 3 inputs (condition C, value-if-true A, value-if-false B)
- Requires broadcasting when input shapes don't match
- Has condition tensor with bool/uint8_t type while other tensors use float

The kernel implements a ternary conditional operation:
```cpp
Y[idx] = C[idx] ? A[idx] : B[idx];
```

**Key Implementation Details**:
- Kernel signature takes 4 pointers: `T const* A, T const* B, std::uint8_t const* C, T* Y`
- Condition tensor explicitly typed as `std::uint8_t` (ONNX represents bool as uint8)
- Kernel launch passes 4 device buffers: `deviceBuf_A`, `deviceBuf_B`, `deviceBuf_C`, `deviceBuf_Y`

**Broadcasting Implementation Decision**:

The CPU implementation handles broadcasting by generating CPU code that calls `SOFIE::UTILITY::UnidirectionalBroadcast()` before the main computation loop. I have not implemented broadcasting in my implementation due to time constraints but this is how I would

**Rationale**:
- Keeps kernel code simple - always operates on matching-size tensors with straightforward indexing
- Avoids complex stride calculations in GPU kernel
- Broadcasting happens on CPU during model preparation (often with constant tensors)
- Matches existing SOFIE pattern where broadcasting is a separate preprocessing step

```cpp
// If A needs broadcasting, do it on CPU first
if (fShapeA != fShapeY) {
    out << SP << "SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName 
        << ">(tensor_" << fNA << ", " << ConvertShapeToString(fShapeA) 
        << ", " << ConvertShapeToString(fShapeY)
        << ", fTensor_" << fNBroadcastedA << ");\n";
    // Then copy fTensor_BroadcastedA to deviceBuf_A
}
```

**Trade-offs**:
- **Pro**: Simpler kernel, easier to verify correctness
- **Con**: Extra CPU→GPU transfer if broadcasting is needed at runtime

**Testing**:
- Added GPU test in `TestCustomModelsFromONNXForAlpakaCuda.cxx` (lines 196-244)
- Used same test case as CPU version for consistency:
  ```cpp
  std::vector<float> input1 = {1, 2};           // Shape: [2]
  std::vector<float> input2 = {3, 4, 5, 6};     // Shape: [4] (will broadcast)
  bool cond[] = {true, false, true};             // Shape: [3]
  // Expected output: [1, 2, 5, 6, 1, 2]        // Shape: [3, 2]
  ```
- This test verifies broadcasting logic: input1[2] broadcasts to [3,2], input2[4] broadcasts to [3,2]
- Three separate device buffers allocated and copied: `input1_device`, `input2_device`, `cond_device`
- Proper bool handling: allocated `alpaka::allocBuf<bool, Idx>` for condition tensor

**Test Structure**:
The GPU test properly handles the ALPAKA buffer API:
1. Allocate host buffers and fill with test data
2. Allocate device buffers
3. Copy host→device using `alpaka::memcpy(queue, device_buf, host_buf)`
4. Launch inference: `session.infer(input1_device, input2_device, cond_device)`
5. Copy result device→host
6. Verify against expected output



## AI Usage Disclosure

### What I Wrote Myself
- All operator implementation code in `ROperator_Tanh.hxx` and `ROperator_Where.hxx`
- All GPU test code in `TestCustomModelsFromONNXForAlpakaCuda.cxx`
- Decision-making on implementation approaches (CPU-side vs GPU-side broadcasting)
- Understanding of SOFIE's architecture through code exploration

### What AI Assisted With

#### 1. C++ Syntax and Language Features I was unfamiliar with previously
- ALPAKA-specific syntax: `ALPAKA_FN_ACC` macro, `alpaka::getIdx` usage patterns
- Template parameter syntax: `template<typename TAcc, typename T>`
- Understanding `__restrict__` keyword and when to use it
- Proper usage of `reinterpret_cast` for ALPAKA buffer pointers

#### 2. Understanding Existing Codebase
- Explaining how SOFIE's code generation works (RModel → string generation → file output)
- Walking through how CMake generates test executables
- Understanding ONNX operator semantics (particularly Where's broadcasting behavior)
- Explaining the relationship between `.onnx` files, generated headers, and tests

#### 3. Debugging Assistance
- Identifying typos through systematic comparison with reference implementations
- Understanding cryptic compiler errors related to ALPAKA template instantiation
- Explaining proper memory flow: host allocation → device allocation → memcpy → compute

#### 4. Development Tooling
- Clangd configuration syntax and how LSP works
- CMake compilation database generation (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`)
- Include path resolution for ROOT's hierarchical structure
- VS Code integration with Clangd

#### 5. Documentation and Presentation
- Markdown formatting for technical documents
- Structuring observations into logical sections
- Explaining complex concepts clearly and concisely
- Grammar checking and clarity improvements
