# Changelog

## 2.1.0 (unreleased)

Minor release. New options and a more compact compiled-program format; one change of results
(coefficient truncation, see below).

### New
- `compile_program(..., build_embedding=...)` (also on `Circuit.compile` and
  `build_quasi_sampler`): the embedding angles that `build_min_abs` prunes at. Circuits with
  embedding gates can now be pruned at compile time; `build_min_abs` without `build_embedding`
  on such a circuit raises.
- Evaluation options on the preset (all off by default): `batch_chunk` sweeps a large
  parameter or embedding batch in slices to bound peak memory (`batch_chunk_grad` allows it
  under a gradient); `state_checkpoint=K` keeps every K-th intermediate state during training
  and recomputes the rest, trading time for memory; `cuda_graph` replays the forward
  evaluation of small circuits as a captured GPU graph.
- `CompiledProgram.close()` and `with compile_program(...) as prog:` release evaluation
  caches.
- Compile backends: a backend package can register itself (`register_backend`,
  `register_preset`, or the `padopauli.backends` entry-point group) and `preset="<name>"`
  then compiles on that backend. `padopauli.backend_api` lists the engine internals such a
  package may rely on.

### Changed
- `build_min_abs` now tracks a signed coefficient per observable while compiling, so paths of
  opposite sign that cancel are pruned less aggressively. Retained term sets and values of
  programs compiled with `build_min_abs` differ from 2.0.x (closer to the exact result);
  everything else is unchanged.
- The compiled program is stored in a more compact form: resident memory after compiling and
  peak memory while training are lower, and the first evaluation after compiling starts
  sooner. Code that read the step matrices of a compiled program directly must use the new
  types (`padopauli.backend_api`); `CompiledProgram.n_terms` and `.step_nnz` give the program
  size without touching them.
- Compilation of circuits with 32 or more qubits is a little faster.
- `pennylane_expvals` evaluates a batch of parameter vectors row by row instead of silently
  using only the first row.

### Notes
- Supported platforms are unchanged: Linux x86-64 (CPU, NVIDIA CUDA, AMD ROCm), macOS Apple
  Silicon (CPU), Windows x64 (CPU); Python 3.11 and 3.12; torch 2.10–2.12 (Windows: 2.11–2.12).
  `cuda_graph` applies to CUDA and ROCm only.

## 2.0.2 (2026-10-21)

Patch release. No API changes; results are unchanged.

### Improvements
- Faster compilation, with lower peak memory while compiling.
- Faster evaluation of large parameter batches, and the first evaluation after compiling starts sooner.
- Compiling with `build_thetas` that require gradients no longer uses extra memory.
- `scipy` is no longer a dependency.

### Fixes
- Inputs that used to give a silently wrong result now raise an error: a `PauliRotation` with
  lower-case or unknown Pauli letters, or with a Pauli word whose length differs from its qubit
  list; a rotation with neither or both of `param_idx` and `embedding_idx`; missing or too-short
  `thetas` or `embedding`; a `dtype` other than `"float32"` or `"float64"`; and `build_min_abs`
  together with embedding gates.
- Observable coefficients given as NumPy scalars or 0-dim tensors are accepted.
- `Circuit.compile()` no longer modifies the gate objects of the circuit.
- Asking for a second derivative with `diff_mode="vjp"` now raises an error that points to
  `diff_mode="autograd"`.
- The documentation of `observables=` now matches the behaviour: a nested list of terms is one
  multi-term observable.

### Notes
- Small circuits evaluated one parameter set at a time are a little faster than in 2.0.1; the
  further speed-up mentioned in the 2.0.1 notes is still planned.
- Supported platforms are unchanged: Linux x86-64 (CPU, NVIDIA CUDA, AMD ROCm), macOS Apple
  Silicon (CPU), Windows x64 (CPU); Python 3.11 and 3.12; torch 2.10–2.12 (Windows: 2.11–2.12).

## 2.0.1 (2026-10-07)

Patch release. No API changes; results are unchanged.

### Improvements
- Lower GPU memory and faster evaluation: a forward + backward pass needs about half the memory
  of 2.0.0, and larger programs that ran out of memory on an 80 GB GPU now fit.
- The first backward pass no longer spends tens of seconds building caches.
- Batched evaluation (many parameter sets at once) is faster.

### Fixes
- Recompiling a `Circuit` no longer keeps the previous program in GPU memory while the new one
  is built.
- An out-of-memory error during evaluation is reported immediately instead of being retried
  silently.

### Notes
- Circuits with amplitude-damping noise can differ from 2.0.0 by one unit in the last place
  (floating-point rounding only).
- Small circuits evaluated one parameter set at a time can be slightly slower per step; a
  follow-up release addresses this.
- Supported platforms are unchanged: Linux x86-64 (CPU, NVIDIA CUDA, AMD ROCm), macOS Apple
  Silicon (CPU), Windows x64 (CPU); Python 3.11 and 3.12; torch 2.10–2.12 (Windows: 2.11–2.12).

## 2.0.0 (2026-09-07)

First release of the rebuilt 2.x distribution: prebuilt wheels for Linux, macOS and Windows, the
`Circuit` API with the `cpu` / `gpu` / `hybrid` presets, differentiation, the quasi-probability
sampler, tutorials, examples and the paper's reproduction suite with recorded results.

---

# 변경 이력 — 한국어

## 2.1.0 (미출시)

마이너 릴리스입니다. 새 옵션과 더 작은 컴파일 결과 형식이 들어갔고, 결과가 바뀌는 변경이 하나
있습니다(계수 절단, 아래 참조).

### 새 기능
- `compile_program(..., build_embedding=...)`(`Circuit.compile`, `build_quasi_sampler`에도):
  `build_min_abs`가 절단에 쓰는 임베딩 각도입니다. 임베딩 게이트가 있는 회로도 컴파일 때
  절단할 수 있고, 그런 회로에서 `build_embedding` 없이 `build_min_abs`를 쓰면 오류가 납니다.
- 프리셋의 평가 옵션(기본값은 모두 끔): `batch_chunk`는 큰 파라미터·임베딩 배치를 조각내 최고
  메모리를 묶고(`batch_chunk_grad`는 gradient 계산에서도 허용), `state_checkpoint=K`는 학습
  때 K번째 중간 상태만 저장하고 나머지를 다시 계산해 시간을 메모리와 바꾸며, `cuda_graph`는
  작은 회로의 forward 평가를 GPU 그래프로 캡처해 재생합니다.
- `CompiledProgram.close()`와 `with compile_program(...) as prog:`가 평가 캐시를 놓습니다.
- 컴파일 백엔드: 백엔드 패키지가 스스로 등록하면(`register_backend`, `register_preset`,
  또는 `padopauli.backends` entry point) `preset="<이름>"`으로 그 백엔드에서 컴파일됩니다.
  그런 패키지가 쓸 수 있는 엔진 내부 이름은 `padopauli.backend_api`에 있습니다.

### 변경
- `build_min_abs`가 컴파일 중 관측량마다 부호 있는 계수를 따라가므로, 부호가 반대라 상쇄되는
  경로를 덜 잘라냅니다. `build_min_abs`로 컴파일한 프로그램의 항 집합과 값이 2.0.x와
  달라집니다(정확한 값에 더 가깝습니다). 그 밖의 결과는 그대로입니다.
- 컴파일 결과를 더 작은 형식으로 저장합니다. 컴파일 뒤 상주 메모리와 학습 중 최고 메모리가
  줄고, 컴파일 뒤 첫 평가가 더 빨리 시작합니다. 컴파일된 프로그램의 스텝 행렬을 직접 읽던
  코드는 새 타입(`padopauli.backend_api`)을 써야 하고, 프로그램 크기는
  `CompiledProgram.n_terms`·`.step_nnz`로 읽을 수 있습니다.
- 32큐비트 이상 회로의 컴파일이 조금 빨라졌습니다.
- `pennylane_expvals`가 파라미터 벡터 배치를 첫 행만 조용히 쓰지 않고 행마다 평가합니다.

### 참고
- 지원 플랫폼은 그대로입니다: Linux x86-64(CPU, NVIDIA CUDA, AMD ROCm), macOS Apple
  Silicon(CPU), Windows x64(CPU); Python 3.11·3.12; torch 2.10–2.12(Windows: 2.11–2.12).
  `cuda_graph`는 CUDA·ROCm에서만 동작합니다.

## 2.0.2 (2026-10-21)

패치 릴리스입니다. API 변경 없음, 결과 수치 동일.

### 개선
- 컴파일이 빨라지고, 컴파일 중 최고 메모리 사용량이 줄었습니다.
- 여러 파라미터 벌을 한 번에 평가하는 배치 평가가 빨라졌고, 컴파일 뒤 첫 평가도 더 빨리 시작합니다.
- gradient를 추적하는 `build_thetas`로 컴파일할 때 메모리를 더 쓰지 않습니다.
- `scipy`가 더 이상 의존성이 아닙니다.

### 수정
- 조용히 틀린 결과를 내던 입력이 이제 에러를 냅니다: `PauliRotation`에 소문자나 알 수 없는 Pauli
  글자, 또는 큐비트 수와 길이가 다른 Pauli 단어를 준 경우; `param_idx`와 `embedding_idx`를 둘 다 주지
  않았거나 둘 다 준 회전; `thetas`나 `embedding`이 없거나 짧은 경우; `"float32"`·`"float64"`가 아닌
  `dtype`; 임베딩 게이트와 `build_min_abs`를 함께 쓴 경우.
- 관측량 계수로 NumPy 스칼라나 0차원 텐서를 받습니다.
- `Circuit.compile()`이 회로의 게이트 객체를 더 이상 바꾸지 않습니다.
- `diff_mode="vjp"`로 2차 미분을 요청하면 `diff_mode="autograd"`를 안내하는 에러를 냅니다.
- `observables=` 설명을 실제 동작에 맞게 고쳤습니다: 항을 한 번 더 감싼 리스트가 다항 관측량 하나입니다.

### 참고
- 작은 회로를 파라미터 한 벌씩 평가하는 경우 2.0.1보다 조금 빨라졌습니다. 2.0.1 노트에서 말한 추가
  개선은 계속 진행 중입니다.
- 지원 플랫폼은 그대로입니다: Linux x86-64(CPU, NVIDIA CUDA, AMD ROCm), macOS Apple Silicon(CPU),
  Windows x64(CPU); Python 3.11·3.12; torch 2.10~2.12(Windows는 2.11~2.12).

## 2.0.1 (2026-10-07)

패치 릴리스입니다. API 변경 없음, 결과 수치 동일.

### 개선
- GPU 메모리 감소와 평가 속도 향상: forward + backward에 2.0.0의 절반 정도 메모리를 쓰고, 80 GB GPU에서
  메모리 부족이던 큰 프로그램이 들어갑니다.
- 첫 backward 호출이 수십 초 동안 캐시를 만들던 지연이 사라졌습니다.
- 배치 평가(여러 파라미터 벌을 한 번에)가 빨라졌습니다.

### 수정
- `Circuit`을 다시 컴파일할 때 이전 프로그램이 GPU 메모리에 남아 있던 문제를 고쳤습니다.
- 평가 중 메모리 부족 오류를 조용히 재시도하지 않고 즉시 보고합니다.

### 참고
- amplitude damping 노이즈가 든 회로는 2.0.0과 마지막 자리 하나가 다를 수 있습니다(부동소수점
  반올림 차이).
- 작은 회로를 파라미터 한 벌씩 평가하면 스텝당 조금 느려질 수 있습니다. 다음 릴리스에서 다룹니다.
- 지원 플랫폼은 그대로입니다: Linux x86-64(CPU, NVIDIA CUDA, AMD ROCm), macOS Apple Silicon(CPU),
  Windows x64(CPU); Python 3.11·3.12; torch 2.10~2.12(Windows는 2.11~2.12).

## 2.0.0 (2026-09-07)

새로 빌드한 2.x 배포판의 첫 릴리스: Linux·macOS·Windows용 사전 빌드 휠, `cpu`/`gpu`/`hybrid` 프리셋을
갖춘 `Circuit` API, 미분, 준확률 샘플러, 튜토리얼, 예제, 논문 재현 스위트와 기록 데이터.
