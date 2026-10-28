# Changelog

## 2.2.1 (2026-10-28)

Patch release. One correctness fix for large GPU programs; every result that was right before is identical bit for bit.

### Fixes
- On AMD GPUs with the ROCm build of PyTorch, circuits on 64 or more qubits whose compilation selected more than 67 million rows in one step could compile to wrong expectation values with inflated term counts (the `cpu` preset was not affected). The engine now gathers rows in bounded pieces on the GPU, which avoids a fault in the underlying gather kernel; smaller programs take exactly the same code path as before.

### Notes
- The fault is in the GPU kernel, not in the propagation rules: the same circuit compiled on the CPU preset, or with fewer than 64 qubits, was already right. If you compiled 64-qubit-or-larger circuits on an AMD GPU with an earlier version, recompile with this one.

## 2.2.0 (2026-10-28)

Minor release. Every result of an existing input is identical to 2.1.1; the changes add Clifford gates, make the `hybrid` preset do what its name says, and remove device synchronisations from compilation.

### New
- Native Clifford gates `Y`, `Z`, `SDG` (S†), `SX` (√X), `CY` and `SWAP` next to `H`, `S`, `X`, `CNOT`, `CZ`: `qc.y / z / sdg / sx / cy / swap`, `CliffordGate("SWAP", [a, b])`, drawn by `draw()` and mirrored in the PennyLane reference path.
- `preset="hybrid"` now compiles on the host and evaluates on the GPU by streaming the program step by step (at most `chunk_size` rows at a time). Nothing of the program stays on the GPU between steps, so the GPU holds the state buffers and the current step only; the values are the same bits as `preset="gpu"`. `cuda_graph` is ignored for a streamed program (it would keep every step resident), and `diff_mode="autograd"` keeps the copies in its graph until backward.

### Improvements
- Compilation performs about half as many device-to-host synchronisations per step, and the first evaluation almost none: the builders hand their structural facts (unique rows and columns, ±1 signs, the cos block) to the stored format instead of re-deriving them.
- `hybrid` compilation sorts each step on the GPU before moving it to the host (the host sort was about half of a hybrid compile's time).

### Notes
- On AMD GPUs with the ROCm build of PyTorch, circuits on 64 or more qubits whose compilation selects more than 67 million rows in one step can compile to wrong expectation values in this version (and in 2.1.x). 2.2.1 fixes it; use 2.2.1 for such circuits.

## 2.1.1 (2026-10-28)

Patch release. Documentation and input checks only; compiled programs, stored format and
every result are identical to 2.1.0.

### Improvements
- `preset="gpu"` or `"hybrid"` on a machine whose torch sees no CUDA/ROCm device now raises a
  `RuntimeError` that names the preset and the fix (`preset="cpu"`, or a GPU build of torch),
  instead of torch's raw error from inside the compilation.
- Docstrings and the tutorial notes explain truncation accuracy: `max_weight` alone has no error bound and
  should be combined with `build_thetas` / `build_min_abs`; a program pruned at `build_thetas`
  keeps that term set, so recompile when the parameters move far from it; when compiling once
  pays off against propagating per parameter set; and that `Circuit.compile` releases the
  previous program before compiling (a failed compile leaves the circuit uncompiled; keep
  programs you still need with `compile_program(...)` and `program.close()`).

### Fixes
- Reproducibility scripts: the zero-filter ablation releases the previous program before
  taking its memory baseline (its `resident_gb` column was under-measured before), and the
  scaling sweep records an evaluation failure as `eval_error` instead of a silent empty field.

## 2.1.0 (2026-10-27)

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
- Preset option `min_abs_rule` (default `"exact"`): `"bound"` prunes on one upper bound per
  Pauli term instead of one signed coefficient per observable. It never drops a term the exact
  rule keeps, keeps somewhat more, and its compile cost does not grow with the number of
  observables, so it suits programs with hundreds or thousands of observables (for example
  `build_quasi_sampler` with high-order Z combinations).
- `build_quasi_sampler(...)` samplers expose their compiled program as `.program`.
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
- numpy 1.26 is supported again (the requirement is `numpy>=1.26`).

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

## 2.2.1 (2026-10-28)

패치 버전. 큰 GPU 프로그램의 정확성 수정 하나. 전에 맞던 결과는 모두 비트 단위로 같다.

### 수정
- AMD GPU의 ROCm PyTorch 빌드에서, 64큐비트 이상 회로가 컴파일 중 한 스텝에서 6,700만 행 넘게 선택하면 기댓값이 틀리고 항 수가 부풀 수 있었다(`cpu` 프리셋은 영향 없음). 엔진이 GPU에서 행을 일정 크기 조각으로 모아 바탕 gather 커널의 결함을 피한다. 더 작은 프로그램은 전과 똑같은 코드 경로를 지난다.

### 참고
- 결함은 전파 규칙이 아니라 GPU 커널에 있다. 같은 회로를 cpu 프리셋으로, 또는 64큐비트 미만으로 컴파일한 결과는 전에도 맞았다. 이전 버전으로 AMD GPU에서 64큐비트 이상 회로를 컴파일했다면 이 버전으로 다시 컴파일하기 바란다.

## 2.2.0 (2026-10-28)

마이너 버전. 기존 입력의 모든 결과는 2.1.1과 비트 단위로 같다. Clifford 게이트 추가, `hybrid` 프리셋의 평가 방식 변경, 컴파일의 장치 동기화 감소.

### 새 기능
- 네이티브 Clifford 게이트 `Y`, `Z`, `SDG`(S†), `SX`(√X), `CY`, `SWAP` 추가(기존 `H`, `S`, `X`, `CNOT`, `CZ`). `qc.y / z / sdg / sx / cy / swap`, `CliffordGate("SWAP", [a, b])`, `draw()`와 PennyLane 참조 경로에도 반영.
- `preset="hybrid"`는 이제 호스트에서 컴파일하고 GPU에서는 프로그램을 스텝 단위(최대 `chunk_size` 행)로 올려 평가한다. 스텝 사이에 프로그램이 GPU에 남지 않으므로 GPU에는 상태 버퍼와 현재 스텝만 있다. 값은 `preset="gpu"`와 비트 단위로 같다. 스트리밍 프로그램에서는 `cuda_graph`를 쓰지 않고(모든 스텝을 상주시키게 되므로), `diff_mode="autograd"`는 backward까지 사본을 그래프에 붙들어 둔다.

### 개선
- 컴파일의 스텝당 장치→호스트 동기화가 약 절반으로, 첫 평가의 동기화는 거의 0으로 줄었다. 빌더가 아는 구조 사실(행·열 유일성, ±1 부호, cos 블록)을 저장 형식이 다시 계산하지 않고 넘겨받는다.
- `hybrid` 컴파일은 각 스텝을 GPU에서 정렬한 뒤 호스트로 내린다(호스트 정렬이 hybrid 컴파일 시간의 절반 가까이였다).

### 참고
- AMD GPU의 ROCm PyTorch 빌드에서, 64큐비트 이상 회로가 컴파일 중 한 스텝에서 6,700만 행 넘게 선택하면 이 버전(과 2.1.x)은 기댓값이 틀릴 수 있다. 2.2.1에서 수정했으므로 그런 회로는 2.2.1을 쓰기 바란다.

## 2.1.1 (2026-10-28)

패치 버전. 문서와 입력 검사만 바뀌었고 컴파일 결과, 저장 형식, 모든 수치는 2.1.0과 같다.

### 개선
- torch가 CUDA/ROCm 장치를 보지 못하는 머신에서 `preset="gpu"`나 `"hybrid"`를 쓰면 프리셋 이름과
  해결책(`preset="cpu"` 또는 GPU용 torch 설치)을 담은 `RuntimeError`가 난다. 전에는 컴파일 안쪽에서
  torch의 원시 에러가 났다.
- docstring과 튜토리얼 노트에 절단 정확도 안내를 추가했다. `max_weight`만으로는 오차 상한이 없어
  `build_thetas` / `build_min_abs`와 함께 써야 하고, `build_thetas`에서 절단한 프로그램은 그 항 집합을
  유지하므로 파라미터가 멀리 움직이면 다시 컴파일해야 한다. 컴파일 한 번이 파라미터마다 전파하는
  방식보다 유리해지는 조건, 그리고 `Circuit.compile`이 이전 프로그램을 먼저 해제한다는 점(실패하면
  미컴파일 상태가 되며, 붙들 프로그램은 `compile_program(...)`과 `program.close()`로)도 적었다.

### 수정
- 재현 스크립트: zero filter 소거 실험이 기준 메모리를 재기 전에 이전 프로그램을 해제한다(기존
  `resident_gb` 열은 과소 측정이었다). 스케일링 스윕은 평가 실패를 빈 값 대신 `eval_error`로 기록한다.

## 2.1.0 (2026-10-27)

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
- 프리셋 옵션 `min_abs_rule`(기본 `"exact"`): `"bound"`는 관측량마다의 부호 있는 계수 대신
  Pauli 항마다 상한 하나로 절단합니다. 정확한 규칙이 남기는 항은 절대 버리지 않고 조금 더
  남기며, 컴파일 비용이 관측량 수에 비례해 늘지 않으므로 관측량이 수백·수천 개인
  프로그램(예: 고차 Z 조합의 `build_quasi_sampler`)에 맞습니다.
- `build_quasi_sampler(...)`가 돌려주는 샘플러가 컴파일된 프로그램을 `.program`으로 보여 줍니다.
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
- numpy 1.26을 다시 지원합니다(요구 사항 `numpy>=1.26`).

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
