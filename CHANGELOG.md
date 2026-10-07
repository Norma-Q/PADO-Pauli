# Changelog

Notable changes to the `padopauli` wheels published from this repository. The recorded data under
`results_on_A100/` and `results_on_MI300X/` was produced with 2.0.0; 2.0.1 reproduces it unchanged
(see the 2.0.1 entry).

## 2.0.1 — 2026-10-07

A patch release: same public API, same compiled-program format, same numbers.

**Evaluation.** Every step matrix of a compiled program is a signed partial permutation, so the
evaluator now applies it as an index gather instead of a sparse matrix product. Measured on an
NVIDIA A100 80 GB (16-qubit grid Ising, `zero_filter=False`):

| | 2.0.0 | 2.0.1 |
|---|---|---|
| single evaluation, `max_weight=6` | 84 ms | 31 ms |
| forward + backward peak memory, `max_weight=6` | 24.5 GB | 10.3 GB |
| forward + backward, `max_weight=7` (16 M terms) | out of memory | 50 GB |
| first backward call, `max_weight=7` (cache build) | 50 s | 0.2 s |

Batched forward passes (200 parameter sets) are 1.3–1.5× faster. Small circuits evaluated one
parameter set at a time can be 10–20 % slower per training step (12–20 qubits: 27→30, 39→45,
51→61 ms), because the gather path issues more small kernels; a fused kernel is planned.

**Memory.** Step matrices are stored coalesced, so the evaluator's caches alias the program instead
of copying it; the evaluation sweep reserves one block the size of its working set up front, which
keeps the allocator's reserved pool close to the allocated amount; intermediate buffers are released
as soon as the next step no longer needs them; `Circuit.compile()` releases the previous program
before compiling, so a recompile never holds two programs at once; the `hybrid` preset moves the
state vector to the compute device before expanding it.

**Errors.** An out-of-memory error during evaluation is raised immediately. 2.0.0 caught it, retried
with another sparse layout and reported only the second failure.

**Numerics.** 2.0.1 reproduces every recorded experiment in `results_on_A100/` and a 22-case
comparison suite bit for bit, with one exception: circuits that contain amplitude-damping noise can
differ from 2.0.0 by at most one unit in the last place (float64 ≤ 9e-16, float32 ≤ 5e-7). The
compiled program is identical; the two-term sums of that step are rounded differently by the gather
kernel and by the sparse-matrix library.

**Platforms.** Unchanged: Linux x86-64 (CPU, NVIDIA CUDA and AMD ROCm GPUs), macOS Apple Silicon
(CPU), Windows x64 (CPU); Python 3.11 and 3.12; torch 2.10–2.12 (Windows: 2.11–2.12).

**Known limits carried over.** A 16-qubit, `max_weight=7` program without zero filter still runs out
of memory on an 80 GB GPU for a 200-parameter batch. `build_min_abs` prunes multi-term observables
by coefficient magnitude only, ignoring sign cancellation between terms; a fix is planned for a
later release.

## 2.0.0 — 2026-09-07

First release of the rebuilt 2.x distribution: prebuilt wheels (Linux x86-64, macOS Apple Silicon,
Windows x64; Python 3.11 and 3.12), the `Circuit` builder and `compile_program` API with the
`cpu` / `gpu` / `hybrid` presets, exact zero filtering, manual-VJP and autograd differentiation, the
quasi-probability sampler, tutorials, examples, the paper's reproduction suite and the recorded A100
and MI300X results. Zenodo: [10.5281/zenodo.22627345](https://doi.org/10.5281/zenodo.22627345).

---

# 변경 이력 — 한국어

이 저장소에서 배포하는 `padopauli` 휠의 주요 변경 사항입니다. `results_on_A100/`·`results_on_MI300X/`의
기록 데이터는 2.0.0으로 만든 것이고, 2.0.1은 이를 그대로 재현합니다(아래 2.0.1 항목 참조).

## 2.0.1 — 2026-10-07

패치 릴리스입니다. 공개 API, 컴파일된 프로그램의 형식, 결과 수치가 모두 같습니다.

**평가.** 컴파일된 프로그램의 모든 스텝 행렬은 부호 있는 부분 순열이므로, 평가기가 희소 행렬 곱 대신
인덱스 gather로 적용하도록 바꿨습니다. NVIDIA A100 80 GB, 16큐비트 격자 Ising, `zero_filter=False` 기준:

| | 2.0.0 | 2.0.1 |
|---|---|---|
| 단일 평가, `max_weight=6` | 84 ms | 31 ms |
| forward + backward 최고 메모리, `max_weight=6` | 24.5 GB | 10.3 GB |
| forward + backward, `max_weight=7`(1,600만 항) | 메모리 부족 | 50 GB |
| 첫 backward 호출, `max_weight=7`(캐시 생성) | 50 s | 0.2 s |

200개 파라미터 배치 forward는 1.3~1.5배 빨라졌습니다. 작은 회로를 파라미터 한 벌씩 평가하면 학습 스텝당
10~20% 느려질 수 있습니다(12~20큐비트: 27→30, 39→45, 51→61 ms). gather 경로가 작은 커널을 더 많이
띄우기 때문이며, 융합 커널을 준비 중입니다.

**메모리.** 스텝 행렬을 정렬(coalesce)된 상태로 저장해 평가기 캐시가 프로그램을 복사하지 않고 공유합니다.
평가 스윕이 시작할 때 작업 집합 크기의 블록을 한 번 잡았다 놓아 allocator의 예약 메모리를 실제 사용량
근처로 유지합니다. 중간 버퍼는 다음 스텝에 필요 없어지는 즉시 해제합니다. `Circuit.compile()`은 컴파일
전에 이전 프로그램을 먼저 해제해 재컴파일 시 두 프로그램이 공존하지 않습니다. `hybrid` 프리셋은 상태
벡터를 연산 장치로 옮긴 뒤 확장합니다.

**오류.** 평가 중 메모리 부족 오류는 즉시 발생합니다. 2.0.0은 이를 잡아 다른 희소 레이아웃으로 재시도한
뒤 두 번째 실패만 보고했습니다.

**수치.** 2.0.1은 `results_on_A100/`의 모든 기록 실험과 22케이스 대조 스위트를 비트 단위로 재현합니다.
예외는 하나로, amplitude damping 노이즈가 든 회로는 2.0.0과 최대 1 ulp(float64 ≤ 9e-16, float32 ≤ 5e-7)
다를 수 있습니다. 컴파일된 프로그램은 동일하고, 그 스텝의 두 항 합을 gather 커널과 희소 행렬 라이브러리가
다르게 반올림할 뿐입니다.

**플랫폼.** 변화 없음: Linux x86-64(CPU, NVIDIA CUDA·AMD ROCm GPU), macOS Apple Silicon(CPU), Windows
x64(CPU); Python 3.11·3.12; torch 2.10~2.12(Windows는 2.11~2.12).

**그대로인 한계.** 16큐비트 `max_weight=7` 프로그램을 zero filter 없이 200개 파라미터 배치로 돌리면 80 GB
GPU에서 여전히 메모리가 부족합니다. `build_min_abs`는 다항 관측량을 계수 크기만으로 절단하고 항 사이의
부호 상쇄를 무시합니다. 이후 릴리스에서 고칠 예정입니다.

## 2.0.0 — 2026-09-07

새로 빌드한 2.x 배포판의 첫 릴리스: 사전 빌드 휠(Linux x86-64, macOS Apple Silicon, Windows x64; Python
3.11·3.12), `Circuit` 빌더와 `compile_program` API, `cpu`/`gpu`/`hybrid` 프리셋, 정확한 zero filter,
수동 VJP·autograd 미분, 준확률 샘플러, 튜토리얼, 예제, 논문 재현 스위트와 A100·MI300X 기록 데이터.
Zenodo: [10.5281/zenodo.22627345](https://doi.org/10.5281/zenodo.22627345).
