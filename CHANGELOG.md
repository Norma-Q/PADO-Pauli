# Changelog

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
