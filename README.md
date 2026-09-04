# PADO-Pauli

<p align="center">
  <img src="img/pado_logo.png" alt="PADO-Pauli" width="560">
</p>

**PADO-Pauli** (shipped as the Python package **`padopauli`**) is a GPU-accelerated,
differentiable **Pauli-propagation surrogate (PPS) engine** for simulating
expectation values of parameterized quantum circuits. Instead of
evolving a state vector, it propagates observables through the circuit in the Pauli
basis and compiles the result into a reusable tensor program — compile once, then
evaluate expectation values and gradients many times at different parameters.

- **Differentiable**: expectation values are PyTorch tensors; gradients flow to circuit
  parameters (and to classical layers feeding them) via a built-in manual VJP, or
  equivalently through host autograd (`diff_mode="autograd"`).
- **GPU-accelerated**: execution presets `"cpu"`, `"gpu"`, and `"hybrid"` choose where
  compilation and evaluation run. There is no vendor-specific code — the torch build you
  install (CUDA or ROCm) selects the GPU; on ROCm builds the `"cuda"` device maps to AMD
  GPUs.
- **Compile once, evaluate many**: batched parameters and data-embedding angles reuse the
  same compiled program; a quasi-sampler is available for sampling workloads.
- **Noise-aware**: depolarizing and amplitude-damping channels can be placed in the
  circuit.
- Optional cross-checks against PennyLane for small systems (`pip install
  padopauli[reference]`).

The engine ships as a compiled binary package; this repository contains the
documentation, worked examples, and the paper-reproduction suite.

## Requirements

| OS / arch | Python | torch | Supported presets |
|---|---|---|---|
| Linux x86-64 | 3.11 / 3.12 | 2.10 – 2.12 | `cpu`, `gpu`, `hybrid` |
| macOS Apple Silicon (arm64) | 3.11 / 3.12 | 2.10 – 2.12 | `cpu` |
| Windows x64 | 3.11 / 3.12 | 2.11 – 2.12 | `cpu` |

torch is auto-installed only if absent. macOS GPU execution is not supported because the
engine has no MPS path. Windows NVIDIA execution is not part of the supported 2.0.0
surface until it is validated on real GPU hardware.

**Linux GPU users (NVIDIA/CUDA or AMD/ROCm)**: install the torch 2.10–2.12 build matching
your GPU **first**, then install padopauli — the dependency `torch>=2.10,<2.13` is then
already satisfied and pip will not touch your torch. If you let pip resolve torch itself
it pulls the default build from PyPI, which can replace a working CUDA or ROCm install.

```bash
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128   # NVIDIA
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/rocm7.2 # AMD
```

## Install

The package is on [PyPI](https://pypi.org/project/padopauli/) as prebuilt wheels for the
platforms in the table above. torch and the other dependencies install automatically:

```bash
pip install padopauli
```

The notebooks and reproducibility scripts also use the `[reference]` extra (PennyLane,
for exact small-system cross-checks; validated with PennyLane 0.45.1 and
pennylane-lightning 0.45.0):

```bash
pip install "padopauli[reference]"
```

The same wheels are attached to this repository's
[GitHub Releases](https://github.com/Norma-Q/PADO-Pauli/releases) if you prefer to install
a specific file directly. Pick the one matching your OS and Python version
(`python --version`):

```bash
# Linux x86-64
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl   # Python 3.11
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl   # Python 3.12

# macOS Apple Silicon
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp311-cp311-macosx_11_0_arm64.whl   # Python 3.11
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp312-cp312-macosx_11_0_arm64.whl   # Python 3.12

# Windows x64
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp311-cp311-win_amd64.whl   # Python 3.11
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp312-cp312-win_amd64.whl   # Python 3.12
```

## Quickstart

```python
import torch
from padopauli import Circuit

qc = Circuit(n_qubits=4)
qc.rx(0, param_idx=0).ry(1, param_idx=1)
qc.cnot(0, 1)
qc.rzz(1, 2, param_idx=2)
qc.compile(observables=[("Z", [0])], preset="cpu")  # or preset="gpu"

thetas = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
vals = qc.expvals(thetas)   # shape (n_observables,)
vals.sum().backward()      # gradients in thetas.grad
```

`observables` accepts `(pauli, qubits[, coeff])` term tuples or prebuilt `PauliSum`
objects. The public API is the top-level namespace only
(`from padopauli import ...`); subpackages are implementation details.

## What is in this repository

```
tutorial/           Runnable API walkthrough notebooks, from quickstart to
                    preset tuning and training loops
examples/           One notebook per application case in the paper
                    (127-qubit kicked Ising, VQE H2, QAOA MaxCut, SAFE ma-QAOA)
reproducibility/    Measurement scripts behind the paper's figures and tables
                    (see reproducibility/README.md for the artifact-to-script map)
engine_benchmarks/  Cross-engine benchmark harness (CUDA only: cuPauliProp,
                    Qiskit pauli-prop, PauliPropagation.jl)
results_on_MI300X/  Recorded runs behind the paper, one tree per measurement
results_on_A100/    platform; the main result files carry a .runmeta.json sidecar
                    (reproducibility/DEVICE_RUNS.md describes the two platforms)
```

The notebooks and scripts run against the installed `padopauli` package (with the
`[reference]` extra, see Install). They additionally need
`matplotlib` for the figures and `jupyter` to run the notebooks:

```bash
pip install matplotlib jupyter
git clone https://github.com/Norma-Q/PADO-Pauli.git && cd PADO-Pauli/tutorial
jupyter lab
```

## License

Two separate and distinct sets of terms apply: one to the binary package, one to
the contents of this repository. Neither extends to the other.

- The **`padopauli` binary package** (the wheels) is proprietary. Noncommercial
  use is free regardless of organizational type or research funding source,
  including company-sponsored academic research.
  Each user must obtain the wheel directly from the official distribution source;
  redistribution or internal sharing is not permitted. See the
  [NORMA Binary License 1.0](NORMA-BINARY-LICENSE.md).
- Use of `padopauli` or its Outputs in or for a commercial product, service, paid
  engagement, production workflow, sale, or commercial licensing requires prior
  approval and a separate written agreement executed by both parties. Contact
  <contact@norma.co.kr> before such use.
- **Attribution.** If you publicly disclose an output of `padopauli`, or any result
  derived from one — in a publication, presentation, report, product documentation,
  or any other public disclosure — state expressly and visibly that it was produced
  with the software, identified by its official name, **PADO-Pauli** (Python package
  `padopauli`), together with the applicable attribution: the copyright notice
  "NORMA, Inc." and, once available, a citation of the associated publication. The
  form is free — an acknowledgment sentence, a footnote, a methods-section statement,
  or a citation all satisfy it — but the statement itself is required by Section 6
  of the Binary License.
- The **contents of this repository** (notebooks, scripts, documentation, and
  recorded data) are licensed under the Apache License 2.0 — see
  [LICENSE](LICENSE).

PADO-Pauli source code is not publicly distributed. Requests based on a stated need,
including contribution and maintenance proposals, are considered case by case. Email
<quantumlab@norma.co.kr> with your affiliation, intended use, and proposed contribution.
NORMA, Inc. may share source code at its discretion for an approved noncommercial
purpose. Commercial development, modification, use, distribution, or
commercialization involving the source requires prior approval and a separate written
agreement executed by both parties; NORMA, Inc. provides the agreement form on request.

Developed and maintained by Hyunwoo Kim (<hw_kim@norma.co.kr>,
<kimhw7537@gmail.com>) and Youngseok Lee (<ys_lee@norma.co.kr>,
<pop756hh@gmail.com>) at NORMA, Inc.

© 2026 NORMA, Inc.

<p align="center">
  <img src="img/NORMA_CI.png" alt="NORMA, Inc." width="160">
</p>

---

# PADO-Pauli — 한국어

<p align="center">
  <img src="img/pado_logo.png" alt="PADO-Pauli" width="560">
</p>

**PADO-Pauli**(Python 패키지명: **`padopauli`**)는 매개변수화 양자 회로의
기대값을 시뮬레이션하는 GPU 가속 미분 가능 **파울리 전파 대리(PPS) 엔진**입니다.
상태 벡터를 전개하는 대신 관측량을 파울리 기저에서 회로를 통해 전파하고,
그 결과를 재사용 가능한 텐서 프로그램으로 컴파일합니다. 한 번 컴파일한 뒤 여러
매개변수에 대한 기대값과 기울기를 반복해서 평가할 수 있습니다.

- **미분 가능**: 기대값은 PyTorch 텐서입니다. 내장된 수동 VJP 또는 호스트
  autograd(`diff_mode="autograd"`)를 통해 회로 매개변수와 그 매개변수를 생성하는
  고전 계층까지 기울기가 전파됩니다.
- **GPU 가속**: 실행 프리셋 `"cpu"`, `"gpu"`, `"hybrid"`로 컴파일과 평가가
  실행될 위치를 선택합니다. 벤더 전용 코드는 없으며, 설치된 torch 빌드(CUDA 또는
  ROCm)가 GPU를 결정합니다. ROCm 빌드에서는 `"cuda"` 장치가 AMD GPU에
  대응됩니다.
- **한 번 컴파일하고 반복 평가**: 배치 매개변수와 데이터 임베딩 각도에 동일한
  컴파일 프로그램을 재사용합니다. 샘플링 작업을 위한 준샘플러도 제공합니다.
- **노이즈 지원**: 탈분극 및 진폭 감쇠 채널을 회로에 배치할 수 있습니다.
- 소규모 시스템에서는 PennyLane과 선택적으로 교차 검증할 수 있습니다
  (`pip install padopauli[reference]`).

엔진은 컴파일된 바이너리 패키지로 배포됩니다. 이 저장소에는 문서, 실행 예제,
논문 결과 재현 도구가 포함되어 있습니다.

## 요구 사항

| 운영체제 / 아키텍처 | Python | torch | 지원 프리셋 |
|---|---|---|---|
| Linux x86-64 | 3.11 / 3.12 | 2.10 – 2.12 | `cpu`, `gpu`, `hybrid` |
| macOS Apple Silicon (arm64) | 3.11 / 3.12 | 2.10 – 2.12 | `cpu` |
| Windows x64 | 3.11 / 3.12 | 2.11 – 2.12 | `cpu` |

torch는 없을 때만 자동 설치됩니다. macOS는 엔진에 MPS 경로가 없어 GPU 실행을
지원하지 않습니다. Windows NVIDIA 실행은 실제 GPU 하드웨어에서 검증되기 전까지
2.0.0 지원 범위에 포함되지 않습니다.

**Linux GPU 사용자(NVIDIA/CUDA 또는 AMD/ROCm)**: GPU에 맞는 torch 2.10–2.12 빌드를
**먼저** 설치한 뒤 padopauli를 설치하십시오. `torch>=2.10,<2.13` 의존성이 이미
충족되므로 pip가 설치된 torch를 건드리지 않습니다. pip에 torch 해결을 맡기면
PyPI 기본 빌드를 받아오므로, 동작 중인 CUDA 또는 ROCm 설치가 교체될 수 있습니다.

```bash
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128   # NVIDIA
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/rocm7.2 # AMD
```

## 설치

패키지는 위 표의 플랫폼용 사전 빌드 휠로
[PyPI](https://pypi.org/project/padopauli/)에 게시되어 있습니다. torch와 다른
의존성은 자동으로 설치됩니다.

```bash
pip install padopauli
```

노트북과 재현 스크립트는 정확한 소규모 시스템 교차 검증을 위해 PennyLane이
포함된 `[reference]` 추가 의존성도 사용합니다(PennyLane 0.45.1,
pennylane-lightning 0.45.0에서 검증).

```bash
pip install "padopauli[reference]"
```

같은 휠이 이 저장소의
[GitHub Releases](https://github.com/Norma-Q/PADO-Pauli/releases)에도 첨부되어
있으므로 특정 파일을 직접 설치할 수도 있습니다. 운영체제와 Python 버전
(`python --version`)에 맞는 휠을 선택하십시오.

```bash
# Linux x86-64
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl   # Python 3.11
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl   # Python 3.12

# macOS Apple Silicon
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp311-cp311-macosx_11_0_arm64.whl   # Python 3.11
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp312-cp312-macosx_11_0_arm64.whl   # Python 3.12

# Windows x64
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp311-cp311-win_amd64.whl   # Python 3.11
pip install https://github.com/Norma-Q/PADO-Pauli/releases/download/v2.0.0/padopauli-2.0.0-cp312-cp312-win_amd64.whl   # Python 3.12
```

## 빠른 시작

```python
import torch
from padopauli import Circuit

qc = Circuit(n_qubits=4)
qc.rx(0, param_idx=0).ry(1, param_idx=1)
qc.cnot(0, 1)
qc.rzz(1, 2, param_idx=2)
qc.compile(observables=[("Z", [0])], preset="cpu")  # 또는 preset="gpu"

thetas = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
vals = qc.expvals(thetas)   # 형태: (n_observables,)
vals.sum().backward()       # 기울기는 thetas.grad에 저장
```

`observables`에는 `(pauli, qubits[, coeff])` 형식의 항 튜플 또는 미리 생성한
`PauliSum` 객체를 전달할 수 있습니다. 공개 API는 최상위 네임스페이스
(`from padopauli import ...`)뿐이며, 하위 패키지는 구현 세부 사항입니다.

## 저장소 구성

```
tutorial/           빠른 시작부터 프리셋 조정과 학습 루프까지 실행 가능한
                    API 안내 노트북
examples/           논문의 응용 사례별 노트북
                    (127큐비트 kicked Ising, VQE H2, QAOA MaxCut, SAFE ma-QAOA)
reproducibility/    논문 그림과 표를 만드는 측정 스크립트
                    (산출물-스크립트 대응은 reproducibility/README.md)
engine_benchmarks/  엔진 간 벤치마크 하니스 (CUDA 전용: cuPauliProp,
                    Qiskit pauli-prop, PauliPropagation.jl)
results_on_MI300X/  논문 수치의 기록 데이터, 측정 플랫폼별 트리
results_on_A100/    (주요 결과 파일에는 .runmeta.json 사이드카;
                    두 플랫폼의 사양은 reproducibility/DEVICE_RUNS.md)
```

노트북과 스크립트는 설치된 `padopauli` 패키지로 실행됩니다. 설치 절에서 설명한
`[reference]` 추가 의존성도 필요합니다. 그림 생성에는 `matplotlib`, 노트북
실행에는 `jupyter`가 추가로 필요합니다.

```bash
pip install matplotlib jupyter
git clone https://github.com/Norma-Q/PADO-Pauli.git && cd PADO-Pauli/tutorial
jupyter lab
```

## 라이선스

바이너리 패키지와 이 저장소의 내용에는 서로 별개의 독립된 두 가지 조건이 적용되며,
어느 한쪽이 다른 쪽에 미치지 않습니다.

- **`padopauli` 바이너리 패키지**(휠)는 전용 라이선스에 따라 배포됩니다. 조직
  유형이나 연구비 출처와 관계없이 비상업적 사용은 무료이며, 기업 후원 학술
  연구도 이에 포함됩니다. 각 사용자는 공식 배포 경로에서 휠을 직접 받아야 하며,
  재배포 또는 기관 내부 공유는 허용되지 않습니다. 자세한 조건은
  [NORMA 바이너리 라이선스 1.0](NORMA-BINARY-LICENSE.ko.md)을
  참조하십시오(기준 문서는 [영문본](NORMA-BINARY-LICENSE.md)입니다).
- 상업적 제품, 서비스, 유료 업무, 운영 환경, 판매 또는 상업적 라이선스에
  `padopauli`나 그 결과물을 사용하려면 사전 승인과 양 당사자가 체결한 별도
  서면계약이 필요합니다. 사용 전에 <contact@norma.co.kr>로 문의하십시오.
- **출처 표시.** `padopauli`로 생성한 연구 결과를 논문, 프리프린트, 학위논문, 발표,
  기술 보고서 등 어떠한 형태로든 공개할 때에는 그 결과가 이 소프트웨어로
  생성되었음을 공식 명칭 **PADO-Pauli**(Python 패키지 `padopauli`)로 명시하고,
  그에 적용되는 출처 표시, 즉 저작권 고지 "NORMA, Inc."와 (공개된 후에는) 관련
  출판물의 인용을 함께 기재하십시오. 형식은 자유입니다 — 사사 문장, 각주, 방법론
  절의 기술, 인용 모두 무방합니다 — 다만 명시 자체는 바이너리 라이선스 제5조가
  요구하는 사항입니다.
- 이 **저장소의 내용**(노트북, 스크립트, 문서 및 기록 데이터)은 Apache License
  2.0에 따라 제공됩니다. [LICENSE](LICENSE)를 참조하십시오.

PADO-Pauli 소스 코드는 공개 배포되지 않습니다. 기여 및 유지보수 제안을 포함하여
필요성을 밝힌 요청은 건별로 검토합니다. 소속, 사용 목적 및 제안하는 기여 내용을
적어 <quantumlab@norma.co.kr>로 문의하십시오. NORMA, Inc.는 승인한 비상업적 목적에
한하여 재량으로 소스 코드를 공유할 수 있습니다. 소스와 관련된 상업적 개발,
수정, 사용, 배포 또는 상업화에는 사전 승인과 양 당사자가 체결한 별도 서면계약이
필요하며, 계약서 양식은 요청 시 NORMA, Inc.가 제공합니다.

NORMA, Inc.의 Hyunwoo Kim(<hw_kim@norma.co.kr>, <kimhw7537@gmail.com>)과
Youngseok Lee(<ys_lee@norma.co.kr>, <pop756hh@gmail.com>)가 개발하고 유지보수합니다.

© 2026 NORMA, Inc.

<p align="center">
  <img src="img/NORMA_CI.png" alt="NORMA, Inc." width="160">
</p>
