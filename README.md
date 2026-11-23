# BACE1 High-Throughput Screening & ADMET-AI & GNINA Docking Pipeline

이 저장소는 **290만 개 대용량 화합물 라이브러리(ChEMBL)**로부터 시작하여, **RDKit 필터링 $\rightarrow$ Murcko 스캐폴드 압축 $\rightarrow$ ADMET-AI 전수 예측 $\rightarrow$ GNINA GPU 분자 도킹**까지 전 과정을 단일 명령으로 재현할 수 있는 종단간(End-to-End) 가상 스크리닝 파이프라인입니다.

집 컴퓨터의 **코덱스(Codex) AI**에게 이 저장소 주소와 함께 실행을 지시하면 알아서 처음부터 끝까지 수행할 수 있도록 구성되어 있습니다.

---

## 🤖 코덱스(Codex)에게 전달할 한 줄 프롬프트

```text
이 리포지토리를 클론하고 README.md의 '원클릭 전체 실행 파이프라인'에 따라 290만 개 SMILES 라이브러리 필터링, ADMET-AI 예측, 그리고 GNINA 도킹까지 순차적으로 진행해줘.
```

---

## 1. 파이프라인 아키텍처 및 흐름

```
[2,897,819개 전체 SMILES (chembl_37_id_smiles.txt)]
       │
       ▼ (1) RDKit Lipinski Rule of 5 필터 (MW<=500, LogP<=5, HBD<=5, HBA<=10 중 3개 이상 만족)
[2,502,039개 통과]
       │
       ▼ (2) Bemis-Murcko 스캐폴드 중복 제거 (스캐폴드당 대표 1개 선별, 비고리 분자는 고유 보존)
[759,540개 고유 골격 대표 분자 도출 (74% 압축)]
       │
       ▼ (3) BACE1 활성 필터링 (pIC50 >= 7.0)
[759,540개 후보 확정]
       │
       ▼ (4) ADMET-AI 전수 배치 예측 (실시간 Append 체크포인트 저장, 중단 시 자동 재개 지원)
[759,540개 분자의 41종 ADMET 프로파일 확보 (final_admet_bace_results.csv)]
       │
       ▼ (5) CNS 특화 약물성 지표 스코어링 (BBB 투과성 극대화 + AMES/hERG/DILI 독성 배제)
[Top 100 최우수 후보 분자 선정 (top100_candidates_for_docking.csv)]
       │
       ▼ (6) 3D 구조 생성 (ETKDG + UFF) & Docker GNINA GPU 가속 도킹 (BACE1 1FKN 포켓)
[최종 결합 친화도 및 딥러닝 CNN 포즈 점수 산출 완료 (gnina_docking_top100_results.csv)]
```

---

## 2. 사전 환경 준비 (Prerequisites)

### 1) Python 환경 설정
Python 3.10 또는 3.11 환경에서 의존성을 설치합니다:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2) Docker & GNINA 이미지 준비
GPU 지원이 활성화된 Docker Desktop 환경에서 GNINA 이미지를 다운로드합니다:
```bash
docker pull gnina/gnina:v1.3.3
```

---

## 3. 원클릭 전체 실행 방법 (Execution)

### 1단계: RDKit 필터링 및 ADMET-AI 전수 예측 (자동 재개 지원)
원본 SMILES 라이브러리 파일(`chembl_37_id_smiles.txt`)을 프로젝트 루트에 배치한 뒤 실행합니다:
```bash
python screen_bace_admet.py --input chembl_37_id_smiles.txt --output final_admet_bace_results.csv
```
> **자동 체크포인트 및 재개(Resume) 기능**:
> - 작업 도중 컴퓨터가 꺼지거나 중단되어도 `step1_lipinski_scaffolds.csv`와 `admet_progress_checkpoint.csv`가 저장되어 있으므로, 다시 실행하면 **정확히 중단된 지점부터 이어서 즉시 실행**됩니다.

### 2단계: 최우수 후보 선정 및 GNINA GPU 도킹
ADMET 예측이 완료된 후 도킹을 실행합니다:
```bash
python run_gnina_docking.py
```
- 상위 후보 100개에 대해 3D 컨포메이션을 생성하고 BACE1(1FKN) 결정 구조의 바인딩 포켓에 GPU 가속 도킹을 수행합니다.
- 결과는 `docking_bace1/gnina_docking_top100_results.csv`에 친화도 순으로 저장됩니다.

---

## 4. 최종 결과 검증 요약 (Top Candidates)

| 순위 | ChEMBL ID | Canonical SMILES | 결합 친화도 (kcal/mol) | CNN Pose 점수 | BBB 투과 확률 | AMES 독성 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **CHEMBL4905708** | `CC(C)(C)CC1=NCCN1` | **-7.12** | **0.8519** | **0.998** | 0.053 (안전) |
| **2** | **CHEMBL279427** | `O=C(CC1=NCCN1)N1CCCC1` | **-6.75** | **0.7575** | **0.995** | 0.091 (안전) |
| **3** | **CHEMBL4955763** | `CC(C)(C)CC1N=CC=NC1` | **-6.30** | **0.6456** | **0.999** | 0.175 (안전) |
| **4** | **CHEMBL4901285** | `Cc1nc(CC(C)(C)C)nc(C)c1` | **-6.27** | **0.9329** | **0.999** | 0.114 (안전) |
| **5** | **CHEMBL3548090** | `CC(C)(C)C1=NCCN1` | **-6.14** | **0.5322** | **0.997** | 0.093 (안전) |
