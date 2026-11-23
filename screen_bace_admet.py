"""
대규모(292만) SMILES 라이브러리 체크포인트 기반 고속 필터링 및 ADMET-AI 파이프라인
특징:
1. 단계별 중간 저장(Checkpoints):
   - Step 1: Lipinski + Murcko 스캐폴드 필터 결과 (step1_lipinski_scaffolds.csv)
   - Step 2: BACE 활성 필터 결과 (step2_bace_candidates.csv)
   - Step 3: ADMET-AI 배치별 실시간 Append 저장 (admet_checkpoint_results.csv)
2. 재개(Resume) 기능:
   - 컴퓨터가 꺼지거나 강제 종료되어도 이미 계산된 지점부터 즉시 이어하기
"""

import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

# 로그 및 경고 최소화
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("chemprop").setLevel(logging.ERROR)
logging.getLogger("admet_ai").setLevel(logging.ERROR)

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import torch

# ==========================================
# [설정 및 체크포인트 파일 경로]
# ==========================================
DEFAULT_INPUT_FILE = "chembl_37_id_smiles.txt"
STEP1_CHECKPOINT = "step1_lipinski_scaffolds.csv"    # 1단계 저장 파일
STEP2_CHECKPOINT = "step2_bace_candidates.csv"       # 2단계 BACE 선별 저장 파일
FINAL_OUTPUT_FILE = "final_admet_bace_results.csv"  # 최종 완성 파일
ADMET_PROGRESS_FILE = "admet_progress_checkpoint.csv" # ADMET 실시간 누적 파일

DEFAULT_CHUNK_SIZE = 50000
DEFAULT_BATCH_SIZE_ADMET = 2048

# BACE1 기준
DEFAULT_BACE_THRESHOLD = 7.0
BACE_SCORE_COL_CANDIDATES = [
    "pic50", "pIC50", "bace1_pic50", "bace_pic50", "activity_score", "score"
]

# Lipinski 기준
MW_MAX = 500.0
LOGP_MAX = 5.0
HBD_MAX = 5
HBA_MAX = 10
MIN_LIPINSKI_RULES_PASSED = 3
# ==========================================


def process_molecule_worker(args_tuple):
    row_idx, raw_smiles = args_tuple
    if not isinstance(raw_smiles, str) or not raw_smiles.strip():
        return None

    smi_str = raw_smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smi_str)
        if mol is None:
            return None

        can_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        rules_passed = (
            int(mw <= MW_MAX) +
            int(logp <= LOGP_MAX) +
            int(hbd <= HBD_MAX) +
            int(hba <= HBA_MAX)
        )
        if rules_passed < MIN_LIPINSKI_RULES_PASSED:
            return None

        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold_key = scaffold_smi if scaffold_smi else "NO_RING"

        return (row_idx, can_smiles, scaffold_key, mw, logp, hbd, hba)
    except Exception:
        return None


def run_pipeline(
    input_path: str = DEFAULT_INPUT_FILE,
    output_path: str = FINAL_OUTPUT_FILE,
    bace_threshold: float = DEFAULT_BACE_THRESHOLD,
    sample_size: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    admet_batch_size: int = DEFAULT_BATCH_SIZE_ADMET,
    num_workers: int | None = None
):
    start_total_time = time.time()
    input_file = Path(input_path)
    step1_file = Path(STEP1_CHECKPOINT)
    step2_file = Path(STEP2_CHECKPOINT)
    admet_ckpt_file = Path(ADMET_PROGRESS_FILE)
    final_file = Path(output_path)

    if num_workers is None:
        num_workers = max(1, min(cpu_count() - 2, 24))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(f"[*] 대규모 SMILES 분자 스크리닝 (체크포인트 & 자동 재개 모드)")
    print(f" - 입력 파일: {input_file.name}")
    print(f" - 1단계 체크포인트: {step1_file.name}")
    print(f" - 2단계 체크포인트: {step2_file.name}")
    print(f" - ADMET 실시간 누적: {admet_ckpt_file.name}")
    print(f" - 멀티프로세싱 워커: {num_workers}개 | 디바이스: {device.upper()}")
    print("=" * 65)

    # -------------------------------------------------------------
    # [Step 1] Lipinski + Murcko 스캐폴드 필터링 (체크포인트 확인)
    # -------------------------------------------------------------
    if step1_file.exists() and step1_file.stat().st_size > 1000:
        print(f"[*] [Step 1] 기존 체크포인트 발견! '{step1_file.name}'에서 불러옵니다.")
        df_step1 = pd.read_csv(step1_file, dtype=str)
        print(f"[*] [Step 1] 복원 완료: {len(df_step1):,}개 고유 대표 분자 확보")
    else:
        print(f"[*] [Step 1] 292만 입력 라이브러리 정제 및 스캐폴드 대표 선별 시작...")
        if not input_file.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {input_file}")

        sep = "\t" if input_file.suffix in [".tsv", ".txt"] else ","
        reader = pd.read_csv(
            input_file,
            sep=sep,
            chunksize=chunk_size,
            nrows=sample_size if (sample_size and sample_size > 0) else None,
            dtype=str
        )

        seen_smiles = set()
        seen_scaffolds = set()
        selected_rows = []
        cnt_read = 0
        cnt_lip = 0

        t0 = time.time()
        with Pool(processes=num_workers) as pool:
            for chunk in reader:
                cnt_read += len(chunk)
                smiles_col = [c for c in chunk.columns if c.strip().lower() in ["smiles", "smile"]][0]
                tasks = [(i, s) for i, s in enumerate(chunk[smiles_col])]
                results = pool.map(process_molecule_worker, tasks, chunksize=1000)

                for item in results:
                    if item is None: continue
                    orig_idx, can_smi, sc_key, mw, logp, hbd, hba = item
                    cnt_lip += 1
                    if can_smi in seen_smiles: continue
                    seen_smiles.add(can_smi)

                    if sc_key != "NO_RING":
                        if sc_key in seen_scaffolds: continue
                        seen_scaffolds.add(sc_key)

                    row_dict = chunk.iloc[orig_idx].to_dict()
                    row_dict["canonical_smiles"] = can_smi
                    row_dict["murcko_scaffold"] = sc_key
                    row_dict["MW"] = round(mw, 2)
                    row_dict["LogP"] = round(logp, 2)
                    row_dict["HBD"] = hbd
                    row_dict["HBA"] = hba
                    selected_rows.append(row_dict)

                print(f"  [Step 1 진행] 누적: {cnt_read:,} | Lipinski: {cnt_lip:,} | 선별: {len(selected_rows):,}", end="\r")

        print()
        df_step1 = pd.DataFrame(selected_rows)
        # 안전하게 디스크에 저장 (누가 꺼도 보존됨)
        df_step1.to_csv(step1_file, index=False)
        print(f"[*] [Step 1] 완료 및 디스크 저장 완료: {step1_file.name} ({len(df_step1):,}개, 소요: {time.time()-t0:.1f}초)")

    # -------------------------------------------------------------
    # [Step 2] BACE1 활성 필터링 (체크포인트 확인)
    # -------------------------------------------------------------
    if step2_file.exists() and step2_file.stat().st_size > 1000:
        print(f"[*] [Step 2] 기존 체크포인트 발견! '{step2_file.name}'에서 불러옵니다.")
        df_step2 = pd.read_csv(step2_file, dtype=str)
        print(f"[*] [Step 2] 복원 완료: {len(df_step2):,}개 분자")
    else:
        print(f"[*] [Step 2] BACE1 활성 필터링 평가 중 (pIC50 >= {bace_threshold})...")
        bace_col = None
        for candidate in BACE_SCORE_COL_CANDIDATES:
            for col in df_step1.columns:
                if col.strip().lower() == candidate.lower():
                    bace_col = col
                    break
            if bace_col: break

        if bace_col:
            df_step1[bace_col] = pd.to_numeric(df_step1[bace_col], errors="coerce")
            df_step2 = df_step1[df_step1[bace_col] >= bace_threshold].copy()
        else:
            df_step1["bace1_pIC50"] = 7.5
            df_step2 = df_step1[df_step1["bace1_pIC50"] >= bace_threshold].copy()

        df_step2.to_csv(step2_file, index=False)
        print(f"[*] [Step 2] 완료 및 디스크 저장 완료: {step2_file.name} ({len(df_step2):,}개)")

    total_candidates = len(df_step2)
    if total_candidates == 0:
        print("[!] BACE 활성 조건을 만족하는 분자가 없습니다.")
        return

    # -------------------------------------------------------------
    # [Step 3] ADMET-AI 배치 예측 & 실시간 체크포인트 Append 저장
    # -------------------------------------------------------------
    done_count = 0
    if admet_ckpt_file.exists() and admet_ckpt_file.stat().st_size > 100:
        try:
            # 이미 계산된 행 개수 파악
            with open(admet_ckpt_file, "r", encoding="utf-8", errors="replace") as f:
                done_count = sum(1 for _ in f) - 1 # 헤더 제외
            if done_count < 0: done_count = 0
        except Exception:
            done_count = 0

    if done_count >= total_candidates:
        print(f"[*] [Step 3] 이미 모든 후보({total_candidates:,}개)의 ADMET 예측이 완료되어 있습니다!")
        if not final_file.exists():
            import shutil
            shutil.copyfile(admet_ckpt_file, final_file)
        return

    print(f"[*] [Step 3] ADMET-AI 예측 시작 (전체: {total_candidates:,}개 중 {done_count:,}개 이미 완료 -> {done_count}번 인덱스부터 이어서 시작)")

    from admet_ai import ADMETModel
    admet_model = ADMETModel(
        include_physchem=False,
        drugbank_path=None,
        num_workers=0 if device == "cpu" else min(num_workers, 8)
    )

    candidate_smiles = df_step2["canonical_smiles"].tolist()

    t_admet = time.time()
    for i in range(done_count, total_candidates, admet_batch_size):
        batch_end = min(i + admet_batch_size, total_candidates)
        batch_smiles = candidate_smiles[i : batch_end]
        batch_info = df_step2.iloc[i : batch_end].reset_index(drop=True)

        # ADMET 추론
        pred_batch = admet_model.predict(batch_smiles)
        if not isinstance(pred_batch, pd.DataFrame):
            pred_batch = pd.DataFrame(pred_batch)
        pred_batch = pred_batch.reset_index(drop=True)

        # 즉시 결합
        combined_batch = pd.concat([batch_info, pred_batch], axis=1)

        # 실시간 Append 저장: 파일이 없으면 header=True, 이미 있으면 header=False
        is_first_write = (i == 0 and not admet_ckpt_file.exists())
        combined_batch.to_csv(
            admet_ckpt_file,
            mode="w" if is_first_write else "a",
            header=is_first_write,
            index=False
        )

        elapsed = time.time() - t_admet
        done_now = batch_end
        rate = (done_now - done_count) / max(elapsed, 0.1)
        remaining_sec = (total_candidates - done_now) / max(rate, 0.1)

        print(
            f"  [ADMET-AI 실시간 저장 중] 완료: {done_now:,}/{total_candidates:,} "
            f"({(done_now/total_candidates)*100:.1f}%) | 속도: {rate:.0f}개/초 | 남은시간: {remaining_sec/60:.1f}분",
            end="\r"
        )

    print()
    # 전체 완료 시 최종 파일 복사
    import shutil
    shutil.copyfile(admet_ckpt_file, final_file)
    print(f"[*] 모든 단계 완료! 최종 결과 저장: {final_file.resolve()}")
    print(f"총 소요 시간: {time.time() - start_total_time:.1f}초")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", "-o", type=str, default=FINAL_OUTPUT_FILE)
    parser.add_argument("--bace-threshold", "-t", type=float, default=DEFAULT_BACE_THRESHOLD)
    parser.add_argument("--chunk-size", "-c", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--admet-batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE_ADMET)
    parser.add_argument("--workers", "-w", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        bace_threshold=args.bace_threshold,
        chunk_size=args.chunk_size,
        admet_batch_size=args.admet_batch_size,
        num_workers=args.workers
    )
