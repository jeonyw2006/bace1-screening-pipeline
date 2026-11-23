"""
GNINA GPU 도킹 실행 스크립트 (BACE1 1FKN 단백질 대상)
1. 선별된 Top 100 분자 SMILES로부터 RDKit 3D 컨포머 생성 (ETKDG + UFF)
2. 개별 SDF 파일로 저장
3. Docker GNINA GPU 가속을 통해 BACE1 활성 자리 결합 도킹 수행
4. 결합 친화도(Affinity kcal/mol), CNN 점수(CNNscore, CNNaffinity) 파싱 및 종합 결과 CSV 저장
"""

import re
import subprocess
import time
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

DOCKING_DIR = Path("docking_bace1").resolve()
LIGANDS_DIR = DOCKING_DIR / "ligands"
OUTPUT_POSES_DIR = DOCKING_DIR / "docked_poses"
LIGANDS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_POSES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DOCKING_DIR / "top100_candidates_for_docking.csv"
FINAL_DOCKING_RESULTS_CSV = DOCKING_DIR / "gnina_docking_top100_results.csv"

def generate_3d_sdf(df):
    print(f"[*] {len(df)}개 분자에 대해 3D 컨포메이션 생성 중...")
    valid_ligands = []
    for idx, row in df.iterrows():
        cid = str(row["chembl_id"])
        smi = str(row["canonical_smiles"])
        sdf_path = LIGANDS_DIR / f"{cid}.sdf"
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res != 0:
                # fallback
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol)
            mol.SetProp("_Name", cid)
            
            with Chem.SDWriter(str(sdf_path)) as writer:
                writer.write(mol)
            valid_ligands.append((cid, sdf_path, row))
        except Exception as e:
            continue
            
    print(f"[*] 3D 구조 생성 완료: {len(valid_ligands)}개 분자 준비 완료")
    return valid_ligands

def run_gnina_docking(valid_ligands):
    print(f"[*] GNINA GPU 도킹 시작 (총 {len(valid_ligands)}개 대상)...")
    docking_results = []
    
    # Windows 절대 경로 -> Docker 볼륨 마운트
    # C:\Users\DS\deep learning\docking_bace1 -> /data
    host_data_dir = str(DOCKING_DIR).replace("\\", "/")
    
    t0 = time.time()
    for i, (cid, sdf_file, row_data) in enumerate(valid_ligands):
        out_sdf = OUTPUT_POSES_DIR / f"{cid}_docked.sdf"
        
        # docker command
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{host_data_dir}:/data",
            "gnina/gnina:v1.3.3",
            "gnina",
            "-r", "/data/receptor.pdb",
            "-l", f"/data/ligands/{cid}.sdf",
            "--autobox_ligand", "/data/crystal_ligand.pdb",
            "-o", f"/data/docked_poses/{cid}_docked.sdf",
            "--exhaustiveness", "8",
            "--num_modes", "3",
            "--device", "0"
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = res.stdout
            
            # 파싱: mode 1의 affinity, intramol, CNN pose score, CNN affinity
            # 표 형식: mode | affinity | intramol | CNN pose score | CNN affinity
            # 1   -7.53   -0.32   0.8492   3.918
            lines = output.splitlines()
            mode1_affinity = None
            mode1_cnn_score = None
            mode1_cnn_affinity = None
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "1":
                    try:
                        mode1_affinity = float(parts[1])
                        mode1_cnn_score = float(parts[3])
                        mode1_cnn_affinity = float(parts[4])
                        break
                    except ValueError:
                        pass
            
            res_dict = row_data.to_dict()
            res_dict["docking_affinity_kcal_mol"] = mode1_affinity
            res_dict["CNN_pose_score"] = mode1_cnn_score
            res_dict["CNN_affinity"] = mode1_cnn_affinity
            res_dict["docked_sdf_path"] = str(out_sdf)
            docking_results.append(res_dict)
            
            print(f"[{i+1}/{len(valid_ligands)}] {cid}: Affinity = {mode1_affinity} kcal/mol, CNNscore = {mode1_cnn_score}", flush=True)
            
        except Exception as e:
            print(f"[{i+1}/{len(valid_ligands)}] {cid}: 도킹 오류 - {e}", flush=True)
            
    print(f"[*] 도킹 전체 완료! 소요 시간: {time.time()-t0:.1f}초")
    df_results = pd.DataFrame(docking_results)
    # 친화도가 낮을수록(음수 절댓값이 클수록), CNNscore가 높을수록 우수
    if "docking_affinity_kcal_mol" in df_results.columns:
        df_results = df_results.sort_values(by=["docking_affinity_kcal_mol", "CNN_pose_score"], ascending=[True, False])
    df_results.to_csv(FINAL_DOCKING_RESULTS_CSV, index=False)
    print(f"[*] 최종 도킹 결과 저장 완료: {FINAL_DOCKING_RESULTS_CSV}")

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    valid_ligands = generate_3d_sdf(df)
    run_gnina_docking(valid_ligands)
