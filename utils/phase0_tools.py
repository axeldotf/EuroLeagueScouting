import os
import re
import pandas as pd
from typing import Dict
from openpyxl import load_workbook, Workbook
from tqdm import tqdm

def extract_table(input_file: str, output_file: str, apply_column_mapping: bool = False) -> None:
    """
    Phase 0 - Extract and Clean Raw Player Data

    Extracts structured stats from a messy Excel sheet (starting at 'RNK').
    Optionally maps column names. Removes incomplete rows and fixes known issues.
    Saves cleaned data as both Excel and Pickle.
    """
    column_mapping = {
        "2PT%": "2P%", "PF 100 Poss": "2PA 100 POSS", "2PTA": "2PA/G", "2PTM": "2PM/G",
        "3PT%": "3P%", "3PTA": "3PA/G", "3PTM": "3PM/G", "AST": "Ast/G", "BLK": "BLK/G",
        "DF 100 Poss": "DF 100 POSS", "DF": "DF/G", "DR": "DR/G", "FGA": "FGA/G",
        "FGM": "FGM/G", "FT 100 Poss": "FTA 100 POSS", "FTA": "FTA/G", "FTM": "FTM/G",
        "MIN": "Min/G", "IND NET RTG": "NET RTG", "OR": "OR/G", "PTS": "PTS/G",
        "TM NAME": "TEAM", "ST": "ST/G", "TO%": "TO Ratio", "TO RATIO": "TO Ratio", "TO": "TO/G", "TR": "TR/G",
        "VAL": "Val/G"
    }

    wb = load_workbook(input_file, data_only=True)
    ws = wb.active

    start_row, start_col = None, None
    for row in ws.iter_rows():
        for cell in row:
            if cell.value == 'RNK':
                start_row, start_col = cell.row, cell.column
                break
        if start_row:
            break
    if start_row is None:
        raise ValueError("[ERROR] 'RNK' header not found in input sheet.")

    original_header = [ws.cell(row=start_row, column=c).value for c in range(start_col, ws.max_column + 1)]
    header = [column_mapping.get(col, col) for col in original_header] if apply_column_mapping else original_header

    try:
        name_col_idx = header.index('NAME')
    except ValueError:
        raise ValueError("[ERROR] 'NAME' column not found.")

    estimated_rows = 0
    for r in range(start_row + 1, ws.max_row + 1):
        if all(ws.cell(row=r, column=c).value is None for c in range(start_col, ws.max_column + 1)):
            break
        estimated_rows += 1

    table_data = [header]
    for r in tqdm(range(start_row + 1, start_row + 1 + estimated_rows), desc="Extracting rows"):
        row_values = [ws.cell(row=r, column=c).value for c in range(start_col, ws.max_column + 1)]
        if row_values[name_col_idx] is not None:
            table_data.append(row_values)

    new_wb = Workbook()
    new_ws = new_wb.active
    for r_idx, row in enumerate(table_data, start=1):
        for c_idx, value in enumerate(row, start=1):
            new_ws.cell(row=r_idx, column=c_idx).value = value or ""
    new_wb.save(output_file)

    df = pd.DataFrame(table_data[1:], columns=table_data[0])

    df.to_pickle(output_file.replace('.xlsx', '.pkl'))


def process_table(input_dir: str, output_dir: str) -> None:
    """
    Processes raw Excel files (seasonal + all-time) and saves clean versions in both Excel and Pickle.
    """
    files_to_process = [
        {'input_relative': os.path.join('23 24', 'Players stats.xlsx'), 'output_filename': '23_24_Players_stats_clean.xlsx'},
        {'input_relative': os.path.join('24 25', 'Players stats.xlsx'), 'output_filename': '24_25_Players_stats_clean.xlsx'},
        {'input_relative': os.path.join('all time', 'Statistiche giocatori - All time adv.xlsx'), 'output_filename': 'All_time_Players_adv_stats_clean.xlsx'},
        {'input_relative': os.path.join('all time', 'Statistiche giocatori - All time trad.xlsx'), 'output_filename': 'All_time_Players_trad_stats_clean.xlsx'}
    ]

    print("\n[INFO] Starting Excel processing pipeline...")
    for file in files_to_process:
        input_path = os.path.join(input_dir, file['input_relative'])
        output_path = os.path.join(output_dir, file['output_filename'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        apply_mapping = '23_24' in output_path or '24_25' in output_path
        print(f"[PROCESSING] {file['output_filename']} (Column mapping: {apply_mapping})")
        extract_table(input_path, output_path, apply_column_mapping=apply_mapping)
    print("[DONE] All Excel files processed.\n")


def merge_alltime(trad_path: str, adv_path: str, output_path: str) -> None:
    """
    Merges traditional and advanced all-time stats into a unified dataset.
    Cleans names, removes redundancy, and consolidates rows.
    """
    def read_file(path: str) -> pd.DataFrame:
        alt_path = path.replace('.xlsx', '.pkl')
        return pd.read_pickle(alt_path) if os.path.exists(alt_path) else pd.read_excel(path)

    trad = read_file(trad_path)
    adv = read_file(adv_path)

    for df in [trad, adv]:
        if 'NAME' in df.columns:
            df['NAME'] = df['NAME'].apply(lambda x: ' '.join(x.split(', ')[::-1]) if isinstance(x, str) and ', ' in x else x)

    merge_keys = ['RNK', 'NAME', 'TEAM', 'ROLE', 'NAT', 'AGE', 'GP', 'W/L', 'W%']
    merged_df = pd.merge(trad, adv, on=merge_keys, how='outer', suffixes=('', '_adv'))
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    merged_df.drop(columns=[col for col in ['MIN', 'SEASON_adv', 'HEIGHT_adv'] if col in merged_df.columns], inplace=True)

    grouping_keys = ['ROLE', 'TEAM', 'NAT', 'AGE', 'GP', 'W/L', 'W%']
    consolidated = []
    for keys, group in tqdm(merged_df.groupby(grouping_keys, dropna=False), desc="Consolidating"):
        base = dict(zip(grouping_keys, keys))
        rest = group.drop(columns=grouping_keys)
        combined = rest.apply(lambda col: col.dropna().unique()[0] if col.nunique(dropna=True) == 1 else col.dropna().iloc[0])
        consolidated.append({**base, **combined.to_dict()})
    final_df = pd.DataFrame(consolidated)

    final_df.sort_values(by=['NAME', 'SEASON'], inplace=True, ignore_index=True)
    final_df['RNK'] = range(1, len(final_df) + 1)

    front_cols = final_df.columns[7:11].tolist()
    ordered = front_cols + [col for col in final_df.columns if col not in front_cols]
    final_df = final_df[ordered]

    final_df.to_excel(output_path, index=False)
    final_df.to_pickle(output_path.replace('.xlsx', '.pkl'))
    print(f"[DONE] Merged all-time dataset saved to: {output_path}")

