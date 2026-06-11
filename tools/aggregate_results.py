"""
Aggregate CSV results from results/X/ directory.
Each row represents a dataset, each column represents a treatment.
"""
import glob
import os
import pandas as pd
from pathlib import Path

def aggregate_results(input_dir: str, output_dir: str = None):
    """
    Reads all CSV files from input_dir and aggregates them.
    
    Each CSV file has treatments in first column and metrics in other columns.
    Output will have treatments as columns and datasets as rows.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Find all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read all CSV files
    all_data = {}
    metrics = None
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # filename without extension
        df = pd.read_csv(csv_file)
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Get treatment column name (first column)
        trt_col = df.columns[0]
        
        # Get metric columns (all except first)
        if metrics is None:
            metrics = list(df.columns[1:])
        
        # Store data with treatment as key
        all_data[dataset_name] = {}
        for _, row in df.iterrows():
            treatment = row[trt_col].strip() if isinstance(row[trt_col], str) else row[trt_col]
            for metric in metrics:
                if metric not in all_data[dataset_name]:
                    all_data[dataset_name][metric] = {}
                all_data[dataset_name][metric][treatment] = row[metric]
    
    # Get all treatments (from first dataset, assuming all have same treatments)
    first_dataset = list(all_data.keys())[0]
    first_metric = metrics[0]
    treatments = list(all_data[first_dataset][first_metric].keys())
    
    print(f"Treatments: {treatments}")
    print(f"Metrics: {metrics}")
    print(f"Datasets: {list(all_data.keys())}")
    
    # Create aggregated DataFrames for each metric
    for metric in metrics:
        rows = []
        for dataset_name in all_data.keys():
            row = {"dataset": dataset_name}
            for treatment in treatments:
                row[treatment] = all_data[dataset_name][metric].get(treatment, None)
            rows.append(row)
        
        result_df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = output_path / f"aggregated_{metric}.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nSaved {metric} aggregation to: {output_file}")
        print(result_df.to_string(index=False))

def aggregate_results_0(input_dir: str, output_dir: str = None):
    """
    Reads all CSV files from input_dir and aggregates them.
    
    Each CSV file has treatments in first column and metrics in other columns.
    Output will have treatments as columns and datasets as rows.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Find all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read all CSV files
    agg_all = pd.DataFrame(columns=["dataset","b4-sd","b4-md","Init-sd", "Init-md","Rfn-sd", "Rfn-md"])
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # filename without extension
        tmp = {"dataset":"","b4-sd":0,"Init-sd":0,"Rfn-sd":0,"b4-md":0,"Init-md":0,"Rfn-md":0}
        tmp["dataset"] = dataset_name
        with open(csv_file, "r") as f:
            for ln in f:
                if "sd" in ln:
                    if "b4" in ln:
                        tmp["b4-sd"] = ln.split(",")[-1].strip()
                    if "Init" in ln:
                        tmp["Init-sd"] = ln.split(",")[-1].strip()
                    if "Rfn" in ln:
                        tmp["Rfn-sd"] = ln.split(",")[-1].strip()
                if "md" in ln:
                    if "b4" in ln:
                        tmp["b4-md"] = ln.split(",")[-1].strip()
                    if "Init" in ln:
                        tmp["Init-md"] = ln.split(",")[-1].strip()
                    if "Rfn" in ln:
                        tmp["Rfn-md"] = ln.split(",")[-1].strip()
        agg_all = pd.concat([agg_all, pd.DataFrame([tmp])], ignore_index=True)
   
    output_file = output_path / f"aggregated.csv"
    agg_all.to_csv(output_file, index=False)
    print(f"\nSaved aggregation to: {output_file}")
    print(agg_all.to_string(index=False))

def aggregate_results_sensitivity(input_dir: str, output_dir: str = None):
    """
    Reads all CSV files from input_dir and aggregates them.
    
    Each CSV file has treatments in first column and metrics in other columns.
    Output will have treatments as columns and datasets as rows.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Find all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Read all CSV files
    all_sigmas = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1]
    all_cols = ["Dataset"]
    for sigma in all_sigmas:
        all_cols.append("Initial-"+str(sigma)) 
    for sigma in all_sigmas:
        all_cols.append("Refined-"+str(sigma)) 
    agg_all = pd.DataFrame(columns=all_cols)
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # filename without extension
        tmp = {x:"" for x in all_cols}
        tmp["Dataset"] = dataset_name
        with open(csv_file, "r") as f:
            for ln in f:
                if "initial" in ln:
                    values = ln.split(",")
                    i = 1
                    for col in all_cols:
                        if "Initial" in col:
                            tmp[col] = values[i].strip()
                            i+=1
                if "refined" in ln:
                    values = ln.split(",")
                    i = 1
                    for col in all_cols:
                        if "Refined" in col:
                            tmp[col] = values[i].strip()
                            i+=1
        agg_all = pd.concat([agg_all, pd.DataFrame([tmp])], ignore_index=True)
   
    output_file = output_path / f"aggregated.csv"
    agg_all.to_csv(output_file, index=False)
    print(f"\nSaved aggregation to: {output_file}")
    print(agg_all.to_string(index=False))

def aggregate_results_rq2_1(input_dir: str, output_dir: str = None):
    """
    Reads all per-dataset CSV files from input_dir and produces
    one aggregated pivot table per metric column.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "aggregates"
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect only dataset CSVs (skip the aggregates folder)
    csv_files = sorted(
        p for p in input_path.glob("*.csv")
        if p.is_file()
    )

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} dataset CSV files")

    # ── Read all CSVs ──────────────────────────────────────────
    all_data = {}       # {dataset: {metric: {trt: value}}}
    metrics = None       # metric column names (everything except trt)

    for csv_file in csv_files:
        dataset_name = csv_file.stem
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()

        trt_col = df.columns[0]  # "trt"

        if metrics is None:
            metrics = list(df.columns[1:])

        all_data[dataset_name] = {}
        for _, row in df.iterrows():
            treatment = str(row[trt_col]).strip()
            for metric in metrics:
                if metric not in all_data[dataset_name]:
                    all_data[dataset_name][metric] = {}
                all_data[dataset_name][metric][treatment] = row[metric]

    # ── Determine treatment order ──────────────────────────────
    first_dataset = list(all_data.keys())[0]
    treatments = list(all_data[first_dataset][metrics[0]].keys())

    print(f"Treatments: {treatments}")
    print(f"Metrics:    {metrics}")
    print(f"Datasets:   {len(all_data)} total")

    # ── Write one pivot CSV per metric ─────────────────────────
    for metric in metrics:
        rows = []
        for dataset_name in sorted(all_data.keys()):
            row = {"dataset": dataset_name}
            for treatment in treatments:
                row[treatment] = all_data[dataset_name].get(metric, {}).get(treatment, None)
            rows.append(row)

        result_df = pd.DataFrame(rows)
        out_file = output_path / f"aggregated_{metric}.csv"
        result_df.to_csv(out_file, index=False)
        print(f"\n[Created] {out_file}")
        print(result_df.head(10).to_string(index=False))
        if len(result_df) > 10:
            print(f"  ... ({len(result_df)} datasets total)")


if __name__ == "__main__":
    # Choose paths that contain raw CSV files. XX redirects to where all csv files exist
    input_dir = "RQ3/results/clusterings/"
    output_dir = "RQ3/results/clusterings/aggregates"
    ## Choose the right function to aggregate results
    # For RQ0 - instability: aggregate_results_0()
    # For RQ0 - sensitivity analysis : aggregate_results_sensitivity()
    # For RQ2 and RQ3: aggregate_results()
    # For labeling budget specifically: aggregate_results_rq2_1()
    aggregate_results(input_dir, output_dir)
    aggregate_results_0(input_dir, output_dir)
    aggregate_results_rq2_1(input_dir, output_dir)
    aggregate_results_sensitivity(input_dir, output_dir)
    aggregate_results(input_dir, output_dir)

