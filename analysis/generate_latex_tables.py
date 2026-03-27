import json
import math
import os
from pathlib import Path

def format_p_value(p_val):
    """Format p-value for LaTeX table"""
    if p_val < 0.001:
        return f"{p_val:.2e}"
    elif p_val < 0.01:
        return f"{p_val:.3f}"
    else:
        return f"{p_val:.3f}"

def format_statistic(stat):
    """Format statistic value"""
    if isinstance(stat, float):
        return f"{stat:.2f}"
    else:
        return str(stat)

def format_effect_size(effect):
    """Format effect size"""
    return f"{effect:.2f}"

def format_cohens_dz(dz):
    """Format Cohen's dz. Use '—' when missing or unreliable."""
    if dz is None:
        return "—"
    try:
        if math.isnan(dz):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{dz:.2f}"

def format_ci(ci):
    """Format confidence interval. Use '—' when CI is missing or unreliable (e.g. small n)."""
    if ci is None or len(ci) != 2:
        return "—"
    lo, hi = ci[0], ci[1]
    try:
        if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"[{lo:.2f}, {hi:.2f}]"

def get_sig_symbol(significant):
    """Get significance symbol"""
    return "Yes" if significant else "No"

def normalize_dataset_name(dataset):
    """Normalize dataset name for display and escape underscores for LaTeX"""
    mapping = {
        "BI2015a": "BI2015a (ERP)",
        "BNCI2014_001": r"BNCI2014\_001 (MI)",
        "Lee2019_MI": r"Lee2019\_MI (MI)",
        "Lee2019_SSVEP": r"Lee2019\_SSVEP"
    }
    result = mapping.get(dataset, dataset)
    # Escape any remaining underscores
    result = result.replace("_", r"\_")
    return result

def normalize_eval_mode(eval_mode):
    """Normalize evaluation mode name"""
    mapping = {
        "CrossSession": "Cross-session",
        "WithinSession": "Within-session",
        "CrossSubject": "Cross-subject"
    }
    return mapping.get(eval_mode, eval_mode)

def normalize_model_name(model):
    """Normalize model name for display (e.g., CNN-NCP -> CNN-NCPv2)"""
    mapping = {
        "CNN-NCP": "CNN-NCPv2",
        "cnn_ncp": "CNN-NCPv2",
        "CNN_NCP": "CNN-NCPv2",
    }
    return mapping.get(model, model)

def swap_pairwise_comparison(comparison, cohens_dz, cohens_dz_ci):
    """Swap comparison to make HYDRA model1, negating effect size if needed"""
    parts = comparison.split(" vs ")
    if len(parts) != 2:
        return comparison, cohens_dz, cohens_dz_ci
    
    model1, model2 = parts[0], parts[1]
    
    # If HYDRA is already model1, return as is
    if model1 == "HYDRA":
        return comparison, cohens_dz, cohens_dz_ci
    
    # If HYDRA is model2, swap and negate
    if model2 == "HYDRA":
        new_comparison = f"HYDRA vs {model1}"
        new_dz = -cohens_dz
        new_ci = [-cohens_dz_ci[1], -cohens_dz_ci[0]]  # Swap and negate CI bounds
        return new_comparison, new_dz, new_ci
    
    # If HYDRA is not in comparison, return as is (but we'll filter these out)
    return comparison, cohens_dz, cohens_dz_ci

def generate_omnibus_table(data, metric_name, metric_key, tune, output_lines):
    """Generate omnibus test table"""
    tune_label = "tuned" if tune else "untuned"
    metric_label = "ROC-AUC" if metric_key == "clean_roc_auc" else "AUPC"
    
    output_lines.append(f"% =========================\n")
    output_lines.append(f"% Omnibus ({metric_label}, {tune_label})\n")
    output_lines.append(f"% =========================\n")
    output_lines.append(f"\\begin{{table}}[t]\n")
    output_lines.append(f"\\caption{{\\textbf{{{metric_label} Omnibus Tests (subject-level collapsed, {tune_label}).}} Omnibus statistical tests on subject-level collapsed {metric_label} across models (HYDRA, CNN-NCPv2, EEGNet, REEGNet). Omnibus comparisons use the Friedman test; effect sizes are reported using Kendall's $W$.}}\n")
    output_lines.append(f"\\label{{tab:stats_omnibus_{metric_key}_{tune_label}}}\n")
    output_lines.append(f"  \\begin{{center}}\n")
    output_lines.append(f"    \\begin{{small}}\n")
    output_lines.append(f"      \\begin{{sc}}\n")
    output_lines.append(f"        \\begin{{tabular}}{{llcccc}}\n")
    output_lines.append(f"          \\toprule\n")
    output_lines.append(f"          Dataset & Evaluation & Test & Statistic & Effect Size ($W$) & Sig. \\\\\n")
    output_lines.append(f"          \\midrule\n")
    
    # Collect all rows
    rows = []
    for dataset in ["BI2015a", "BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP"]:
        if dataset not in data:
            continue
        for eval_mode in ["CrossSession", "WithinSession", "CrossSubject"]:
            if eval_mode not in data[dataset]:
                continue
            tune_key = str(tune)
            if tune_key not in data[dataset][eval_mode]:
                continue
            if metric_key not in data[dataset][eval_mode][tune_key]:
                continue
            
            omnibus = data[dataset][eval_mode][tune_key][metric_key].get("omnibus", {})
            if not omnibus:
                continue
            
            test_type = omnibus.get("test_type", "").title()
            statistic = omnibus.get("statistic", 0)
            effect_size = omnibus.get("effect_size", 0)
            significant = omnibus.get("significant", False)
            
            dataset_display = normalize_dataset_name(dataset)
            eval_display = normalize_eval_mode(eval_mode)
            
            rows.append((
                dataset_display,
                eval_display,
                test_type,
                format_statistic(statistic),
                format_effect_size(effect_size),
                get_sig_symbol(significant)
            ))
    
    # Output rows
    for row in rows:
        output_lines.append(f"          {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\\n")
    
    output_lines.append(f"          \\bottomrule\n")
    output_lines.append(f"        \\end{{tabular}}\n")
    output_lines.append(f"      \\end{{sc}}\n")
    output_lines.append(f"    \\end{{small}}\n")
    output_lines.append(f"  \\end{{center}}\n")
    output_lines.append(f"  \\vskip -0.1in\n")
    output_lines.append(f"\\end{{table}}\n")
    output_lines.append(f"\n")

def generate_pairwise_table(data, metric_name, metric_key, tune, dataset, output_lines):
    """Generate pairwise test table for a specific dataset"""
    tune_label = "tuned" if tune else "untuned"
    metric_label = "ROC-AUC" if metric_key == "clean_roc_auc" else "AUPC"
    
    # Get dataset display name
    dataset_display = normalize_dataset_name(dataset)
    
    # Collect all rows FIRST before writing table structure
    rows = []
    for eval_mode in ["CrossSession", "WithinSession", "CrossSubject"]:
        if eval_mode not in data[dataset]:
            continue
        tune_key = str(tune)
        if tune_key not in data[dataset][eval_mode]:
            continue
        if metric_key not in data[dataset][eval_mode][tune_key]:
            continue
        
        # Check if omnibus test was significant - only include pairwise if it was
        omnibus = data[dataset][eval_mode][tune_key][metric_key].get("omnibus", {})
        omnibus_significant = omnibus.get("significant", False)
        
        # Skip pairwise comparisons if omnibus was not significant
        if not omnibus_significant:
            continue
        
        pairwise_list = data[dataset][eval_mode][tune_key][metric_key].get("pairwise", [])
        
        # Filter to only comparisons involving HYDRA and reorder so HYDRA is model1
        hydra_comparisons = []
        for comp in pairwise_list:
            comparison = comp.get("comparison", "")
            if "HYDRA" in comparison:
                # Swap if needed to make HYDRA model1
                swapped_comp, swapped_dz, swapped_ci = swap_pairwise_comparison(
                    comparison,
                    comp.get("cohens_dz", 0),
                    comp.get("cohens_dz_ci", [0, 0])
                )
                
                # Only include if HYDRA is now model1
                if swapped_comp.startswith("HYDRA vs"):
                    hydra_comparisons.append({
                        "comparison": swapped_comp,
                        "test_type": comp.get("test_type", ""),
                        "p_value": comp.get("p_value", 1.0),
                        "p_adj": comp.get("p_adj", 1.0),
                        "cohens_dz": swapped_dz,
                        "cohens_dz_ci": swapped_ci,
                        "significant": comp.get("significant", False)
                    })
        
        # Sort comparisons alphabetically by model2
        hydra_comparisons.sort(key=lambda x: x["comparison"])
        
        eval_display = normalize_eval_mode(eval_mode)
        
        for comp in hydra_comparisons:
            model2_raw = comp["comparison"].replace("HYDRA vs ", "")
            model2 = normalize_model_name(model2_raw)
            test_type = comp["test_type"].replace("_", "-").title()
            p_val = format_p_value(comp["p_value"])
            p_adj = format_p_value(comp["p_adj"])
            dz = format_cohens_dz(comp["cohens_dz"])
            ci = format_ci(comp["cohens_dz_ci"])
            
            rows.append((
                eval_display,
                f"HYDRA vs {model2}",
                test_type,
                p_val,
                p_adj,
                dz,
                ci
            ))
    
    # Only output table if there are rows
    if rows:
        output_lines.append(f"% =========================\n")
        output_lines.append(f"% Pairwise ({metric_label}, {tune_label}, {dataset_display})\n")
        output_lines.append(f"% =========================\n")
        output_lines.append(f"\\begin{{table}}[t]\n")
        output_lines.append(f"\\caption{{\\textbf{{{metric_label} Pairwise Tests ({dataset_display}, {tune_label}).}} Pairwise statistical tests comparing HYDRA to other models (CNN-NCPv2, EEGNet, REEGNet) on subject-level collapsed {metric_label}. Tests use paired t-test or Wilcoxon signed-rank test; effect sizes are reported using Cohen's $d_z$ computed on paired differences, with 95\\% confidence intervals.}}\n")
        output_lines.append(f"\\label{{tab:stats_pairwise_{metric_key}_{dataset.lower().replace('_', '')}_{tune_label}}}\n")
        output_lines.append(f"  \\begin{{center}}\n")
        output_lines.append(f"    \\begin{{small}}\n")
        output_lines.append(f"      \\begin{{sc}}\n")
        output_lines.append(f"        \\begin{{tabular}}{{llccccc}}\n")
        output_lines.append(f"          \\toprule\n")
        output_lines.append(f"          Evaluation & Comparison & Test & $p$ & $p_{{adj}}$ & $d_z$ & 95\\% CI \\\\\n")
        output_lines.append(f"          \\midrule\n")
        
        # Output rows
        for row in rows:
            output_lines.append(f"          {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\\n")
        
        output_lines.append(f"          \\bottomrule\n")
        output_lines.append(f"        \\end{{tabular}}\n")
        output_lines.append(f"      \\end{{sc}}\n")
        output_lines.append(f"    \\end{{small}}\n")
        output_lines.append(f"  \\end{{center}}\n")
        output_lines.append(f"  \\vskip -0.1in\n")
        output_lines.append(f"\\end{{table}}\n")
        output_lines.append(f"\n")
    else:
        # If no rows, don't output the table at all
        pass

def main():
    # Load the JSON data
    json_path = Path(r"E:\Research\Dissertation\full_backup_7_16_2025\moabb_experiments\analysis\analysis\statistical_results\hydra\stats_summary.json")
    output_dir = Path(r"E:\Research\Dissertation\full_backup_7_16_2025\moabb_experiments\analysis\analysis\hydra_performance_summary")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Generate tables
    output_lines = []
    
    # 1. Clean ROC-AUC Omnibus tests (untuned)
    generate_omnibus_table(data, "Clean ROC-AUC", "clean_roc_auc", False, output_lines)
    
    # 2. Clean ROC-AUC Omnibus tests (tuned)
    generate_omnibus_table(data, "Clean ROC-AUC", "clean_roc_auc", True, output_lines)
    
    # 3. Clean ROC-AUC Pairwise tests (untuned) - split by dataset
    for dataset in ["BI2015a", "BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP"]:
        if dataset in data:
            generate_pairwise_table(data, "Clean ROC-AUC", "clean_roc_auc", False, dataset, output_lines)
    
    # 4. Clean ROC-AUC Pairwise tests (tuned) - split by dataset
    for dataset in ["BI2015a", "BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP"]:
        if dataset in data:
            generate_pairwise_table(data, "Clean ROC-AUC", "clean_roc_auc", True, dataset, output_lines)
    
    # 5. AUPC Omnibus tests (untuned)
    generate_omnibus_table(data, "AUPC", "aupc_collapsed", False, output_lines)
    
    # 6. AUPC Omnibus tests (tuned)
    generate_omnibus_table(data, "AUPC", "aupc_collapsed", True, output_lines)
    
    # 7. AUPC Pairwise tests (untuned) - split by dataset
    for dataset in ["BI2015a", "BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP"]:
        if dataset in data:
            generate_pairwise_table(data, "AUPC", "aupc_collapsed", False, dataset, output_lines)
    
    # 8. AUPC Pairwise tests (tuned) - split by dataset
    for dataset in ["BI2015a", "BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP"]:
        if dataset in data:
            generate_pairwise_table(data, "AUPC", "aupc_collapsed", True, dataset, output_lines)
    
    # Write to file
    output_file = output_dir / "hydra_statistical_tables.txt"
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"LaTeX tables written to: {output_file}")

if __name__ == "__main__":
    main()
