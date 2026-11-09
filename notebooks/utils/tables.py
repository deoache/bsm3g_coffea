import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import hist
from .data import load_hists
from coffea.util import load
import copy
from coffea.processor import accumulate
from .data import load_hists, get_total_mc_uncertainty
from .fitting import find_optimal_tf_poly


def get_process_yield_and_errors(h_proc):
    """Computes the nominal yield, statistical uncertainty and global systematic uncertainty."""
    if "variation" in h_proc.axes.name:
        h_nom = h_proc[{"variation": "nominal"}]
    else:
        h_nom = h_proc

    val_nom = float(np.sum(h_nom.values()))

    if h_nom.variances() is not None:
        var_stat = float(np.sum(h_nom.variances()))
    else:
        var_stat = val_nom

    err2_syst_up = 0.0
    err2_syst_down = 0.0

    if "variation" in h_proc.axes.name:
        var_axis = h_proc.axes["variation"]
        variations = set()
        for name in var_axis:
            if name == "nominal":
                continue
            base_name = name.replace("Up", "").replace("Down", "")
            variations.add(base_name)

        for syst in variations:
            up_key = f"{syst}Up"
            down_key = f"{syst}Down"

            val_up = (
                float(np.sum(h_proc[{"variation": up_key}].values()))
                if up_key in var_axis
                else val_nom
            )
            val_down = (
                float(np.sum(h_proc[{"variation": down_key}].values()))
                if down_key in var_axis
                else val_nom
            )

            delta_up = val_up - val_nom
            delta_down = val_down - val_nom

            if delta_up * delta_down < 0:
                d_up = delta_up if delta_up > 0 else -delta_down
                d_down = -delta_down if delta_down < 0 else delta_up
            else:
                max_delta = max(abs(delta_up), abs(delta_down))
                d_up = max_delta if delta_up > 0 else 0.0
                d_down = max_delta if delta_down < 0 else 0.0

            err2_syst_up += d_up**2
            err2_syst_down += d_down**2

    err_syst_global = np.sqrt(max(err2_syst_up, err2_syst_down))

    return val_nom, np.sqrt(max(var_stat, 0.0)), err_syst_global


def generate_latex_table_qcd(out_signal, procs_order, year="2022preEE", sr_workflow="1b1mu"):
    """Generates the LaTeX table with the background yields (including QCD) and data."""
    main_key = "mass"
    yields = {}

    for p in procs_order:
        if p in out_signal:
            h_proc = out_signal[p][main_key] if isinstance(out_signal[p], dict) else out_signal[p]
            val, stat, syst = get_process_yield_and_errors(h_proc)
            yields[p] = {"val": val, "stat": stat, "syst": syst}

    total_val = sum(y["val"] for y in yields.values())
    total_stat = np.sqrt(sum(y["stat"] ** 2 for y in yields.values()))
    total_syst = np.sqrt(sum(y["syst"] ** 2 for y in yields.values()))

    latex_code = "\\begin{tabular}{@{} l c c @{}}\n"
    title_str = f"\\textbf{{{sr_workflow} (with QCD estimation) --- {year}}}"
    latex_code += f"\\multicolumn{{3}}{{@{{}} l @{{}}}}{{{title_str}}} \\\\[4pt]\n"
    latex_code += "\\hline\n"
    latex_code += " & \\textbf{Events $\\pm$ Stat $\\pm$ Syst} & \\textbf{Percentage} \\\\\n"
    latex_code += "\\hline\n"

    for p in procs_order:
        if p in yields:
            y = yields[p]
            label_name = r"TTBar" if p == "tt" else p
            pct = (y["val"] / total_val * 100.0) if total_val > 0 else 0.0

            val_str = f"${y['val']:.2f} \\pm {y['stat']:.2f} \\pm {y['syst']:.2f}$"
            pct_str = f"${pct:.1f}\\%$"

            latex_code += f"{label_name} & {val_str} & {pct_str} \\\\\n"

    latex_code += "\\hline\n"
    total_val_str = f"${total_val:.2f} \\pm {total_stat:.2f} \\pm {total_syst:.2f}$"
    latex_code += f"Total backg. & {total_val_str} & -- \\\\\n"

    if "Data" in out_signal:
        h_data = out_signal["Data"][main_key] if isinstance(out_signal["Data"], dict) else out_signal["Data"]
        data_val, _, _ = get_process_yield_and_errors(h_data)
        latex_code += f"Data & ${data_val:.0f}$ & -- \\\\\n"

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"

    return latex_code


def compile_latex_to_png(table_code, output_path):
    """Compiles the LaTeX code to PDF and converts it to PNG."""
    clean_table_code = re.sub(r"\\begin\{table\}(\[.*?\])?", "", table_code)
    clean_table_code = re.sub(r"\\end\{table\}", "", clean_table_code)

    tex_template = f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage[active,tightpage]{{preview}}

\\paperwidth=28cm
\\pdfpagewidth=\\paperwidth
\\paperheight=15cm
\\pdfpageheight=\\paperheight

\\setlength\\PreviewBorder{{10pt}}

\\begin{{document}}
\\begin{{preview}}
\\resizebox{{\\linewidth}}{{!}}{{
{clean_table_code}
}}
\\end{{preview}}
\\end{{document}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "table.tex")
        pdf_file = os.path.join(tmpdir, "table.pdf")
        png_prefix = os.path.join(tmpdir, "table_out")

        with open(tex_file, "w") as f:
            f.write(tex_template)

        proc = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                tmpdir,
                tex_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            print("  ❌ LaTeX compilation error:")
            log_file = os.path.join(tmpdir, "table.log")
            if os.path.exists(log_file):
                with open(log_file, "r") as log:
                    for line in log:
                        if line.startswith("!"):
                            print(f"     {line.strip()}")
            return

        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", pdf_file, png_prefix],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            generated_png = f"{png_prefix}-1.png"
            if os.path.exists(generated_png):
                shutil.move(generated_png, output_path)
                print(f"  ✓ PNG saved successfully: {output_path}")

        except Exception as e:
            print(f"  ❌ Error converting PDF to PNG: {e}")


def generate_qcd_tables_and_pngs(qcd_store, years, plots_dir=None):
    """Processes each SR, injects the estimated QCD background and generates the .txt tables and .png images."""
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/qcd_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    signal_configs = {
        "1b1e": {"qcd_key": "qcd_ele"},
        "1b1mu": {"qcd_key": "qcd_mu"},
    }
    procs_order = ["tt", "WJetsToLNu", "SingleTop", "DYJetsToLL", "Diboson", "QCD"]
    output_filename = plots_dir / "outputs_qcd_estimation.txt"

    with open(output_filename, "w") as f:
        for sr_workflow, config in signal_configs.items():
            qcd_key = config["qcd_key"]

            for yr in years:
                print(f"Processing table: SR={sr_workflow} | Year={yr}...")
                out_signal = load_hists(yr, sr_workflow)
                
                # Inject QCD into out_signal
                any_proc = next(p for p in out_signal.keys() if p != "Data")
                h_template = out_signal[any_proc]["mass"]
                
                if "category" in h_template.axes.name and sr_workflow in h_template.axes["category"]:
                    h_template_1d = h_template[{"category": sr_workflow}]
                else:
                    h_template_1d = h_template

                if "variation" in h_template_1d.axes.name:
                    h_template_1d = h_template_1d[{"variation": "nominal"}]

                var_axis_name = [ax.name for ax in h_template_1d.axes][0]
                if var_axis_name in h_template_1d.axes.name:
                    h_template_1d = h_template_1d.project(var_axis_name)

                h_qcd_year = qcd_store[qcd_key][yr]
                h_qcd_inj = hist.Hist(*h_template_1d.axes, storage=hist.storage.Weight())
                h_qcd_inj.values()[:] = h_qcd_year.values()[:]
                h_qcd_inj.variances()[:] = h_qcd_year.variances()[:]

                out_signal["QCD"] = {"mass": h_qcd_inj}

                table_code = generate_latex_table_qcd(
                    out_signal=out_signal,
                    procs_order=procs_order,
                    year=yr,
                    sr_workflow=sr_workflow,
                )

                f.write(f"% ==========================================\n")
                f.write(f"% SR: {sr_workflow} | Year: {yr}\n")
                f.write(f"% ==========================================\n\n")
                f.write(table_code)
                f.write("\n\n")

                table_filename = f"table_{sr_workflow}_{yr}_qcd.png"
                png_path = plots_dir / table_filename
                compile_latex_to_png(table_code, str(png_path))

    print(f"\nTables saved to '{output_filename}' and PNGs to '{plots_dir}'.")



def fmt_sci(val, decimals=2):
    """Converts a float to stylized scientific notation for LaTeX."""
    if val == 0:
        return "0"
    s = f"{val:.{decimals}e}"
    base, exp = s.split("e")
    return f"{base} \\times 10^{{{int(exp)}}}"


def get_process_yield_and_errors(h_proc):
    """Computes the nominal yield, statistical uncertainty and global systematic uncertainty."""
    if "variation" in h_proc.axes.name:
        h_nom = h_proc[{"variation": "nominal"}]
    else:
        h_nom = h_proc

    val_nom = float(np.sum(h_nom.values()))

    if h_nom.variances() is not None:
        var_stat = float(np.sum(h_nom.variances()))
    else:
        var_stat = val_nom

    err2_syst_up = 0.0
    err2_syst_down = 0.0

    if "variation" in h_proc.axes.name:
        var_axis = h_proc.axes["variation"]
        variations = set()
        for name in var_axis:
            if name == "nominal":
                continue
            base_name = name.replace("Up", "").replace("Down", "")
            variations.add(base_name)

        for syst in variations:
            up_key = f"{syst}Up"
            down_key = f"{syst}Down"

            val_up = (
                float(np.sum(h_proc[{"variation": up_key}].values()))
                if up_key in var_axis
                else val_nom
            )
            val_down = (
                float(np.sum(h_proc[{"variation": down_key}].values()))
                if down_key in var_axis
                else val_nom
            )

            delta_up = val_up - val_nom
            delta_down = val_down - val_nom

            if delta_up * delta_down < 0:
                d_up = delta_up if delta_up > 0 else -delta_down
                d_down = -delta_down if delta_down < 0 else delta_up
            else:
                max_delta = max(abs(delta_up), abs(delta_down))
                d_up = max_delta if delta_up > 0 else 0.0
                d_down = max_delta if delta_down < 0 else 0.0

            err2_syst_up += d_up**2
            err2_syst_down += d_down**2

    err_syst_global = np.sqrt(max(err2_syst_up, err2_syst_down))

    return val_nom, np.sqrt(max(var_stat, 0.0)), err_syst_global


def generate_latex_table_ttbar(out, procs_order, year="2022preEE", workflow="2b1mu"):
    """Generates the ttbar estimation LaTeX table with yields, ratios and mu_ttbar."""
    yields = {}

    for p in procs_order:
        if p in out:
            h_proc = out[p]["mass"] if isinstance(out[p], dict) else out[p]
            val, stat, syst = get_process_yield_and_errors(h_proc)
            yields[p] = {"val": val, "stat": stat, "syst": syst}

    h_data = (
        out["Data"]["mass"][{"variation": "nominal"}]
        if "variation" in out["Data"]["mass"].axes.name
        else out["Data"]["mass"]
    )
    data_val = float(np.round(np.sum(h_data.values())))
    data_stat = np.sqrt(data_val)

    total_val = sum(y["val"] for y in yields.values())
    total_stat = np.sqrt(sum(y["stat"] ** 2 for y in yields.values()))
    total_syst = np.sqrt(sum(y["syst"] ** 2 for y in yields.values()))

    non_tt_yields = [y for p, y in yields.items() if p != "tt"]
    non_tt_val = sum(y["val"] for y in non_tt_yields)
    non_tt_stat = np.sqrt(sum(y["stat"] ** 2 for y in non_tt_yields))
    non_tt_syst = np.sqrt(sum(y["syst"] ** 2 for y in non_tt_yields))

    diff_val = data_val - non_tt_val
    diff_stat = np.sqrt(data_stat**2 + non_tt_stat**2)
    diff_syst = non_tt_syst

    if total_val > 0 and data_val > 0:
        ratio_val = data_val / total_val
        ratio_stat = ratio_val * np.sqrt(
            (data_stat / data_val) ** 2 + (total_stat / total_val) ** 2
        )
        ratio_syst = ratio_val * (total_syst / total_val)
    else:
        ratio_val, ratio_stat, ratio_syst = 0.0, 0.0, 0.0

    tt_val = yields.get("tt", {}).get("val", 0.0)
    tt_stat = yields.get("tt", {}).get("stat", 0.0)
    tt_syst = yields.get("tt", {}).get("syst", 0.0)

    if tt_val > 0 and diff_val > 0:
        mu_tt = diff_val / tt_val
        mu_stat = mu_tt * np.sqrt(
            (diff_stat / diff_val) ** 2 + (tt_stat / tt_val) ** 2
        )
        mu_syst = mu_tt * np.sqrt(
            (diff_syst / diff_val) ** 2 + (tt_syst / tt_val) ** 2
        )
    else:
        mu_tt, mu_stat, mu_syst = 0.0, 0.0, 0.0

    latex_code = "\\begin{table}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{@{} l c c @{}}\n"

    latex_code += f"\\textbf{{{workflow}}} & & \\\\\n"
    latex_code += f"\\textbf{{{year}}} & & \\\\[4pt]\n"

    latex_code += "\\hline\n"
    latex_code += " & \\textbf{Events $\\pm$ Stat $\\pm$ Syst} & \\textbf{Percentage} \\\\\n"
    latex_code += "\\hline\n"

    for p in procs_order:
        if p in yields:
            y = yields[p]
            label_name = r"TTBar" if p == "tt" else p
            pct = (y["val"] / total_val * 100.0) if total_val > 0 else 0.0
            latex_code += f"{label_name} & ${y['val']:.2f} \\pm {y['stat']:.2f} \\pm {y['syst']:.2f}$ & ${pct:.2f}\\%$ \\\\\n"

    latex_code += "\\hline\n"
    pct_non_tt = (non_tt_val / total_val * 100.0) if total_val > 0 else 0.0

    latex_code += f"Total background & ${total_val:.2f} \\pm {total_stat:.2f} \\pm {total_syst:.2f}$ & -- \\\\\n"
    latex_code += f"Data & ${int(data_val)} \\pm {data_stat:.2f}$ & -- \\\\\n"
    latex_code += f"Data / Total background & ${ratio_val:.2f} \\pm {fmt_sci(ratio_stat, 2)} \\pm {ratio_syst:.2f}$ & -- \\\\\n"
    latex_code += "\\hline\n"
    latex_code += f"Non-$t\\bar{{t}}$ background & ${non_tt_val:.2f} \\pm {non_tt_stat:.2f} \\pm {non_tt_syst:.2f}$ & ${pct_non_tt:.2f}\\%$ \\\\\n"
    latex_code += f"Data $-$ non-$t\\bar{{t}}$ background & ${diff_val:.2f} \\pm {diff_stat:.2f} \\pm {diff_syst:.2f}$ & -- \\\\\n"
    latex_code += "\\hline\n"
    latex_code += f"$r_{{t\\bar{{t}}}}$ & ${mu_tt:.2f} \\pm {fmt_sci(mu_stat, 2)} \\pm {mu_syst:.2f}$ & -- \\\\\n"
    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\end{table}"

    return latex_code


def compile_latex_to_png(table_code, output_path):
    """Compiles the table in 'article' with dynamic canvas adjustment to remove cropping and empty margins."""
    clean_table_code = re.sub(
        r"\\begin\{table\}(\[.*?\])?", "", table_code
    )
    clean_table_code = re.sub(r"\\end\{table\}", "", clean_table_code)

    tex_template = f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage[active,tightpage]{{preview}}

% Expand the text width capacity without fixing the physical page size in the PDF
\\setlength{{\\textwidth}}{{100cm}}
\\setlength{{\\hsize}}{{100cm}}

\\setlength\\PreviewBorder{{10pt}}

\\begin{{document}}
\\begin{{preview}}
\\hbox{{%
{clean_table_code}
}}
\\end{{preview}}
\\end{{document}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "table.tex")
        pdf_file = os.path.join(tmpdir, "table.pdf")
        png_prefix = os.path.join(tmpdir, "table_out")

        with open(tex_file, "w") as f:
            f.write(tex_template)

        proc = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                tmpdir,
                tex_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            print("  ❌ LaTeX compilation error:")
            log_file = os.path.join(tmpdir, "table.log")
            if os.path.exists(log_file):
                with open(log_file, "r") as log:
                    for line in log:
                        if line.startswith("!"):
                            print(f"     {line.strip()}")
            return

        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", pdf_file, png_prefix],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            generated_png = f"{png_prefix}-1.png"
            if os.path.exists(generated_png):
                shutil.move(generated_png, output_path)
                print(f"  ✓ PNG saved successfully: {output_path}")

        except Exception as e:
            print(f"  ❌ Error converting PDF to PNG: {e}")


def generate_ttbar_tables_and_pngs(workflows, years, procs, plots_dir=None):
    """Processes each ttbar workflow/year, generates the LaTeX .txt tables and compiles the PNG images."""
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/ttbar_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_filename = plots_dir / "outputs_ttbar_agreement.txt"

    with open(output_filename, "w") as f:
        for wf in workflows:
            for yr in years:
                print(f"Processing ttbar table: {wf} - {yr}...")
                out = load_hists(yr, wf)
                table_code = generate_latex_table_ttbar(
                    out, procs_order=procs, year=yr, workflow=wf
                )

                f.write(f"% ==========================================\n")
                f.write(f"% Workflow: {wf} | Year: {yr}\n")
                f.write(f"% ==========================================\n\n")
                f.write(table_code)
                f.write("\n\n")

                png_path = plots_dir / f"ttbar_agreement_{wf}_{yr}_table.png"
                compile_latex_to_png(table_code, str(png_path))

    print(f"\nTables saved to '{output_filename}' and images to '{plots_dir}'.")



def generate_latex_table_corrected(
    out_signal,
    optimal_coefs,
    procs_order,
    year="2022preEE",
    sr_workflow="1b1mu",
    cr_workflow="2b1mu",
):
    """Generates the LaTeX table comparing uncorrected and corrected ttbar yields."""
    channel = "muon" if "mu" in sr_workflow else "electron"
    variable_axis = f"{channel}_met_mass"
    category_signal = f"{sr_workflow} events"
    main_key = "mass"

    yields_uncorrected = {}
    yields_corrected = {}

    for p in procs_order:
        if p in out_signal:
            h_proc = out_signal[p][main_key] if isinstance(out_signal[p], dict) else out_signal[p]
            val, stat, syst = get_process_yield_and_errors(h_proc)
            yields_uncorrected[p] = {"val": val, "stat": stat, "syst": syst}

    out_signal_corr = copy.deepcopy(out_signal)
    h_tt_sr = (
        out_signal_corr["tt"][main_key]
        if isinstance(out_signal_corr["tt"], dict)
        else out_signal_corr["tt"]
    )

    proj_tmp = (
        h_tt_sr[{"variation": "nominal"}]
        if "variation" in h_tt_sr.axes.name
        else h_tt_sr
    )
    if "category" in proj_tmp.axes.name:
        cat_val = (
            category_signal
            if category_signal in proj_tmp.axes["category"]
            else 0
        )
        proj_tmp = proj_tmp[{"category": cat_val}]
    proj_tmp = proj_tmp.project(variable_axis)

    centers_sr = proj_tmp.axes[0].centers
    sf_bin = np.polyval(optimal_coefs, centers_sr)
    sf_bin = np.clip(sf_bin, 0.01, 5.0)

    axis_idx = h_tt_sr.axes.name.index(variable_axis)
    slicer = [slice(None)] * len(h_tt_sr.axes)
    reshape_pattern = [1] * len(h_tt_sr.axes)
    reshape_pattern[axis_idx] = len(sf_bin)
    sf_reshaped = sf_bin.reshape(reshape_pattern)

    view = h_tt_sr.view(flow=False)
    view["value"][tuple(slicer)] *= sf_reshaped
    view["variance"][tuple(slicer)] *= sf_reshaped**2

    for p in procs_order:
        if p in out_signal_corr:
            h_proc = (
                out_signal_corr[p][main_key]
                if isinstance(out_signal_corr[p], dict)
                else out_signal_corr[p]
            )
            val, stat, syst = get_process_yield_and_errors(h_proc)
            yields_corrected[p] = {"val": val, "stat": stat, "syst": syst}

    total_val_uncorr = sum(y["val"] for y in yields_uncorrected.values())
    total_stat_uncorr = np.sqrt(
        sum(y["stat"] ** 2 for y in yields_uncorrected.values())
    )
    total_syst_uncorr = np.sqrt(
        sum(y["syst"] ** 2 for y in yields_uncorrected.values())
    )

    total_val_corr = sum(y["val"] for y in yields_corrected.values())
    total_stat_corr = np.sqrt(
        sum(y["stat"] ** 2 for y in yields_corrected.values())
    )
    total_syst_corr = np.sqrt(
        sum(y["syst"] ** 2 for y in yields_corrected.values())
    )

    latex_code = "\\begin{tabular}{@{} l c c @{}}\n"
    title_str = f"\\textbf{{{sr_workflow} (Corrected via {cr_workflow}) --- {year}}}"
    latex_code += f"\\multicolumn{{3}}{{@{{}} l @{{}}}}{{{title_str}}} \\\\[4pt]\n"
    latex_code += "\\hline\n"
    latex_code += " & \\textbf{Events $\\pm$ Stat $\\pm$ Syst} & \\textbf{Percentage} \\\\\n"
    latex_code += "\\hline\n"

    for p in procs_order:
        if p in yields_corrected:
            y_corr = yields_corrected[p]
            y_uncorr = yields_uncorrected[p]
            label_name = r"TTBar" if p == "tt" else p

            pct_corr = (
                (y_corr["val"] / total_val_corr * 100.0)
                if total_val_corr > 0
                else 0.0
            )
            pct_uncorr = (
                (y_uncorr["val"] / total_val_uncorr * 100.0)
                if total_val_uncorr > 0
                else 0.0
            )

            if p == "tt":
                val_str = (
                    f"${y_corr['val']:.2f}({y_uncorr['val']:.2f}) \\pm "
                    f"{y_corr['stat']:.2f}({y_uncorr['stat']:.2f}) \\pm "
                    f"{y_corr['syst']:.2f}({y_uncorr['syst']:.2f})$"
                )
                pct_str = f"${pct_corr:.1f}\\%({pct_uncorr:.1f}\\%)$"
            else:
                val_str = f"${y_corr['val']:.2f} \\pm {y_corr['stat']:.2f} \\pm {y_corr['syst']:.2f}$"
                pct_str = f"${pct_corr:.1f}\\%({pct_uncorr:.1f}\\%)$"

            latex_code += f"{label_name} & {val_str} & {pct_str} \\\\\n"

    latex_code += "\\hline\n"
    total_val_str = (
        f"${total_val_corr:.2f}({total_val_uncorr:.2f}) \\pm "
        f"{total_stat_corr:.2f}({total_stat_uncorr:.2f}) \\pm "
        f"{total_syst_corr:.2f}({total_syst_uncorr:.2f})$"
    )
    latex_code += f"Total backg. & {total_val_str} & -- \\\\\n"
    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"

    return latex_code


def generate_ttbar_corrected_tables_and_pngs(sr_to_crs, years, procs, plots_dir=None):
    """Computes the polynomial fit in the CR, scales the SR, generates the LaTeX tables and compiles the PNGs."""
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/ttbar_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    main_key = "mass"
    output_filename = plots_dir / "outputs_corrected_mc.txt"

    with open(output_filename, "w") as f:
        for sr_workflow, cr_workflows in sr_to_crs.items():
            channel = "muon" if "mu" in sr_workflow else "electron"
            variable_axis = f"{channel}_met_mass"

            for cr_workflow in cr_workflows:
                for yr in years:
                    print(
                        f"Processing corrected table: SR={sr_workflow} | CR={cr_workflow} | Year={yr}..."
                    )

                    out_cr = load_hists(yr, cr_workflow)
                    mc_non_tt_cr = accumulate(
                        [
                            out_cr[p][main_key]
                            for p in procs
                            if p in out_cr and p != "tt"
                        ]
                    )
                    data_cr_nom = out_cr["Data"][main_key][
                        {"variation": "nominal"}
                    ].project(variable_axis)

                    edges_cr = data_cr_nom.axes[0].edges
                    centers_cr = 0.5 * (edges_cr[:-1] + edges_cr[1:])

                    data_vals_cr = data_cr_nom.values()
                    vals_non_tt_cr, var_non_tt_cr = get_total_mc_uncertainty(
                        mc_non_tt_cr, target_axis=variable_axis
                    )
                    vals_tt_cr, var_tt_cr = get_total_mc_uncertainty(
                        out_cr["tt"][main_key], target_axis=variable_axis
                    )

                    D_cr = data_vals_cr - vals_non_tt_cr
                    M_cr = vals_tt_cr + 1e-32
                    accpt_cr = D_cr / M_cr

                    var_D_cr = (
                        data_cr_nom.variances()
                        if data_cr_nom.variances() is not None
                        else data_vals_cr
                    ) + var_non_tt_cr
                    accpt_var_cr = (var_D_cr / (M_cr**2)) + (
                        (D_cr**2 * var_tt_cr) / (M_cr**4)
                    )
                    yerr_accpt_cr = np.sqrt(np.maximum(accpt_var_cr, 0))

                    valid_mask = (
                        (M_cr > 1e-5)
                        & (yerr_accpt_cr > 0)
                        & ~np.isnan(accpt_cr)
                        & ~np.isinf(accpt_cr)
                    )

                    best_degree, optimal_coefs, _ = find_optimal_tf_poly(
                        centers_cr[valid_mask],
                        accpt_cr[valid_mask],
                        yerr_accpt_cr[valid_mask],
                        max_degree=5,
                        alpha=0.05,
                    )

                    out_signal = load_hists(yr, sr_workflow)

                    table_code = generate_latex_table_corrected(
                        out_signal=out_signal,
                        optimal_coefs=optimal_coefs,
                        procs_order=procs,
                        year=yr,
                        sr_workflow=sr_workflow,
                        cr_workflow=cr_workflow,
                    )

                    f.write(f"% ==========================================\n")
                    f.write(
                        f"% SR: {sr_workflow} | CR: {cr_workflow} | Year: {yr}\n"
                    )
                    f.write(f"% ==========================================\n\n")
                    f.write(table_code)
                    f.write("\n\n")

                    table_filename = f"{sr_workflow}_{variable_axis}_ttbar_corrected_via_{cr_workflow}_{yr}_table.png"
                    png_path = plots_dir / table_filename

                    compile_latex_to_png(table_code, str(png_path))

    print(
        f"\nTables saved to '{output_filename}' and PNG plots to '{plots_dir}'."
    )

def generate_latex_table_qcd_tf(
    out, procs_order, category, year="2022preEE", workflow="qcd_tf_mu"
):
  """Generates the LaTeX table for a specific region of the qcd_tf_* workflows."""
  main_key = "transverse_mass"
  yields = {}

  for p in procs_order:
    if p in out:
      h_proc = out[p][main_key] if isinstance(out[p], dict) else out[p]
      if "category" in h_proc.axes.name:
        h_proc = h_proc[{"category": category}]
      val, stat, syst = get_process_yield_and_errors(h_proc)
      yields[p] = {"val": val, "stat": stat, "syst": syst}

  total_val = sum(y["val"] for y in yields.values())
  total_stat = np.sqrt(sum(y["stat"] ** 2 for y in yields.values()))
  total_syst = np.sqrt(sum(y["syst"] ** 2 for y in yields.values()))

  # Escape underscores for LaTeX
  workflow_latex = workflow.replace("_", r"\_")
  category_latex = category.replace("_", r"\_")

  latex_code = "\\begin{tabular}{@{} l c c @{}}\n"
  title_str = f"\\textbf{{{category_latex} ({workflow_latex}) --- {year}}}"
  latex_code += f"\\multicolumn{{3}}{{@{{}} l @{{}}}}{{{title_str}}} \\\\[4pt]\n"
  latex_code += "\\hline\n"
  latex_code += (
      " & \\textbf{Events $\\pm$ Stat $\\pm$ Syst} & \\textbf{Percentage}"
      " \\\\\n"
  )
  latex_code += "\\hline\n"

  for p in procs_order:
    if p in yields:
      y = yields[p]
      label_name = r"TTBar" if p == "tt" else p
      label_name = label_name.replace("_", r"\_")

      pct = (y["val"] / total_val * 100.0) if total_val > 0 else 0.0

      val_str = f"${y['val']:.2f} \\pm {y['stat']:.2f} \\pm {y['syst']:.2f}$"
      pct_str = f"${pct:.1f}\\%$"

      latex_code += f"{label_name} & {val_str} & {pct_str} \\\\\n"

  latex_code += "\\hline\n"
  total_val_str = (
      f"${total_val:.2f} \\pm {total_stat:.2f} \\pm {total_syst:.2f}$"
  )
  latex_code += f"Total backg. & {total_val_str} & -- \\\\\n"

  # 1. Data yield and error
  data_val = 0.0
  data_stat = 0.0
  if "Data" in out:
    h_data = (
        out["Data"][main_key]
        if isinstance(out["Data"], dict)
        else out["Data"]
    )
    if "category" in h_data.axes.name:
      h_data = h_data[{"category": category}]
    data_val, _, _ = get_process_yield_and_errors(h_data)
    data_stat = np.sqrt(data_val)
    latex_code += f"Data & ${data_val:.0f}$ & -- \\\\\n"

  # 2. Data / bkg ratio (with propagated errors)
  if total_val > 0 and data_val > 0:
    ratio_val = data_val / total_val
    ratio_stat = ratio_val * np.sqrt(
        (data_stat / data_val) ** 2 + (total_stat / total_val) ** 2
    )
    ratio_syst = ratio_val * (total_syst / total_val)
    ratio_str = f"${ratio_val:.2f} \\pm {ratio_stat:.2f} \\pm {ratio_syst:.2f}$"
  else:
    ratio_str = "--"

  latex_code += f"Data / bkg & {ratio_str} & -- \\\\\n"

  # 3. Data - bkg computation (excluding WJets)
  wjets_procs = [p for p in yields if p.startswith("WJets")]
  bg_no_wjets_procs = [p for p in yields if not p.startswith("WJets")]

  bg_no_wjets_val = sum(yields[p]["val"] for p in bg_no_wjets_procs)
  bg_no_wjets_stat = np.sqrt(
      sum(yields[p]["stat"] ** 2 for p in bg_no_wjets_procs)
  )
  bg_no_wjets_syst = np.sqrt(
      sum(yields[p]["syst"] ** 2 for p in bg_no_wjets_procs)
  )

  diff_val = data_val - bg_no_wjets_val
  diff_stat = np.sqrt(data_stat**2 + bg_no_wjets_stat**2)
  diff_syst = bg_no_wjets_syst

  excluded_label = ", ".join(wjets_procs) if wjets_procs else "WJetsToLNu"
  excluded_label_latex = excluded_label.replace("_", r"\_")

  diff_str = f"${diff_val:.2f} \\pm {diff_stat:.2f} \\pm {diff_syst:.2f}$"
  latex_code += f"Data $-$ bkg (w/o {excluded_label_latex}) & {diff_str} & -- \\\\\n"

  latex_code += "\\hline\n"
  latex_code += "\\end{tabular}\n"

  return latex_code


def generate_qcd_tf_tables_and_pngs(
    years,
    workflows=("qcd_tf_mu", "qcd_tf_ele"),
    procs_order=None,
    plots_dir=None,
):
  """Loads the qcd_tf_mu and qcd_tf_ele histograms, generates the LaTeX tables per region/year and compiles the PNGs."""
  if plots_dir is None:
    plots_dir = Path("notebooks/plots/qcd_estimation")
  plots_dir = Path(plots_dir)
  plots_dir.mkdir(parents=True, exist_ok=True)

  if procs_order is None:
    procs_order = ["tt", "WJetsToLNu", "SingleTop", "DYJetsToLL", "Diboson"]

  # Regions assigned to each workflow
  tf_regions = {
      "qcd_tf_mu": ["2T", "1T1LnT"],
      "qcd_tf_ele": ["1LnT1T", "1LnT1nT"],
  }

  output_filename = plots_dir / "outputs_qcd_tf_tables.txt"

  with open(output_filename, "w") as f:
    for wf in workflows:
      regions = tf_regions.get(wf, [])
      for yr in years:
        print(f"Loading histograms for {wf} | Year={yr}...")
        out = load_hists(yr, wf)

        for region in regions:
          print(
              f"  Processing table: Region={region} | Workflow={wf} |"
              f" Year={yr}..."
          )
          table_code = generate_latex_table_qcd_tf(
              out=out,
              procs_order=procs_order,
              category=region,
              year=yr,
              workflow=wf,
          )

          f.write("% ==========================================\n")
          f.write(f"% Workflow: {wf} | Region: {region} | Year: {yr}\n")
          f.write("% ==========================================\n\n")
          f.write(table_code)
          f.write("\n\n")

          png_filename = f"table_{wf}_{region}_{yr}.png"
          png_path = plots_dir / png_filename
          compile_latex_to_png(table_code, str(png_path))

  print(
      f"\nTables saved to '{output_filename}' and PNG images to"
      f" '{plots_dir}'."
  )

def generate_latex_table_qcd_cr(
    out,
    procs_order,
    category="base",
    year="2022preEE",
    workflow="qcd_ele",
):
  """Generates the LaTeX table for the 'base' control region of the qcd_ele and qcd_mu workflows."""
  main_key = "mass"
  yields = {}

  for p in procs_order:
    if p in out:
      h_proc = out[p][main_key] if isinstance(out[p], dict) else out[p]
      # Filter by category if the axis exists in the histogram
      if "category" in h_proc.axes.name:
        h_proc = h_proc[{"category": category}]
      val, stat, syst = get_process_yield_and_errors(h_proc)
      yields[p] = {"val": val, "stat": stat, "syst": syst}

  total_val = sum(y["val"] for y in yields.values())
  total_stat = np.sqrt(sum(y["stat"] ** 2 for y in yields.values()))
  total_syst = np.sqrt(sum(y["syst"] ** 2 for y in yields.values()))

  # Escape underscores for LaTeX
  workflow_latex = workflow.replace("_", r"\_")
  category_latex = category.replace("_", r"\_")

  latex_code = "\\begin{tabular}{@{} l c c @{}}\n"
  title_str = f"\\textbf{{{category_latex} ({workflow_latex}) --- {year}}}"
  latex_code += f"\\multicolumn{{3}}{{@{{}} l @{{}}}}{{{title_str}}} \\\\[4pt]\n"
  latex_code += "\\hline\n"
  latex_code += (
      " & \\textbf{Events $\\pm$ Stat $\\pm$ Syst} & \\textbf{Percentage}"
      " \\\\\n"
  )
  latex_code += "\\hline\n"

  for p in procs_order:
    if p in yields:
      y = yields[p]
      label_name = r"TTBar" if p == "tt" else p
      label_name = label_name.replace("_", r"\_")

      pct = (y["val"] / total_val * 100.0) if total_val > 0 else 0.0

      val_str = f"${y['val']:.2f} \\pm {y['stat']:.2f} \\pm {y['syst']:.2f}$"
      pct_str = f"${pct:.1f}\\%$"

      latex_code += f"{label_name} & {val_str} & {pct_str} \\\\\n"

  latex_code += "\\hline\n"
  total_val_str = (
      f"${total_val:.2f} \\pm {total_stat:.2f} \\pm {total_syst:.2f}$"
  )
  latex_code += f"Total backg. & {total_val_str} & -- \\\\\n"

  # 1. Data yield and error
  data_val = 0.0
  data_stat = 0.0
  if "Data" in out:
    h_data = (
        out["Data"][main_key]
        if isinstance(out["Data"], dict)
        else out["Data"]
    )
    if "category" in h_data.axes.name:
      h_data = h_data[{"category": category}]
    data_val, _, _ = get_process_yield_and_errors(h_data)
    data_stat = np.sqrt(data_val)
    latex_code += f"Data & ${data_val:.0f}$ & -- \\\\\n"

  # 2. Data / bkg ratio
  if total_val > 0 and data_val > 0:
    ratio_val = data_val / total_val
    ratio_stat = ratio_val * np.sqrt(
        (data_stat / data_val) ** 2 + (total_stat / total_val) ** 2
    )
    ratio_syst = ratio_val * (total_syst / total_val)
    ratio_str = f"${ratio_val:.2f} \\pm {ratio_stat:.2f} \\pm {ratio_syst:.2f}$"
  else:
    ratio_str = "--"

  latex_code += f"Data / bkg & {ratio_str} & -- \\\\\n"

  # 3. Data - bkg subtraction (including ALL MC processes, incl. WJets)
  diff_val = data_val - total_val
  diff_stat = np.sqrt(data_stat**2 + total_stat**2)
  diff_syst = total_syst

  diff_str = f"${diff_val:.2f} \\pm {diff_stat:.2f} \\pm {diff_syst:.2f}$"
  latex_code += f"Data $-$ bkg & {diff_str} & -- \\\\\n"

  latex_code += "\\hline\n"
  latex_code += "\\end{tabular}\n"

  return latex_code


def generate_qcd_cr_tables_and_pngs(
    years,
    workflows=("qcd_ele", "qcd_mu"),
    procs_order=None,
    category="base",
    plots_dir=None,
):
  """Loads the qcd_ele and qcd_mu histograms for the 'base' region, generates the LaTeX table and compiles it to PNG."""
  if plots_dir is None:
    plots_dir = Path("notebooks/plots/qcd_estimation")
  plots_dir = Path(plots_dir)
  plots_dir.mkdir(parents=True, exist_ok=True)

  if procs_order is None:
    procs_order = ["tt", "WJetsToLNu", "SingleTop", "DYJetsToLL", "Diboson"]

  output_filename = plots_dir / "outputs_qcd_cr_tables.txt"

  with open(output_filename, "w") as f:
    for wf in workflows:
      for yr in years:
        print(f"Loading histograms for {wf} | Year={yr}...")
        out = load_hists(yr, wf)

        print(
            f"  Processing table: Region={category} | Workflow={wf} |"
            f" Year={yr}..."
        )
        table_code = generate_latex_table_qcd_cr(
            out=out,
            procs_order=procs_order,
            category=category,
            year=yr,
            workflow=wf,
        )

        f.write("% ==========================================\n")
        f.write(f"% Workflow: {wf} | Region: {category} | Year: {yr}\n")
        f.write("% ==========================================\n\n")
        f.write(table_code)
        f.write("\n\n")

        png_filename = f"table_{wf}_{category}_{yr}.png"
        png_path = plots_dir / png_filename
        compile_latex_to_png(table_code, str(png_path))

  print(
      f"\nTables saved to '{output_filename}' and PNG images to"
      f" '{plots_dir}'."
  )

def generate_latex_table_syst_diff(summary_data, bin_headers, workflow):
    """Generates LaTeX systematic difference table displaying per-bin deltas under a multi-column header and total deviation percentage."""
    num_bin_cols = len(bin_headers)
    num_total_cols = 2 + num_bin_cols
    col_spec = "@{} l " + "c " * num_bin_cols + "c @{}"

    latex_code = f"\\begin{{tabular}}{{{col_spec}}}\n"
    title_str = f"\\textbf{{Systematic Difference (Nominal vs Recalculated) --- {workflow}}}"
    latex_code += f"\\multicolumn{{{num_total_cols}}}{{@{{}} l @{{}}}}{{{title_str}}} \\\\[4pt]\n"
    latex_code += "\\hline\n"

    # Multi-column header spanning all energy intervals
    latex_code += f" & \\multicolumn{{{num_bin_cols}}}{{c}}{{\\textbf{{$\\delta_{{\\mathrm{{events}}}}$ per $m_T\\,(\\mathrm{{GeV}})$ bin}}}} & \\\\\n"

    # Sub-headers for years, intervals, and total deviation
    bin_cols_str = " & ".join([f"\\textbf{{{b}}}" for b in bin_headers])
    latex_code += f"\\textbf{{Year}} & {bin_cols_str} & \\textbf{{Dev. total}} \\\\\n"
    latex_code += "\\hline\n"

    for year, metrics in summary_data.items():
        bin_deltas = metrics["bin_deltas"]
        pct_bbb = metrics["pct_bbb"]

        bin_str = " & ".join([f"${d:+.2f}$" for d in bin_deltas])
        latex_code += f"{year} & {bin_str} & {pct_bbb:.2f}\\% \\\\\n"

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"

    return latex_code


def generate_systematic_diff_tables_and_pngs(
    workflows, years, plots_dir=None
):
    """Extracts bin-by-bin deltas across mass intervals, builds LaTeX tables, and compiles high-res PNGs."""
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/ttbar_estimation/syst")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    procs_non_tt = ["DYJetsToLL", "SingleTop", "Diboson", "WJetsToLNu"]
    config_mapping = {
        "2b1mu": ("muon_met_mass", "muon_met_mass_recalculated"),
        "2b1e": ("electron_met_mass", "electron_met_mass_recalculated"),
        "1b1mu1e": ("muon_met_mass", "muon_met_mass_recalculated"),
        "1b1e1mu": ("electron_met_mass", "electron_met_mass_recalculated"),
    }

    output_filename = plots_dir / "outputs_systematic_diff.txt"

    with open(output_filename, "w") as f:
        for workflow in workflows:
            var_nom, var_recalc = config_mapping[workflow]
            summary_data = {}
            bin_headers = []

            print(f"Processing systematic difference table for workflow: {workflow}...")

            for year in years:
                file_path = Path(
                    f"/eos/user/d/driosper/bsm3g_coffea_outputs/{workflow}/{year}/{year}_processed_histograms.coffea"
                )
                if not file_path.exists():
                    print(f"  Skipping {workflow} - {year} (file not found)")
                    continue

                out = load(file_path)

                # 1. Extract nominal distributions
                h_data_nom = out["Data"]["mass"]
                h_mc_non_tt_nom = accumulate(
                    [out[p]["mass"] for p in procs_non_tt if p in out]
                )
                d_vals_nom = (
                    h_data_nom[{"variation": "nominal"}].project(var_nom).values()
                )
                mc_non_tt_vals_nom = (
                    h_mc_non_tt_nom[{"variation": "nominal"}]
                    .project(var_nom)
                    .values()
                )
                diff_nom = d_vals_nom - mc_non_tt_vals_nom

                # 2. Extract recalculated distributions
                h_data_recalc = out["Data"]["mass_recalculated"]
                h_mc_non_tt_recalc = accumulate(
                    [out[p]["mass_recalculated"] for p in procs_non_tt if p in out]
                )
                d_vals_recalc = (
                    h_data_recalc[{"variation": "nominal"}]
                    .project(var_recalc)
                    .values()
                )
                mc_non_tt_vals_recalc = (
                    h_mc_non_tt_recalc[{"variation": "nominal"}]
                    .project(var_recalc)
                    .values()
                )
                diff_recalc = d_vals_recalc - mc_non_tt_vals_recalc

                # 3. Extract bin edges dynamically for headers
                edges = (
                    h_data_nom[{"variation": "nominal"}]
                    .project(var_nom)
                    .axes[0]
                    .edges
                )
                if not bin_headers:
                    bin_headers = [
                        f"({int(edges[i])}--{int(edges[i+1])})"
                        for i in range(len(edges) - 1)
                    ]

                # 4. Calculate per-bin deltas and total deviation percentage
                delta_bin_by_bin = diff_nom - diff_recalc
                y_nom = float(np.sum(diff_nom))

                delta_bbb = float(np.sqrt(np.sum(delta_bin_by_bin**2)))
                pct_bbb = (delta_bbb / y_nom * 100.0) if y_nom > 0 else 0.0

                summary_data[year] = {
                    "bin_deltas": delta_bin_by_bin,
                    "pct_bbb": pct_bbb,
                }

            if summary_data:
                table_code = generate_latex_table_syst_diff(
                    summary_data, bin_headers, workflow
                )

                f.write(f"% ==========================================\n")
                f.write(f"% Systematic Difference Workflow: {workflow}\n")
                f.write(f"% ==========================================\n\n")
                f.write(table_code)
                f.write("\n\n")

                png_path = plots_dir / f"systematic_diff_{workflow}_table.png"
                compile_latex_to_png(table_code, str(png_path))

    print(
        f"\nSystematic difference tables saved in '{output_filename}' and PNG plots in '{plots_dir}'."
    )