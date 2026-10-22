# -*- coding: utf-8 -*-
"""
Análise e Visualização Profissional de Resultados MM/GBSA e Decomposição por Resíduo
Padrão de Publicação Científica (Filtro de Artefatos, Heatmap 100% Preenchido)
"""
from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS E SISTEMAS
# ==============================================================================
PROJECT_DIR = Path(r"C:\Users\User\PycharmProjects\Analises_folp")
DATA_DIR = PROJECT_DIR / "resultados_mmpbsa_decomp"
OUT = PROJECT_DIR / "figuras_mmpbsa"

SYSTEMS_FOLP1 = ["FolP1_WT", "FolP1_Val14Gly", "FolP1_Val14Ile"]
SYSTEMS_FOLP2 = ["FolP2_WT", "FolP2_Arg23Ser", "FolP2_Pro25Leu"]
SYSTEMS = SYSTEMS_FOLP1 + SYSTEMS_FOLP2
REPLICAS = ["rep1", "rep2", "rep3"]
TERMS = ["VDWAALS", "EEL", "EGB", "ESURF", "G gas", "G solv", "TOTAL"]

# Limite para ignorar artefatos de solvatação não-compensada (-175 kcal/mol)
THRESHOLD_MAX_ABS_ENERGY = 30.0

# Paleta de Cores Distintas por Sistema
COLOR_PALETTE = {
    "FolP1_WT": "#08519c",  # Azul Escuro
    "FolP1_Val14Gly": "#3182bd",  # Azul Médio
    "FolP1_Val14Ile": "#6baed6",  # Azul Claro
    "FolP2_WT": "#a50f15",  # Vermelho Escuro
    "FolP2_Arg23Ser": "#de2d26",  # Coral
    "FolP2_Pro25Leu": "#fb6a4a"  # Laranja/Vermelho
}

# Estilo Editorial das Figuras
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "grid.linestyle": ":"
})


# ==============================================================================
# 2. FUNÇÕES DE LEITURA E PROCESSAMENTO
# ==============================================================================
def read_section(path, section):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, line in enumerate(lines)
                  if line.strip().casefold() == section.casefold()), None)
    if start is None:
        raise ValueError(f"Seção {section!r} não encontrada: {path}")
    hi = next((i for i in range(start, len(lines))
               if lines[i].strip().startswith("Frame #")), None)
    if hi is None:
        raise ValueError(f"Cabeçalho Frame # ausente após {section}: {path}")
    rows = []
    for line in lines[hi:]:
        if not line.strip():
            break
        rows.append(next(csv.reader([line])))
    df = pd.DataFrame(rows[1:], columns=[x.strip() for x in rows[0]])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["TOTAL"]).reset_index(drop=True)


def framewise_binding(path):
    c = read_section(path, "Complex Energy Terms")
    r = read_section(path, "Receptor Energy Terms")
    l = read_section(path, "Ligand Energy Terms")
    n = min(len(c), len(r), len(l))
    if n == 0:
        raise ValueError(f"Sem quadros compatíveis: {path}")
    out = pd.DataFrame({"Frame": c["Frame #"].iloc[:n].to_numpy()})
    for term in TERMS:
        if all(term in x.columns for x in (c, r, l)):
            out[term] = c[term].iloc[:n].to_numpy() - r[term].iloc[:n].to_numpy() - l[term].iloc[:n].to_numpy()
    return out


def read_decomp(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hi = next((i for i, line in enumerate(lines) if line.strip().startswith("Residue,Internal")), None)
    if hi is None:
        raise ValueError(f"Cabeçalho de decomposição não encontrado: {path}")
    rows = []
    for line in lines[hi + 2:]:
        s = line.strip()
        if not s:
            if rows: break
            continue
        if not re.match(r"^[A-Z]{3}\s+\d+\s*,", s):
            if rows: break
            continue
        rows.append(next(csv.reader([line])))
    if not rows:
        raise ValueError(f"Nenhuma linha de resíduo em {path}")
    comps = ["Internal", "VDW", "Electrostatic", "Polar_Solvation", "Nonpolar_Solvation", "TOTAL"]
    cols = ["Residue"] + [f"{x}_{stat}" for x in comps for stat in ("Avg", "SD", "SEM")]
    df = pd.DataFrame(rows, columns=cols)
    df["Residue"] = df["Residue"].str.strip()
    for c in cols[1:]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["TOTAL_Avg"]).reset_index(drop=True)


def build_unified_residues(df):
    """ Extrai o número do resíduo e unifica rótulos de mutantes na mesma posição """
    df['ResNum'] = df['Residue'].str.extract(r'(\d+)').astype(int)
    mapping = {}
    for res_num, group in df.groupby("ResNum"):
        unique_names = group["Residue"].unique()
        if len(unique_names) == 1:
            mapping[res_num] = unique_names[0]
        else:
            aa_types = []
            for name in sorted(unique_names):
                m = re.match(r'^([A-Z]{3})', name)
                if m and m.group(1) not in aa_types:
                    aa_types.append(m.group(1))
            mapping[res_num] = f"{'/'.join(aa_types)} {res_num}"
    df['Unified_Residue'] = df['ResNum'].map(mapping)
    return df


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close()


# ==============================================================================
# 3. ROTINAS DE PLOTAGEM
# ==============================================================================

def plot_componentes_energeticos(summary):
    key_terms = ["VDWAALS", "EEL", "G solv", "TOTAL"]
    term_names = {
        "VDWAALS": r"$\Delta E_{\mathrm{vdw}}$ (Van der Waals)",
        "EEL": r"$\Delta E_{\mathrm{elec}}$ (Eletrostática)",
        "G solv": r"$\Delta G_{\mathrm{solv}}$ (Solvatação)",
        "TOTAL": r"$\Delta G_{\mathrm{bind}}$ Total"
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.flatten()

    for idx, term in enumerate(key_terms):
        ax = axes[idx]
        sub = summary[summary.Component == term].set_index("System").reindex(SYSTEMS)

        bars = ax.bar(SYSTEMS, sub.Mean, yerr=sub.SD_replicas.fillna(0), capsize=4,
                      color=[COLOR_PALETTE[s] for s in SYSTEMS],
                      edgecolor="black", linewidth=0.8, alpha=0.9,
                      error_kw={'elinewidth': 1.0, 'ecolor': 'black'})

        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_title(term_names.get(term, term), fontweight="bold", fontsize=11)
        ax.set_ylabel("Energia (kcal/mol)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        ax.set_xticks(range(len(SYSTEMS)))
        ax.set_xticklabels(SYSTEMS, rotation=35, ha="right", fontweight="bold")

        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.8 if height >= 0 else -1.8
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                        xytext=(0, 0), textcoords="offset points",
                        ha='center', va=va, fontsize=8, fontweight='bold')

    fig.suptitle(r"Componentes Termodinâmicos de Ligação MM/GBSA ($\Delta G_{\mathrm{bind}}$)", fontsize=13,
                 fontweight="bold", y=0.98)
    savefig("01_componentes_energeticos_deltaG")


def plot_decomposicao_separada(rs, dec):
    """
    Gráfico 2: Decomposição por Resíduo com barras de tamanho correto.
    """
    # Filtro essencial de artefatos de -175 kcal/mol
    rs_clean = rs[rs["TOTAL_Avg"].abs() <= THRESHOLD_MAX_ABS_ENERGY].copy()

    if not dec.empty:
        dec_clean = dec[dec["TOTAL_Avg"].abs() <= THRESHOLD_MAX_ABS_ENERGY].copy()
        rs_stats = dec_clean.groupby(["System", "Unified_Residue"], as_index=False).agg(
            Mean=("TOTAL_Avg", "mean"),
            SD=("TOTAL_Avg", "std")
        )
    else:
        rs_stats = rs_clean.copy()
        rs_stats["Mean"] = rs_stats["TOTAL_Avg"]
        rs_stats["SD"] = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- PAINEL A1: Linhas FolP1 ---
    ax_a1 = axes[0, 0]
    for sys in SYSTEMS_FOLP1:
        sub = rs_clean[rs_clean.System == sys].sort_values("ResNum")
        if not sub.empty:
            ax_a1.plot(sub.ResNum, sub.TOTAL_Avg, label=sys, color=COLOR_PALETTE[sys],
                       linewidth=1.6, alpha=0.85)

    ax_a1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_a1.set_title("a) FolP1: Perfil de Decomposição Energética", fontsize=11, fontweight="bold")
    ax_a1.set_xlabel("Número do Resíduo (FolP1)", fontweight="bold")
    ax_a1.set_ylabel(r"$\Delta G_{\mathrm{decomp}}$ (kcal/mol)", fontweight="bold")
    ax_a1.grid(True, alpha=0.3, linestyle=":")
    ax_a1.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # --- PAINEL A2: Linhas FolP2 ---
    ax_a2 = axes[0, 1]
    for sys in SYSTEMS_FOLP2:
        sub = rs_clean[rs_clean.System == sys].sort_values("ResNum")
        if not sub.empty:
            ax_a2.plot(sub.ResNum, sub.TOTAL_Avg, label=sys, color=COLOR_PALETTE[sys],
                       linewidth=1.6, alpha=0.85)

    ax_a2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_a2.set_title("b) FolP2: Perfil de Decomposição Energética", fontsize=11, fontweight="bold")
    ax_a2.set_xlabel("Número do Resíduo (FolP2)", fontweight="bold")
    ax_a2.set_ylabel(r"$\Delta G_{\mathrm{decomp}}$ (kcal/mol)", fontweight="bold")
    ax_a2.grid(True, alpha=0.3, linestyle=":")
    ax_a2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # --- PAINEL B1: Barras Hotspots FolP1 ---
    ax_b1 = axes[1, 0]
    df_f1 = rs_clean[rs_clean.System.isin(SYSTEMS_FOLP1)]
    top_f1 = df_f1.groupby("Unified_Residue")["TOTAL_Avg"].apply(lambda x: x.abs().mean()).sort_values(
        ascending=False).head(10).index

    x1 = np.arange(len(top_f1))
    w1 = 0.8 / len(SYSTEMS_FOLP1)

    for i, sys in enumerate(SYSTEMS_FOLP1):
        sub = rs_stats[rs_stats.System == sys].set_index("Unified_Residue").reindex(top_f1)
        vals = sub.Mean.fillna(0).values
        sds = sub.SD.fillna(0).values
        ax_b1.bar(x1 + (i - (len(SYSTEMS_FOLP1) - 1) / 2) * w1, vals, w1, yerr=sds, capsize=2,
                  label=sys, color=COLOR_PALETTE[sys], edgecolor="black", linewidth=0.5)

    ax_b1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_b1.set_xticks(x1)
    ax_b1.set_xticklabels(top_f1, rotation=35, ha="right", fontweight="bold")
    ax_b1.set_title("c) FolP1: Principais Resíduos Hotspots (Média ± DP)", fontsize=11, fontweight="bold")
    ax_b1.set_ylabel(r"$\Delta G_{\mathrm{decomp}}$ (kcal/mol)", fontweight="bold")
    ax_b1.grid(axis="y", alpha=0.3, linestyle=":")
    ax_b1.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # --- PAINEL B2: Barras Hotspots FolP2 ---
    ax_b2 = axes[1, 1]
    df_f2 = rs_clean[rs_clean.System.isin(SYSTEMS_FOLP2)]
    top_f2 = df_f2.groupby("Unified_Residue")["TOTAL_Avg"].apply(lambda x: x.abs().mean()).sort_values(
        ascending=False).head(10).index

    x2 = np.arange(len(top_f2))
    w2 = 0.8 / len(SYSTEMS_FOLP2)

    for i, sys in enumerate(SYSTEMS_FOLP2):
        sub = rs_stats[rs_stats.System == sys].set_index("Unified_Residue").reindex(top_f2)
        vals = sub.Mean.fillna(0).values
        sds = sub.SD.fillna(0).values
        ax_b2.bar(x2 + (i - (len(SYSTEMS_FOLP2) - 1) / 2) * w2, vals, w2, yerr=sds, capsize=2,
                  label=sys, color=COLOR_PALETTE[sys], edgecolor="black", linewidth=0.5)

    ax_b2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_b2.set_xticks(x2)
    ax_b2.set_xticklabels(top_f2, rotation=35, ha="right", fontweight="bold")
    ax_b2.set_title("d) FolP2: Principais Resíduos Hotspots (Média ± DP)", fontsize=11, fontweight="bold")
    ax_b2.set_ylabel(r"$\Delta G_{\mathrm{decomp}}$ (kcal/mol)", fontweight="bold")
    ax_b2.grid(axis="y", alpha=0.3, linestyle=":")
    ax_b2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    savefig("02_decomposicao_top_residuos_hotspots")


def plot_estabilidade_trajetoria(traj):
    fig, ax = plt.subplots(figsize=(11, 6))
    for system in SYSTEMS:
        sub = traj[traj.System == system].groupby("Frame", as_index=False).TOTAL.mean().sort_values("Frame")
        if not sub.empty:
            smooth = sub.TOTAL.rolling(window=3, min_periods=1).mean()
            ax.plot(sub.Frame, smooth, label=system, color=COLOR_PALETTE[system], linewidth=1.8)
            ax.plot(sub.Frame, sub.TOTAL, color=COLOR_PALETTE[system], alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Número do Quadro Amostrado (MMPBSA)", fontweight="bold")
    ax.set_ylabel(r"$\Delta G_{\mathrm{bind}}$ (kcal/mol)", fontweight="bold")
    ax.set_title(r"Variação e Estabilidade do $\Delta G_{\mathrm{bind}}$ ao longo do Tempo", fontsize=12,
                 fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(ncol=3, fontsize=8.5, loc="upper right", framealpha=0.9)
    savefig("03_estabilidade_deltaG_trajetoria")


def plot_heatmap_lado_a_lado(rs):
    """
    Gráfico 4: Heatmap 100% Preenchido sem NENHUMA célula em branco.
    """
    rs_clean = rs[rs["TOTAL_Avg"].abs() <= THRESHOLD_MAX_ABS_ENERGY].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # --- FOLP1 HEATMAP ---
    df_f1 = rs_clean[rs_clean.System.isin(SYSTEMS_FOLP1)]
    if not df_f1.empty:
        heat_f1 = df_f1.pivot(index="Unified_Residue", columns="System", values="TOTAL_Avg").reindex(
            columns=SYSTEMS_FOLP1)
        top_f1 = heat_f1.abs().mean(axis=1).sort_values(ascending=False).head(15).index
        heat_f1 = heat_f1.reindex(top_f1).fillna(0.0)

        sns.heatmap(heat_f1, cmap="RdBu_r", center=0, annot=True, fmt=".1f",
                    linewidths=0.8, linecolor="white",
                    cbar_kws={'label': r'$\Delta G_{\mathrm{decomp}}$ (kcal/mol)'},
                    ax=ax1, annot_kws={"size": 9.5, "weight": "bold"})

        ax1.set_title("a) FolP1 (WT vs Mutantes)", fontsize=11, fontweight="bold", pad=12)
        ax1.set_xlabel("Sistema", fontweight="bold", fontsize=10.5)
        ax1.set_ylabel("Resíduo Hotspot (FolP1)", fontweight="bold", fontsize=10.5)
        ax1.set_xticklabels(SYSTEMS_FOLP1, rotation=35, ha="right", fontweight="bold")

    # --- FOLP2 HEATMAP ---
    df_f2 = rs_clean[rs_clean.System.isin(SYSTEMS_FOLP2)]
    if not df_f2.empty:
        heat_f2 = df_f2.pivot(index="Unified_Residue", columns="System", values="TOTAL_Avg").reindex(
            columns=SYSTEMS_FOLP2)
        top_f2 = heat_f2.abs().mean(axis=1).sort_values(ascending=False).head(15).index
        heat_f2 = heat_f2.reindex(top_f2).fillna(0.0)

        sns.heatmap(heat_f2, cmap="RdBu_r", center=0, annot=True, fmt=".1f",
                    linewidths=0.8, linecolor="white",
                    cbar_kws={'label': r'$\Delta G_{\mathrm{decomp}}$ (kcal/mol)'},
                    ax=ax2, annot_kws={"size": 9.5, "weight": "bold"})

        ax2.set_title("b) FolP2 (WT vs Mutantes)", fontsize=11, fontweight="bold", pad=12)
        ax2.set_xlabel("Sistema", fontweight="bold", fontsize=10.5)
        ax2.set_ylabel("Resíduo Hotspot (FolP2)", fontweight="bold", fontsize=10.5)
        ax2.set_xticklabels(SYSTEMS_FOLP2, rotation=35, ha="right", fontweight="bold")

    fig.suptitle(r"Heatmap de Decomposição Energética por Resíduo ($\Delta G_{\mathrm{decomp}}$)", fontsize=13,
                 fontweight="bold", y=0.98)
    savefig("04_heatmap_decomposicao_residuos")


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {DATA_DIR}\nEdite PROJECT_DIR no início do script.")
    OUT.mkdir(parents=True, exist_ok=True)

    trajs, decomps, component_rows, issues = [], [], [], []

    for system in SYSTEMS:
        for rep in REPLICAS:
            folder = DATA_DIR / system / rep
            cp, dp = folder / "FINAL_RESULTS_MMPBSA.csv", folder / "DECOMP_RESULTS_MMPBSA.dat"
            if not cp.exists() or not dp.exists():
                issues.append(f"{system}/{rep}: CSV ou DECOMP ausente")
                continue
            try:
                t = framewise_binding(cp)
                t["System"], t["Replica"] = system, rep
                trajs.append(t)
                for term in TERMS:
                    if term in t:
                        component_rows.append({
                            "System": system, "Replica": rep, "Component": term,
                            "Mean_kcal_mol": t[term].mean(), "SD_frames_kcal_mol": t[term].std(),
                            "N_frames": t[term].notna().sum()
                        })
                d = read_decomp(dp)
                d["System"], d["Replica"] = system, rep
                decomps.append(d)
            except Exception as e:
                issues.append(f"{system}/{rep}: {type(e).__name__}: {e}")

    if not trajs:
        raise RuntimeError("Nenhuma réplica foi lida; verifique caminhos e formato dos CSVs.")

    traj = pd.concat(trajs, ignore_index=True)
    comps = pd.DataFrame(component_rows)
    dec = pd.concat(decomps, ignore_index=True) if decomps else pd.DataFrame()

    traj.to_csv(OUT / "dados_deltaG_por_quadro.csv", index=False)
    comps.to_csv(OUT / "resumo_componentes_por_replica.csv", index=False)

    if not comps.empty:
        summary = comps.groupby(["System", "Component"], as_index=False).agg(
            Mean=("Mean_kcal_mol", "mean"),
            SD_replicas=("Mean_kcal_mol", "std"),
            N_replicas=("Replica", "nunique")
        )
        summary.to_csv(OUT / "resumo_componentes_por_sistema.csv", index=False)
        plot_componentes_energeticos(summary)

    if not dec.empty:
        dec = build_unified_residues(dec)
        dec.to_csv(OUT / "decomposicao_por_residuo_todas_replicas.csv", index=False)

        rs = dec.groupby(["System", "ResNum", "Unified_Residue"], as_index=False).agg(TOTAL_Avg=("TOTAL_Avg", "mean"))
        rs.to_csv(OUT / "media_decomposicao_por_residuo_sistema.csv", index=False)

        plot_decomposicao_separada(rs, dec)
        if "TOTAL" in traj:
            plot_estabilidade_trajetoria(traj)
        plot_heatmap_lado_a_lado(rs)

    with (OUT / "relatorio_execucao.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dados: {DATA_DIR}\nSaídas: {OUT}\n")
        f.write(f"Réplicas com trajetória lida: {traj[['System', 'Replica']].drop_duplicates().shape[0]}\n")
        f.write(f"Quadros exportados: {len(traj)}\n")
        f.write(
            f"Réplicas com decomposição lida: {dec[['System', 'Replica']].drop_duplicates().shape[0] if not dec.empty else 0}\n\n")
        f.write("Avisos:\n" + ("\n".join(issues) if issues else "Nenhum.\n"))

    print(f"Concluído com sucesso! Figuras salvas em: {OUT}")


if __name__ == "__main__":
    main()