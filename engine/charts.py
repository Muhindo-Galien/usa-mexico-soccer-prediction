"""Matplotlib chart builders. Every function takes the team names and
colors it needs as arguments — nothing here is matchup-specific."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

INK, MUTE, GRID = "#1a1a2e", "#8a8a9e", "#e8e8ee"
GRAY = "#9aa0a6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def make_prediction_chart(p_home, p_draw, p_away, out_dir, *, home_team, away_team,
                           home_color, away_color, match_stage, match_date, city):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"{home_team}\nwin", "Draw\n(90 min)", f"{away_team}\nwin"]
    vals = [p_home, p_draw, p_away]
    colors = [home_color, GRAY, away_color]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(vals) * 100 + 15)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{home_team} vs {away_team}\n{match_stage}  ·  {match_date}  ·  {city}\n"
                 f"Player-Level Ensemble Model", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, f"viz_{home_team}_vs_{away_team}_prediction.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_player_ppi_chart(squad_h, squad_a, out_dir, *, home_team, away_team,
                           home_color, away_color):
    """Side-by-side player performance index comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ppis_h = squad_h["player_ppis"]
    names_h = [p[0].split()[-1] for p in ppis_h]  # last name only
    vals_h = [p[1] for p in ppis_h]
    y_h = np.arange(len(ppis_h))
    ax1.barh(y_h, vals_h, color=home_color, zorder=3, alpha=0.85)
    ax1.set_yticks(y_h)
    ax1.set_yticklabels(names_h, fontsize=10)
    for i, v in enumerate(vals_h):
        ax1.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax1.set_title(f"{home_team} — Player Performance Index", fontsize=12)
    ax1.set_xlabel("PPI Score")
    ax1.xaxis.grid(True, color=GRID, zorder=0)
    ax1.set_axisbelow(True)
    ax1.invert_yaxis()

    ppis_a = squad_a["player_ppis"]
    names_a = [p[0].split()[-1] for p in ppis_a]
    vals_a = [p[1] for p in ppis_a]
    y_a = np.arange(len(ppis_a))
    ax2.barh(y_a, vals_a, color=away_color, zorder=3, alpha=0.85)
    ax2.set_yticks(y_a)
    ax2.set_yticklabels(names_a, fontsize=10)
    for i, v in enumerate(vals_a):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax2.set_title(f"{away_team} — Player Performance Index", fontsize=12)
    ax2.set_xlabel("PPI Score")
    ax2.xaxis.grid(True, color=GRID, zorder=0)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()

    fig.suptitle("Player-Level Performance Index (Club + Tournament Form)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"ppi_{home_team}_vs_{away_team}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def make_position_comparison_chart(squad_h, squad_a, out_dir, *, home_team, away_team,
                                    home_color, away_color, bench_h, bench_a,
                                    bench_impact_fn):
    """Position group strength comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    groups = ["Attack", "Midfield", "Defense", "Goalkeeper", "Bench Impact"]
    h_vals = [squad_h["attack"], squad_h["midfield"], squad_h["defense"],
              squad_h["gk"], bench_impact_fn(bench_h)]
    a_vals = [squad_a["attack"], squad_a["midfield"], squad_a["defense"],
              squad_a["gk"], bench_impact_fn(bench_a)]

    y = np.arange(len(groups))
    h = 0.35
    ax.barh(y + h / 2, h_vals, h, color=home_color, label=home_team, zorder=3, alpha=0.85)
    ax.barh(y - h / 2, a_vals, h, color=away_color, label=away_team, zorder=3, alpha=0.85)

    for i, (hv, av) in enumerate(zip(h_vals, a_vals)):
        ax.text(hv + 0.02, i + h / 2, f"{hv:.2f}", va="center", fontsize=10, color=home_color, fontweight="bold")
        ax.text(av + 0.02, i - h / 2, f"{av:.2f}", va="center", fontsize=10, color=away_color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=12)
    ax.set_title("Position Group Strength Comparison\n(Player Performance Index Aggregation)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, f"position_{home_team}_vs_{away_team}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_score_heatmap(matrix, lambda_h, lambda_a, out_dir, *, home_team, away_team, home_color):
    fig, ax = plt.subplots(figsize=(8, 7))
    display = matrix[:6, :6]
    cmap = LinearSegmentedColormap.from_list("custom", ["#ffffff", "#bcd4ea", home_color])
    im = ax.imshow(display, cmap=cmap, aspect="equal", origin="lower")
    for i in range(6):
        for j in range(6):
            val = display[i, j]
            color = "white" if val > 0.08 else INK
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(range(6))
    ax.set_yticklabels(range(6))
    ax.set_xlabel(f"{away_team} goals")
    ax.set_ylabel(f"{home_team} goals")
    ax.set_title(f"Score Probability Matrix (Player-Adjusted Dixon-Coles)\n"
                 f"{home_team} xG = {lambda_h:.2f}  |  {away_team} xG = {lambda_a:.2f}", fontsize=12)
    fig.colorbar(im, ax=ax, label="Probability", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scorematrix_{home_team}_vs_{away_team}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_model_comparison_chart(models_dict, out_dir, *, home_team, away_team,
                                 home_color, away_color):
    """Compare probabilities across all model components."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    model_names = list(models_dict.keys())
    n = len(model_names)
    x = np.arange(n)
    w = 0.25

    home_vals = [models_dict[m][0] * 100 for m in model_names]
    draw_vals = [models_dict[m][1] * 100 for m in model_names]
    away_vals = [models_dict[m][2] * 100 for m in model_names]

    ax.bar(x - w, home_vals, w, color=home_color, label=home_team, zorder=3, alpha=0.85)
    ax.bar(x,     draw_vals, w, color=GRAY,       label="Draw",     zorder=3, alpha=0.7)
    ax.bar(x + w, away_vals, w, color=away_color, label=away_team, zorder=3, alpha=0.85)

    for i in range(n):
        ax.text(i - w, home_vals[i] + 0.8, f"{home_vals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i,     draw_vals[i] + 0.8, f"{draw_vals[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + w, away_vals[i] + 0.8, f"{away_vals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Component Comparison\n(Team-Level vs Player-Level vs Ensemble)", fontsize=13)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, f"models_{home_team}_vs_{away_team}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
