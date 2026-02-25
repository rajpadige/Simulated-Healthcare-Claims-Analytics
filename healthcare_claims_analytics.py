"""
=============================================================
Simulated Healthcare Claims Analytics
Cost | Utilization | Quality | AI/ML Predictions
=============================================================
Author: [Ritvik Raj Padige]
Date:   2025-11-15
Description:
    End-to-end healthcare claims analytics pipeline using
    synthetic data. Covers data simulation, EDA, cost &
    utilization analysis, quality metrics, and ML modeling.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
N_MEMBERS = 10_000
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "primary":   "#1B4F72",
    "secondary": "#2E86C1",
    "accent":    "#AED6F1",
    "warning":   "#E74C3C",
    "success":   "#27AE60",
    "neutral":   "#BDC3C7",
    "bg":        "#F4F8FB"
}

# ─────────────────────────────────────────────
# PART 1 — DATA SIMULATION
# ─────────────────────────────────────────────

def simulate_claims_data(n: int = N_MEMBERS) -> pd.DataFrame:
    """Generate synthetic healthcare claims dataset."""
    print("=" * 55)
    print("  HEALTHCARE CLAIMS ANALYTICS — SIMULATION START")
    print("=" * 55)
    print(f"\n[1/6] Simulating {n:,} member records...")

    ages = np.clip(np.random.normal(45, 16, n).astype(int), 18, 85)

    # Chronic condition probabilities increase with age
    age_factor = (ages - 18) / 67

    conditions = {
        "diabetes":        np.random.binomial(1, 0.08 + 0.18 * age_factor, n),
        "hypertension":    np.random.binomial(1, 0.10 + 0.30 * age_factor, n),
        "copd":            np.random.binomial(1, 0.02 + 0.10 * age_factor, n),
        "heart_disease":   np.random.binomial(1, 0.02 + 0.14 * age_factor, n),
        "depression":      np.random.binomial(1, 0.08 + 0.05 * age_factor, n),
        "obesity":         np.random.binomial(1, 0.20 + 0.10 * age_factor, n),
    }
    chronic_count = sum(conditions.values())

    plan_types   = np.random.choice(["HMO", "PPO", "HDHP", "EPO"], n, p=[0.30, 0.35, 0.25, 0.10])
    gender       = np.random.choice(["Male", "Female"], n, p=[0.48, 0.52])
    region       = np.random.choice(["Northeast", "South", "Midwest", "West"], n, p=[0.22, 0.35, 0.25, 0.18])

    # Cost simulation (base + condition loading + plan factor + noise)
    plan_factor  = {"HMO": 1.0, "PPO": 1.25, "HDHP": 0.90, "EPO": 1.10}
    pf_array     = np.array([plan_factor[p] for p in plan_types])

    base_cost    = np.random.lognormal(mean=8.0, sigma=1.0, size=n)
    condition_loading = 1 + chronic_count * np.random.uniform(0.15, 0.35, n)
    age_loading  = 1 + np.maximum(0, ages - 40) * 0.012
    noise        = np.random.lognormal(0, 0.3, n)
    total_cost   = base_cost * condition_loading * age_loading * pf_array * noise

    # Utilization
    er_visits    = np.random.poisson(0.15 + 0.25 * (chronic_count / 6), n)
    ip_admits    = np.random.poisson(0.05 + 0.30 * (chronic_count / 6) + 0.03 * (ages > 60), n)
    op_visits    = np.random.poisson(3.0  + 2.5  * (chronic_count / 6), n)
    rx_fills     = np.random.poisson(6.0  + 12.0 * (chronic_count / 6), n)

    # Quality flags (HEDIS-inspired)
    has_a1c      = np.where(conditions["diabetes"] == 1,
                             np.random.binomial(1, 0.72, n), np.nan)
    has_mammogram = np.where((gender == "Female") & (ages >= 40) & (ages <= 74),
                              np.random.binomial(1, 0.68, n), np.nan)
    readmit_30d  = np.where(ip_admits > 0,
                              np.random.binomial(1, 0.12 + 0.05 * (chronic_count / 6), n), 0)
    preventive_visit = np.random.binomial(1, 0.55 + 0.05 * (~conditions["diabetes"]).astype(float), n)

    df = pd.DataFrame({
        "member_id":       [f"M{str(i).zfill(5)}" for i in range(n)],
        "age":             ages,
        "gender":          gender,
        "region":          region,
        "plan_type":       plan_types,
        **conditions,
        "chronic_count":   chronic_count,
        "total_cost":      total_cost.round(2),
        "er_visits":       er_visits,
        "ip_admits":       ip_admits,
        "op_visits":       op_visits,
        "rx_fills":        rx_fills,
        "has_a1c_check":   has_a1c,
        "has_mammogram":   has_mammogram,
        "readmit_30d":     readmit_30d,
        "preventive_visit": preventive_visit,
        "high_cost_flag":  (total_cost > np.percentile(total_cost, 90)).astype(int)
    })

    print(f"    ✓ Dataset shape: {df.shape}")
    print(f"    ✓ Total simulated spend: ${df['total_cost'].sum():,.0f}")
    print(f"    ✓ Mean PMPM: ${df['total_cost'].mean()/12:,.2f}")
    return df


# ─────────────────────────────────────────────
# PART 2 — COST ANALYSIS
# ─────────────────────────────────────────────

def cost_analysis(df: pd.DataFrame):
    print("\n[2/6] Running Cost Analysis...")

    # PMPM by plan
    df["pmpm"] = df["total_cost"] / 12
    pmpm_plan  = df.groupby("plan_type")["pmpm"].mean().sort_values(ascending=False)

    # Pareto — top 10% driving % of total spend
    df_sorted  = df.sort_values("total_cost", ascending=False).copy()
    df_sorted["cum_pct"] = df_sorted["total_cost"].cumsum() / df_sorted["total_cost"].sum()
    top10_spend = df[df["high_cost_flag"] == 1]["total_cost"].sum() / df["total_cost"].sum()

    # Cost by chronic count
    cost_by_chronic = df.groupby("chronic_count")["pmpm"].mean()

    # Cost by age band
    df["age_band"] = pd.cut(df["age"], bins=[17,29,39,49,59,69,85],
                             labels=["18-29","30-39","40-49","50-59","60-69","70+"])
    cost_by_age = df.groupby("age_band", observed=True)["pmpm"].mean()

    fig = plt.figure(figsize=(16, 10), facecolor=COLORS["bg"])
    fig.suptitle("Cost Analysis Dashboard", fontsize=18, fontweight="bold", color=COLORS["primary"], y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- PMPM by Plan ---
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(pmpm_plan.index, pmpm_plan.values, color=[COLORS["primary"], COLORS["secondary"],
                                                               COLORS["accent"], COLORS["neutral"]])
    ax1.set_title("Avg PMPM by Plan Type", fontweight="bold", color=COLORS["primary"])
    ax1.set_ylabel("PMPM ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)

    # --- Cost Distribution (log) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(np.log10(df["total_cost"] + 1), bins=60, color=COLORS["secondary"], alpha=0.8, edgecolor="white")
    ax2.set_title("Cost Distribution (log scale)", fontweight="bold", color=COLORS["primary"])
    ax2.set_xlabel("log₁₀(Total Annual Cost)")
    ax2.set_ylabel("Member Count")

    # --- Pareto Chart ---
    ax3 = fig.add_subplot(gs[0, 2])
    pct_members = [10, 20, 30, 50, 80, 100]
    top_n = [int(p/100 * len(df)) for p in pct_members]
    pct_cost = [df_sorted["total_cost"][:n].sum() / df_sorted["total_cost"].sum() * 100 for n in top_n]
    ax3.plot(pct_members, pct_cost, "o-", color=COLORS["warning"], linewidth=2.5, markersize=6)
    ax3.axvline(x=10, color=COLORS["neutral"], linestyle="--", alpha=0.7)
    ax3.axhline(y=pct_cost[0], color=COLORS["neutral"], linestyle="--", alpha=0.7)
    ax3.annotate(f"Top 10% → {pct_cost[0]:.0f}% of spend", xy=(10, pct_cost[0]),
                 xytext=(25, pct_cost[0]-10), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=COLORS["warning"]), color=COLORS["warning"])
    ax3.set_title("Pareto: Members vs. Spend", fontweight="bold", color=COLORS["primary"])
    ax3.set_xlabel("Top X% of Members")
    ax3.set_ylabel("% of Total Spend")

    # --- Cost by Chronic Burden ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(cost_by_chronic.index.astype(str), cost_by_chronic.values,
            color=[plt.cm.Blues(0.3 + 0.12*i) for i in range(len(cost_by_chronic))])
    ax4.set_title("Avg PMPM by Chronic Conditions", fontweight="bold", color=COLORS["primary"])
    ax4.set_xlabel("# Chronic Conditions")
    ax4.set_ylabel("PMPM ($)")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # --- Cost by Age Band ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(range(len(cost_by_age)), cost_by_age.values, "s-",
             color=COLORS["primary"], linewidth=2.5, markersize=8)
    ax5.set_xticks(range(len(cost_by_age)))
    ax5.set_xticklabels(cost_by_age.index, rotation=30)
    ax5.set_title("Avg PMPM by Age Band", fontweight="bold", color=COLORS["primary"])
    ax5.set_ylabel("PMPM ($)")
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax5.fill_between(range(len(cost_by_age)), cost_by_age.values, alpha=0.15, color=COLORS["secondary"])

    # --- High Cost KPIs ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    kpis = [
        ("Total Simulated Spend", f"${df['total_cost'].sum()/1e6:.1f}M"),
        ("Avg Annual Cost/Member", f"${df['total_cost'].mean():,.0f}"),
        ("Avg PMPM", f"${df['pmpm'].mean():,.2f}"),
        (f"Top 10% Drive", f"{top10_spend*100:.1f}% of spend"),
        ("High-Cost Members (>90th)", f"{df['high_cost_flag'].sum():,}"),
    ]
    for i, (label, val) in enumerate(kpis):
        ax6.text(0.05, 0.88 - i*0.18, label, transform=ax6.transAxes,
                 fontsize=10, color=COLORS["primary"])
        ax6.text(0.05, 0.80 - i*0.18, val, transform=ax6.transAxes,
                 fontsize=14, fontweight="bold", color=COLORS["secondary"])

    ax6.set_title("Key Cost KPIs", fontweight="bold", color=COLORS["primary"])

    plt.savefig(OUTPUT_DIR / "01_cost_analysis.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Top 10% of members drive {top10_spend*100:.1f}% of total cost")
    print(f"    ✓ Chart saved: 01_cost_analysis.png")
    return df


# ─────────────────────────────────────────────
# PART 3 — UTILIZATION ANALYSIS
# ─────────────────────────────────────────────

def utilization_analysis(df: pd.DataFrame):
    print("\n[3/6] Running Utilization Analysis...")

    per1k = 1000 / len(df)

    # Per-1000 rates by plan
    util_plan = df.groupby("plan_type").agg(
        er_rate=("er_visits", lambda x: x.sum() * per1k),
        ip_rate=("ip_admits", lambda x: x.sum() * per1k),
        op_rate=("op_visits", lambda x: x.mean()),
        rx_rate=("rx_fills", lambda x: x.mean())
    ).round(1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=COLORS["bg"])
    fig.suptitle("Utilization Analysis Dashboard", fontsize=18,
                 fontweight="bold", color=COLORS["primary"], y=0.98)

    # --- ER per 1,000 ---
    ax = axes[0][0]
    ax.barh(util_plan.index, util_plan["er_rate"],
            color=COLORS["warning"], alpha=0.85, edgecolor="white")
    ax.set_title("ER Visits per 1,000 Members", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("Visits per 1,000")
    for i, v in enumerate(util_plan["er_rate"]):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)

    # --- IP Admits per 1,000 ---
    ax = axes[0][1]
    ax.barh(util_plan.index, util_plan["ip_rate"],
            color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.set_title("Inpatient Admits per 1,000 Members", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("Admits per 1,000")
    for i, v in enumerate(util_plan["ip_rate"]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)

    # --- Utilization by Chronic Burden ---
    ax = axes[1][0]
    util_chron = df.groupby("chronic_count")[["er_visits","ip_admits"]].mean()
    x = np.arange(len(util_chron))
    w = 0.35
    ax.bar(x - w/2, util_chron["er_visits"], w, label="ER Visits", color=COLORS["warning"], alpha=0.85)
    ax.bar(x + w/2, util_chron["ip_admits"], w, label="IP Admits", color=COLORS["primary"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(util_chron.index.astype(str))
    ax.set_title("Avg Utilization by Chronic Burden", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("# Chronic Conditions")
    ax.legend()

    # --- Rx fills distribution ---
    ax = axes[1][1]
    ax.hist(df["rx_fills"], bins=30, color=COLORS["secondary"], alpha=0.8, edgecolor="white")
    ax.axvline(df["rx_fills"].mean(), color=COLORS["warning"], linestyle="--",
               linewidth=2, label=f"Mean: {df['rx_fills'].mean():.1f}")
    ax.set_title("Rx Fill Distribution", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("Annual Rx Fills")
    ax.set_ylabel("Member Count")
    ax.legend()

    for row in axes:
        for a in row:
            a.set_facecolor(COLORS["bg"])
            for spine in a.spines.values():
                spine.set_edgecolor("#DDDDDD")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "02_utilization_analysis.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    ✓ ER visits/1,000: {df['er_visits'].sum() * per1k:.1f}")
    print(f"    ✓ IP admits/1,000: {df['ip_admits'].sum() * per1k:.1f}")
    print(f"    ✓ Chart saved: 02_utilization_analysis.png")


# ─────────────────────────────────────────────
# PART 4 — QUALITY METRICS
# ─────────────────────────────────────────────

def quality_analysis(df: pd.DataFrame):
    print("\n[4/6] Running Quality Metrics Analysis...")

    # HEDIS-inspired metrics
    a1c_rate   = df[df["diabetes"]==1]["has_a1c_check"].mean() * 100
    mammo_rate = df[(df["gender"]=="Female") & df["age"].between(40,74)]["has_mammogram"].mean() * 100
    readmit_r  = df[df["ip_admits"]>0]["readmit_30d"].mean() * 100
    prev_rate  = df["preventive_visit"].mean() * 100

    metrics = {
        "Diabetic A1c\nMonitoring": a1c_rate,
        "Mammography\nScreening": mammo_rate,
        "Preventive\nVisit Rate": prev_rate,
    }
    lower_is_better = {"30-Day\nReadmission": readmit_r}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=COLORS["bg"])
    fig.suptitle("Quality Metrics Dashboard (HEDIS-Inspired)", fontsize=16,
                 fontweight="bold", color=COLORS["primary"])

    # Gauge-style bar chart for positive metrics
    ax = axes[0]
    names  = list(metrics.keys())
    values = list(metrics.values())
    bars   = ax.barh(names, values, color=[COLORS["success"], COLORS["secondary"], COLORS["primary"]], alpha=0.85)
    ax.axvline(80, color=COLORS["warning"], linestyle="--", linewidth=1.5, label="80% target")
    ax.set_xlim(0, 100)
    ax.set_title("Quality Compliance Rates (%)", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("% of Eligible Members")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}%", va="center", fontsize=10)
    ax.legend()
    ax.set_facecolor(COLORS["bg"])

    # Readmission by plan
    ax = axes[1]
    readmit_plan = df[df["ip_admits"]>0].groupby("plan_type")["readmit_30d"].mean() * 100
    bars2 = ax.bar(readmit_plan.index, readmit_plan.values,
                   color=[COLORS["warning"] if v > 12 else COLORS["success"] for v in readmit_plan.values],
                   alpha=0.85, edgecolor="white")
    ax.axhline(12, color=COLORS["neutral"], linestyle="--", linewidth=1.5, label="12% benchmark")
    ax.set_title("30-Day Readmission Rate by Plan", fontweight="bold", color=COLORS["primary"])
    ax.set_ylabel("Readmission Rate (%)")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.legend()
    ax.set_facecolor(COLORS["bg"])

    # Quality Score by Chronic Burden
    ax = axes[2]
    df["quality_score"] = (
        df["preventive_visit"] * 0.3 +
        (df["has_a1c_check"].fillna(0.5) * df["diabetes"]) * 0.4 +
        (1 - df["readmit_30d"]) * 0.3
    ) * 100
    qs_chron = df.groupby("chronic_count")["quality_score"].mean()
    ax.plot(qs_chron.index, qs_chron.values, "D-", color=COLORS["success"], linewidth=2.5, markersize=8)
    ax.fill_between(qs_chron.index, qs_chron.values, alpha=0.15, color=COLORS["success"])
    ax.set_title("Composite Quality Score\nby Chronic Burden", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("# Chronic Conditions")
    ax.set_ylabel("Quality Score (0–100)")
    ax.set_ylim(0, 100)
    ax.set_facecolor(COLORS["bg"])

    for a in axes:
        for spine in a.spines.values():
            spine.set_edgecolor("#DDDDDD")

    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig(OUTPUT_DIR / "03_quality_metrics.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    ✓ A1c monitoring rate: {a1c_rate:.1f}%")
    print(f"    ✓ Mammography rate: {mammo_rate:.1f}%")
    print(f"    ✓ 30-day readmission rate: {readmit_r:.1f}%")
    print(f"    ✓ Chart saved: 03_quality_metrics.png")
    return df


# ─────────────────────────────────────────────
# PART 5 — AI / ML PREDICTIONS
# ─────────────────────────────────────────────

def ml_analysis(df: pd.DataFrame):
    print("\n[5/6] Building AI / ML Predictive Models...")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (mean_absolute_error, r2_score,
                                      roc_auc_score, classification_report)
    except ImportError:
        print("    ⚠ scikit-learn not installed. Run: pip install scikit-learn")
        return

    # Feature engineering
    feature_cols = [
        "age", "chronic_count", "er_visits", "op_visits", "rx_fills",
        "diabetes", "hypertension", "copd", "heart_disease", "depression", "obesity",
        "preventive_visit", "readmit_30d"
    ]
    df_enc = df.copy()
    for col in ["gender", "plan_type", "region"]:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df[col])
        feature_cols.append(col + "_enc")

    X = df_enc[feature_cols].fillna(0)
    y_cost  = np.log1p(df["total_cost"])       # regression target
    y_class = df["high_cost_flag"]              # classification target

    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cost,  test_size=0.2, random_state=42)
    _,    _,    yk_tr, yk_te  = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # --- Regression model ---
    print("    Training cost prediction model (GBM Regressor)...")
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    reg.fit(X_tr, yc_tr)
    y_pred_reg = reg.predict(X_te)
    mae = mean_absolute_error(np.expm1(yc_te), np.expm1(y_pred_reg))
    r2  = r2_score(yc_te, y_pred_reg)
    print(f"    ✓ Regression — MAE: ${mae:,.0f} | R²: {r2:.3f}")

    # --- Classification model ---
    print("    Training high-cost classifier (GBM Classifier)...")
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_tr, yk_tr)
    y_pred_clf  = clf.predict(X_te)
    y_pred_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(yk_te, y_pred_prob)
    print(f"    ✓ Classifier — AUC-ROC: {auc:.3f}")

    # --- Feature importances ---
    fi_reg = pd.Series(reg.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)
    fi_clf = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=COLORS["bg"])
    fig.suptitle("AI / ML Predictive Models", fontsize=16, fontweight="bold", color=COLORS["primary"])

    # Feature Importance (Regression)
    ax = axes[0]
    ax.barh(fi_reg.index, fi_reg.values, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.set_title(f"Cost Prediction\nFeature Importance\n(R²={r2:.3f})", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("Importance")
    ax.set_facecolor(COLORS["bg"])

    # Feature Importance (Classifier)
    ax = axes[1]
    ax.barh(fi_clf.index, fi_clf.values, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.set_title(f"High-Cost Classifier\nFeature Importance\n(AUC={auc:.3f})", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("Importance")
    ax.set_facecolor(COLORS["bg"])

    # ROC Curve
    ax = axes[2]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(yk_te, y_pred_prob)
    ax.plot(fpr, tpr, color=COLORS["warning"], linewidth=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["warning"])
    ax.set_title("ROC Curve — High-Cost Classifier", fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_facecolor(COLORS["bg"])

    for a in axes:
        for spine in a.spines.values():
            spine.set_edgecolor("#DDDDDD")

    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig(OUTPUT_DIR / "04_ml_models.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Chart saved: 04_ml_models.png")

    return reg, clf, feature_cols


# ─────────────────────────────────────────────
# PART 6 — SUMMARY STATS TABLE
# ─────────────────────────────────────────────

def save_summary_table(df: pd.DataFrame):
    print("\n[6/6] Generating Summary Statistics Table...")
    summary = df.groupby("plan_type").agg(
        members=("member_id", "count"),
        avg_annual_cost=("total_cost", "mean"),
        pmpm=("pmpm", "mean"),
        er_per_1k=("er_visits", lambda x: x.sum() * 1000 / len(x)),
        ip_per_1k=("ip_admits", lambda x: x.sum() * 1000 / len(x)),
        avg_rx_fills=("rx_fills", "mean"),
        pct_high_cost=("high_cost_flag", "mean"),
        avg_quality=("quality_score", "mean")
    ).round(2)

    summary["avg_annual_cost"] = summary["avg_annual_cost"].apply(lambda x: f"${x:,.0f}")
    summary["pmpm"] = summary["pmpm"].apply(lambda x: f"${x:,.2f}")
    summary["er_per_1k"] = summary["er_per_1k"].round(1)
    summary["ip_per_1k"] = summary["ip_per_1k"].round(1)
    summary["pct_high_cost"] = (summary["pct_high_cost"] * 100).round(1).astype(str) + "%"

    summary.to_csv(OUTPUT_DIR / "summary_by_plan.csv")
    print("    ✓ Summary saved: summary_by_plan.csv")
    print("\n" + summary.to_string())
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = simulate_claims_data()
    df = cost_analysis(df)
    utilization_analysis(df)
    df = quality_analysis(df)
    ml_analysis(df)
    save_summary_table(df)

    print("\n" + "=" * 55)
    print("  ALL ANALYSES COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 55)
