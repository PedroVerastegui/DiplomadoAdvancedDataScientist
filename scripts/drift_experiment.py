"""Experimento de monitoreo: Baseline NORMAL vs DRIFT.

Reproduce la idea de la clase: se compara un lote de datos 'normal' contra
uno con 'data drift' y se reporta qué features cambiaron (PSI + % de cambio).
Guarda un reporte JSON en artifacts/reports/drift_report.json.
"""
from __future__ import annotations

import json

from src import config
from src.data import generate_churn_data
from src.monitoring import compute_drift
from src.model import load_model


def run_experiment(n: int = 4000) -> dict:
    config.ensure_dirs()

    normal = generate_churn_data(n=n, seed=config.RANDOM_STATE, drift=False)
    drift = generate_churn_data(n=n, seed=config.RANDOM_STATE + 1, drift=True)

    report = compute_drift(normal, drift)

    # Si hay un modelo entrenado, compara también la probabilidad de churn.
    try:
        model = load_model()
        p_normal = model.predict_proba(normal[config.FEATURES])[:, 1].mean()
        p_drift = model.predict_proba(drift[config.FEATURES])[:, 1].mean()
        report["churn_probability"] = {
            "baseline_mean": round(float(p_normal), 4),
            "current_mean": round(float(p_drift), 4),
            "pct_change": round(float((p_drift - p_normal) / p_normal * 100), 1),
            "status": "drift" if abs(p_drift - p_normal) > 0.1 else "estable",
        }
    except Exception as exc:  # modelo aún no entrenado
        report["churn_probability"] = {"error": f"modelo no disponible: {exc}"}

    out = config.REPORTS_DIR / "drift_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Resultados del monitoring: ¿Qué cambió? (NORMAL vs DRIFT) ===")
    for feat, r in report.items():
        if "pct_change" in r:
            signo = "+" if r["pct_change"] >= 0 else ""
            flag = "  <-- DRIFT" if r.get("status") == "drift" else ""
            print(f"  {feat:<18} {signo}{r['pct_change']:>7.1f}%   "
                  f"PSI={r.get('psi', 'n/a')}  [{r.get('status','')}]{flag}")
    print(f"\nReporte guardado en: {out}")
    return report


if __name__ == "__main__":
    run_experiment()
