"""CLI de la plataforma: generar datos, entrenar, predecir, servir y drift.

Ejemplos:
    python main.py make-data
    python main.py train --data data/raw/customer_churn.csv
    python main.py predict --data data/raw/customer_churn.csv \
        --model artifacts/models/churn_pipeline.joblib --output artifacts/reports/preds.csv
    python main.py drift
    python main.py serve --port 8000
"""
from __future__ import annotations

import argparse
import json

from src import config
from src.data import write_raw_dataset
from src.model import train
from src.predict import predict_batch


def cmd_make_data(args):
    path = write_raw_dataset(n=args.n)
    print(f"Dataset generado: {path} ({args.n} filas)")


def cmd_train(args):
    metrics = train(data_path=args.data, model_path=args.model)
    print("Modelo entrenado. Métricas de evaluación:")
    print(json.dumps(metrics, indent=2))
    print(f"Artefacto: {args.model or config.MODEL_PATH}")


def cmd_predict(args):
    out = predict_batch(args.data, model_path=args.model, output_path=args.output)
    print(out[[config.ID_COL, "churn_probability", "churn_prediction"]].head(10).to_string(index=False))
    if args.output:
        print(f"\nPredicciones guardadas en: {args.output}")


def cmd_drift(args):
    from scripts.drift_experiment import run_experiment
    run_experiment()


def cmd_serve(args):
    import uvicorn
    uvicorn.run("src.service:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Customer Intelligence ML Platform")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("make-data", help="Genera el dataset sintético de churn")
    s.add_argument("--n", type=int, default=4000)
    s.set_defaults(func=cmd_make_data)

    s = sub.add_parser("train", help="Entrena y guarda el modelo")
    s.add_argument("--data", default=None)
    s.add_argument("--model", default=None)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict", help="Predicción por lotes sobre un CSV")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--output", default=None)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("drift", help="Ejecuta el experimento NORMAL vs DRIFT")
    s.set_defaults(func=cmd_drift)

    s = sub.add_parser("serve", help="Levanta el servicio REST con métricas")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
