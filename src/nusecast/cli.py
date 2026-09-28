import argparse
from pathlib import Path

from nusecast.data import load_config, load_panel, make_features
from nusecast.experiment import run_experiment, save_json
from nusecast.training import training_lock


def main():
    parser = argparse.ArgumentParser(description="NUSECast · laboratorio de pronóstico mensual")
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=["demo", "extended", "custom"], default="demo")
    parser.add_argument("--keep-suspect", action="store_true", help="Sensibilidad: conservar los meses sospechosos sin corregir")
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(args.config or root / "config.json")
    if args.profile not in config["profiles"]:
        parser.error("Ese perfil no está en la configuración. Para custom usa el config.json de un experimento guardado.")
    if args.keep_suspect:
        config["exclude_suspect"] = False
    source = args.source or root / "data/raw/Numero_incidentes.csv"
    if not source.is_file():
        parser.error(f"No existe el CSV: {source}")
    if args.command == "prepare":
        panel, coverage, _, audit = load_panel(source, config)
        target = root / "data/processed"
        target.mkdir(parents=True, exist_ok=True)
        make_features(panel, coverage, config, include_next=True).to_csv(target / "features.csv", index=False)
        save_json(target / "audit.json", audit)
        print(f"Panel validado: {audit['months']} meses × {audit['upz_count']} UPZ. {target}")
    else:
        with training_lock(root):
            run_experiment(root, source, config, args.profile)


if __name__ == "__main__":
    main()
