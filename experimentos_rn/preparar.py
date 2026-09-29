import sys; sys.path.insert(0, "../src")
from pathlib import Path
from nusecast.data import load_config, load_panel, make_features, split_label
cfg = load_config(Path("../config.json"))
panel, cov, _, _ = load_panel(Path("../data/raw/Numero_incidentes.csv"), cfg)
fr = make_features(panel, cov, cfg, include_next=True)
fr["split"] = fr.period.map(lambda p: split_label(p, cfg))
fr.to_csv("features.csv", index=False)
print(fr.shape)
