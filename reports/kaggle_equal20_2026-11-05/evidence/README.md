# Evidence gốc Midterm

`27/` và `28/` giữ JSON/NPZ từ hai Kaggle study. Manifests, configs, run histories và saved validation probabilities giúp giải thích training/checkpoint/metrics. Không có raw text, vocabulary hoặc weights.

Nhóm train đủ20 epoch cho mỗi neural model, chọn checkpoint bằng emotion27 Macro-F1 tại0.5. Model28 dùng criterion trên27 emotions dù head có28 outputs. Full study và cách import nằm ở nhánh nguồn `feature/final28-report`, commit `289f2fb`; nhánh ôn không lặp training hoặc inference.
