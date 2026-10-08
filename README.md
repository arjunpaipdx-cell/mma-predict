Octagon Odds: GPU-accelerated UFC fight prediction
Pick two UFC fighters. You get the probability each one wins, plus how (KO, submission, decision) and when (by round) the fight ends. Under the hood:

Stage
NVIDIA tech
What it does
Feature engineering
RAPIDS cuDF
Per-fighter career aggregates over 17k+ fighter-fights × 30 stat columns, leak-free (each fight only sees earlier fights)
Win model
XGBoost on CUDA
Gradient-boosted trees + GPU hyperparameter sweep; SHAP explanations via pred_contribs
Fight simulator
CuPy
Round-by-round Monte Carlo with millions of simulated fights per matchup, batched across matchups on the GPU
App
Streamlit
Matchup picker, method/round charts, prediction drivers, tale of the tape


Every module falls back to pandas/NumPy/CPU automatically (mmapred/backend.py), so the app also runs on a laptop with no NVIDIA GPU.
Results (held-out test set: every UFC fight from Jan 2024 to Oct 2026)
Strict chronological split: train on 2001–2022, tune on 2023, test once on 2024+.

Win model
Accuracy
Log loss
Brier
AUC
XGBoost (this repo)
64.0%
0.643
0.226
0.674
Elo rating baseline
55.7%
0.677
0.242
0.586
Coin flip
50.0%
0.693
0.250
0.500


Simulator (finish vs decision)
Value
Brier, simulator
0.236
Brier, always predict the base rate
0.250
AUC for "does it end early"
0.64
Predicted vs actual finish rate
51.6% vs 50.2%


For context, closing betting lines pick the winner roughly 65–70% of the time. Getting into that range from public stats alone is the realistic target.
CPU vs GPU benchmarks
Measured on a free Google Colab Tesla T4 (benchmarks/run_all.py; CPU = the Colab VM's CPU, GPU timings after one warm-up run).

Stage
Workload
CPU
GPU (T4)
Speedup
Monte Carlo (CuPy)
4,950 matchups × 20k sims = 99M simulated fights
41.7 s
2.2 s
18.8×
XGBoost train (CUDA)
140k rows × 42 features, 500 trees
18.2 s
2.6 s
7.0×
Features (cuDF)
357k fighter-fight rows, grouped cumulative stats
1.6 s
1.2 s
1.4×


The simulator is fully parallel (every matchup × every simulated fight is independent), so it gains the most. Feature engineering is a small table, which is about the size where cuDF only starts to break even: transfer and kernel launch overhead eat most of the gain. A GPU only pays off once there's enough parallel work.


How it works
flowchart LR

  A[UFCStats CSVs] --> B[data.py<br/>parse + clean]

  B --> C[features.py<br/>cuDF cumulative stats + Elo]

  C --> D[train.py<br/>XGBoost CUDA]

  C --> E[simulate.py<br/>CuPy Monte Carlo]

  D -- P(A wins) --> E

  D --> F[app.py]

  E --> F

Leak-free features. For every fight, a fighter's stats come only from their earlier UFC fights: strikes landed and absorbed per minute, accuracy, defense, takedowns, control time, knockdowns, KO/sub rates per round, recent form, Elo, age, reach, and layoff. Small samples are shrunk toward the league average, so a fighter with two fights doesn't get extreme numbers.

Corner bias. UFCStats usually lists the winner first (5,625 vs 3,167 fights). Training on each fight from both corners, and averaging both orientations at prediction time, stops the model from learning "first name wins" and guarantees P(A beats B) = 1 − P(B beats A).

Simulator. In each round, each fighter has a KO hazard (their KO power × the opponent's KO vulnerability) and a submission hazard. Every simulated fight also draws a random "form" multiplier per fighter, and hazards rise slightly in later rounds. Fights that go the distance go to the judges, whose probability is set so the simulator's overall win rate matches XGBoost. XGBoost decides who wins; the simulator adds how and when. The three simulator settings were tuned on 2023 fights only.
Run it
On Colab (GPU): open notebooks/colab_gpu_run.ipynb, set the runtime to T4 GPU, and run all cells. It trains, evaluates, benchmarks, and lets you download artifacts/.

Locally (CPU is fine for the app):

pip install -r requirements.txt          # Mac: brew install libomp first

python -m mmapred.train                  # downloads data, trains, writes artifacts/

python -m mmapred.evaluate               # simulator check on 2024+ fights

streamlit run app.py

python -m mmapred.predict "Islam Makhachev" "Ilia Topuria" --rounds 5
Known limitations
Public stats only: no betting odds, injuries, weight cuts, or camp changes.
Fighter names are the join key; a few fighters share names on UFCStats.
The model is a little underconfident on heavy favourites (in the 0.7–0.8 bucket, the favourite actually won 84%). Isotonic calibration is an easy next step.
Stoppages listed as injuries (e.g. a knee giving out) are excluded from KO stats, but other fluke stoppages still count as KOs.
Only UFC fights count, so prospects coming from other promotions start with few stats.
Next steps
Compare against closing betting odds (needs an odds dataset).
Calibrate probabilities (isotonic regression on the validation year).
Computer vision: pose estimation on fight footage with TensorRT to count strikes from video.
Serve the model with NVIDIA Triton Inference Server.

Data: Greco1899/scrape_ufc_stats (a mirror of UFCStats.com). For fun and learning, not betting advice.
