
Zipf-SGD & Optimal-Schedule Sandbox
===================================

A tiny playground for studying SGD on Zipf-like distributions, with tooling to compare learning‑rate schedules and test optimal-control ideas.

Quick start
-----------

git clone https://github.com/your-name/zipf-sgd.git
cd zipf-sgd
pip install torch matplotlib numpy

# fit Zipf on a corpus
python zipf.py data/corpus.txt \
       --schedule auto \
       --epochs 10 \
       --plot-kld main_kld.png

* First run trains from scratch, saves plots in **pic/**, and drops a plateau.ckpt.
* Any later run skips retraining and immediately launches two follow‑up runs (invtime vs remaining_power) from the checkpoint.

CLI cheatsheet
--------------

Flag | Default | Meaning
---- | ------- | -------
--schedule  | constant | LR rule: constant, invtime, power, auto, or remaining_power
--alpha     | 1.0      | Exponent for power/remaining_power (must be <1 for remaining_power)
--kld-thresh| 0.01     | Plateau if all ΔKLD below this for kld_patience epochs
--plot-kld  | None     | PNG path → save KL-divergence curve
(see python zipf.py -h for full list)

Research roadmap / paper checklist
----------------------------------

* Extra-layer loss & loss curves
* t_final vs S0 graph
* Greedy vs optimal schedules
* Literature scout on LR schedules for LLMs
* Optimal-control derivation and spectrum matching
* Experiments: Zipf, MNIST, linearised transformer
* Discussion: generalisation to full LLM Hessian spectra

License
-------

MIT
