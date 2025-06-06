
Zipf-SGD & Optimal-Schedule Sandbox
===================================

This repository is a research sandbox for exploring *learning‑rate schedules* through the simplest possible language‑model toy: learning **Zipf's law** (the unigram distribution of word frequencies) with stochastic gradient descent.

Why Zipf?
* The unigram model is fast—you can run thousands of SGD steps in seconds.
* Yet its **Hessian spectrum** already has a long tail reminiscent of deep‑networks, so LR‑decay phenomena appear.
* You can solve the optimal‑control version analytically, giving ground‑truth schedules against which heuristic rules (linear, exponential, 1/ *t*, etc.) can be benchmarked.

Results
-----------
![kld_compare](https://github.com/user-attachments/assets/2ed2f36c-2494-4130-8931-1169744651d6)

We compare the classic 1/t (invtime) schedule with our custom remaining‐power schedule over a fixed budget of 5 epochs of training (≈ 50 000 SGD steps total). Although the 1/t curve initially drives down cross‐entropy more rapidly, it plateaus after about 3 epochs (≈ 30 000 steps) at a higher KL‐value. In contrast, our remaining‐power schedule continues to reduce KLD for the full 5 epochs, eventually reaching a much lower minimum.

### Final KL Divergence Comparison

- **1/t (invtime) schedule**  
  - `α = 1.0` (no warm-up)  
  - Final KL: **0.0030 nats/token**  
  - Plateau occurs around **epoch 3**

- **Remaining-power schedule**  
  ```math
    \mathrm{LR}(t) \;=\; \Bigl(1 - \frac{t}{T}\Bigr)^{\frac{\alpha}{\,1 - \alpha\,}}\,, 
    \quad T = \text{total steps}
  ```
  - Plateau detected around **epoch 4**  
  - Final KL: **0.0012 nats/token**

> **Note**: We choose $\alpha$ based on the empirical Zipf exponent $\nu = 1.05$. Since Zipf’s law distributes word frequencies as $p^{-\nu}$, we set  
> ```math
>   \alpha \;=\; 1 \;-\; \frac{1}{2\nu} \;=\; 1 \;-\; \frac{1}{2 \times 1.05} \;\approx\; 0.524.
> ``` 
> This choice ensures our remaining-power schedule closely follows the curvature implied by Zipf’s law.




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
# What it does
* **Fits** a unigram language-model (word frequencies) to any plain-text file.  
* Supports several learning-rate schedules:  
  `constant`, `invtime` *(1 / t)*, `power`, `auto`, and the custom **`remaining_power`**.  
* **Automatically detects plateaus** in KL-divergence (ΔKLD \< *kld_thresh* for *kld_patience* epochs).  
* **Checkpoints** at the first plateau (`plateau.ckpt`).  
* If a checkpoint already exists, it **skips fresh training** and immediately resumes two follow-up runs from the plateau—one with `invtime`, one with `remaining_power`—then overlays their KL-curves for easy comparison.

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
