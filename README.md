ODU Module 5 — Denoising Diffusion Model (DDM)
Old Dominion University | Generative AI | Module 5 technical assignment
Overview
This project trains a denoising diffusion model on Oxford 102 Flowers images and studies how the number of inference-time denoising steps affects sample quality and generation cost. The Jupyter notebook is a standalone Foster-style adaptation, not an exact copy of the original Chapter 8 notebook. It uses a noise-predicting U-Net, exponential moving average (EMA) weights, a cosine noise schedule, and deterministic DDIM-style sampling.
Dataset and environment
Dataset: Oxford 102 Flowers Kaggle copy, 8,189 valid images in the recorded run
Image resolution: 64 × 64 RGB
Execution environment: ODU Wahab Open OnDemand JupyterLab with a GPU (specific GPU model was not recorded)
Python: 3.10.14; TensorFlow: 2.16.2
Random seed: 42
Training: 10 epochs, batch size 32, five dataset repeats per epoch, 1,275 steps per epoch, learning rate 0.0001
Training objective: mean absolute error (MAE) between actual and predicted noise, logged as `n_loss` (not MSE)
Results
Training `n_loss` decreased from 0.1823 in epoch 1 to 0.0889 in epoch 10 (approximately 51.3% reduction). Total reported training time was approximately 20.6 minutes.
The default experiment generated a 4 × 4 grid of 16 images using 20 sampling steps. The ablation generated 16 images each at 50, 200, and 1000 steps, using the same initial noise to support visual comparison.
Denoising steps	Images	Total generation time (s)	Seconds per image
50	16	0.3285	0.0205
200	16	1.2858	0.0804
1000	16	6.3565	0.3973
The recorded sample grids show flower-like colors and forms, but some samples remain dark, noisy, or poorly defined. The 50-, 200-, and 1000-step grids appear similar; more steps did not produce an obvious visual improvement in this run. At 1000 steps, generation took approximately 19.4 times as long per image as at 50 steps. These are observations from one run, not a general benchmark.
Timing note: The separately timed default 20-step run recorded 0.1840 seconds per image; its measurement may include initialization/warm-up overhead, so do not compare it directly against the ablation timings as if all runs had identical timing conditions.
Repository contents
Recommended structure after uploading the files:
```text
ODU-Module5-Diffusion-DDM/
 README.md
 notebooks/
    ODU_Module5_DDM.ipynb
 results/
    training_loss.png
    samples_20.png
    samples_50.png
    samples_200.png
    samples_1000.png
    ablation_side_by_side.png
    ablation_times.csv
     training_history.csv
 report/
 ODU_Module5_DDM_Analysis_Report.pdf
```
The results files are available in `module5_ddm_outputs.zip`; extract them before uploading to the repository if you want the figures to be browsable directly on GitHub. The paths above assume that you upload the files under those names.
How to reproduce
Start a GPU-enabled ODU Wahab Open OnDemand JupyterLab session (or an equivalent TensorFlow GPU environment).
Use Python 3.10 and TensorFlow 2.16.2 where practical. Install the notebook's other imported packages, including NumPy and Matplotlib, if not already available.
Download the Oxford Flowers Kaggle dataset and upload the ZIP as `archive.zip` into the same JupyterLab directory as the notebook. Do not commit the dataset or Kaggle API credentials to GitHub.
Open `ODU_Module5_DDM_Wahab_POOLING_FIXED.ipynb`. This version includes the Keras `AveragePooling2D(pool_size=(2, 2))` correction and automatically discovers/extracts `archive.zip`.
Start from a clean kernel and run all cells in order. The configuration uses `EPOCHS=10`, `BATCH_SIZE=32`, `REPEATS_PER_EPOCH=5`, and `SEED=42` by default.
For a fresh training run, keep `LOAD_MODEL=False`. To resume from a previously saved compatible checkpoint, place the checkpoint at the expected notebook path and set `LOAD_MODEL=True`.
Verify that the training-loss graph, default 20-step grid, three ablation grids, and timing CSV are generated. Save the executed notebook with its outputs visible.
Download `module5_ddm_outputs.zip` and the executed notebook. Reproduction may yield slightly different values depending on hardware, software, and GPU nondeterminism.
Note: Model checkpoints can be large and are not included in the small results ZIP. A fresh run will train the model; do not expect to generate the reported samples from the ZIP alone.
Comparison with earlier modules
The earlier VAE exercise used Fashion-MNIST and the GAN exercise used 64 × 64 grayscale LEGO bricks, whereas this diffusion exercise uses 64 × 64 RGB flowers. Because the datasets differ, sample sharpness and diversity comparisons are qualitative rather than controlled head-to-head measurements. VAEs optimize a variational lower bound on log likelihood; standard GANs do not provide a tractable likelihood; diffusion models have a probabilistic generative formulation, although this notebook's noise-prediction loss and sampling grids are not direct likelihood estimates.
References
Foster, D. (2023). Generative deep learning (2nd ed., Chapter 8). O'Reilly Media.
Original companion code: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion/01_ddm
AI use disclosure
AI assistance was used to help adapt and debug notebook code, organize the README, and draft the written interpretation. Training measurements and generated images are from the executed experiment, not AI-generated or invented results. The author should review and confirm the final submission.
