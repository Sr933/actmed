# ACTMED: Timely Clinical Diagnosis through Active Test Selection (NeurIPS 2025)

<p align="center">
	<a href="./Overview.png">
		<img src="./Overview.png" alt="ACTMED overview diagram" />
	</a>
	<br/>
	<sub>Click image to view full-resolution.</sub>
</p>

<p align="center">
	<b>Authors:</b> Silas Ruhrberg Estévez · Nicolás Astorga · Mihaela van der Schaar
</p>

<p align="center">
	<a href="#paper">Preprint</a> •
	<a href="#citation">BibTeX</a> •
	<a href="#getting-started">Getting Started</a> •
	<a href="#experiments">Experiments</a> •
	<a href="#results">Results</a>
</p>

---

There is growing interest in using machine learning (ML) to support clinical diagnosis, but most approaches rely on static, fully observed datasets and fail to reflect the sequential, resource-aware reasoning clinicians use in practice. Diagnosis remains complex and error prone, especially in high-pressure or resource-limited settings, underscoring the need for frameworks that help clinicians make timely and cost-effective decisions. We propose ACTMED  (Adaptive Clinical Test selection via Model-based Experimental Design), a diagnostic framework that integrates Bayesian Experimental Design (BED) with large language models (LLMs) to better emulate real-world diagnostic reasoning. At each step, ACTMED selects the test expected to yield the greatest reduction in diagnostic uncertainty for a given patient. LLMs act as flexible simulators, generating plausible patient state distributions and supporting belief updates without requiring structured, task-specific training data. Clinicians can remain in the loop; reviewing test suggestions, interpreting intermediate outputs, and applying clinical judgment throughout. We evaluate ACTMED on real-world datasets and show it can optimize test selection to improve diagnostic accuracy, interpretability, and resource use. This represents a step toward transparent, adaptive, and clinician-aligned diagnostic systems that generalize across settings with reduced reliance on domain-specific data.




## Getting Started

### 1) Environment (Conda)

Create and activate the environment from the provided spec:

```bash
conda env create -f actmed_environment.yml
conda activate actmed
```


### 2) Data layout

This repo expects data in the `data/` folder. The datasets from the paper ar already provided, but more can be added.

```
actmed/
	data/
		diabetes/
			diabetes.csv
```

- Place additional datasets into their respective subfolders.
- Ensure file names and formats match what the loaders in `lib/datasets.py` and the BED models `lib/bed.py` expect. For new datasets, custom classes will have to be generated to instruct the model how to load the data and introduce domain specific prompts.

## Experiments

We provide shell scripts in `src/` to launch experiments.

- `src/runExperiment.sh` — main experiments (condition classification)
- `src/runOSCEs.sh` — OSCE-style assessments
- `src/runEntropy.sh` — for comparison against KL-divergence
- `src/runSampling.sh` — sampling analyses 

Execute, for example:

```bash
bash src/runExperiment.sh
```
The datasets, models and seeds are chosen in the bash file.

### Environment variables (GPT‑4o and GPT‑4o‑mini)

Set the following environment variables in your shell (bash). These are required for `lib/model.py` as currently implemented:

```bash
export GPT4O_KEY=""
export GPT4O_ENDPOINT=""
export GPT4O_DEPLOYMENT=""

export GPT4Omini_KEY=""
export GPT4Omini_ENDPOINT=""
export GPT4Omini_DEPLOYMENT=""
```

Notes:
- Names are case-sensitive. Use exactly the keys shown above.
- These correspond to your API key, endpoint/base URL, and deployment/model name for GPT‑4o and GPT‑4o‑mini respectively.

### Using a local LLM (optional)

You can plug in any local or third‑party LLM by implementing `other_chat` in `lib/model.py`. That function receives `(user_prompt, model_name, temperature, top_p)` and should return a plain string response. Then select `model_name='other'` wherever models are chosen (e.g., via scripts or constructors).

Minimal contract:
- Input: `user_prompt` (string), `model_name` (string), `temperature` (float), `top_p` (float)
- Output: a single string with the model’s reply
- Error behavior: return an informative string or raise, and consider simple retries if calling a flaky local server

See references in `lib/bed.py` where `other` is supported alongside `gpt-4o` and `gpt-4o-mini`.

### Adding new datasets

- Place your files under `data/<your_dataset>/...`.
- Create a loader class by subclassing `Dataset` in `lib/datasets.py` (override `preprocess_data`, `get_item`, and `return_feature_names`).
- Create a corresponding BED model by subclassing `BEDModel` in `lib/bed.py` (set domain‑specific prompts by overriding `predict_risk`, `sample_random_variable`, `select_feature_implicit`, and optionally `get_best_global_features`, and `format_known_data`).
- Domain‑specific vignette/formatting helpers live in `lib/helperfunctions.py` (e.g., `hepatitis_clinical_vignette`, `diabetes_clinical_vignette`, `kidney_clinical_vignette`).



## Results

All outputs are written to `results/`:

- `results/main/` — primary experiment outputs
- `results/sampling/` — sampling-based analyses
- `results/entropy/` — entropy based analyses

If you want to keep figures or intermediate artifacts, check `analysis/` (e.g., `analysis/Figure1.py`) for figure generation utilities.

## Repository structure

- `lib/` — core library code (datasets, models, OSCE evaluation, helpers)
- `analysis/` — analysis and plotting scripts (paper figures)
- `src/` — experiment launch scripts
- `data/` — datasets (see layout above)
- `results/` — experiment outputs (auto-generated)

## Paper

- Title: Timely Clinical Diagnosis through Active Test Selection (NeurIPS 2025)
- Preprint: https://arxiv.org/abs/2510.18988

We will update this section with the camera-ready link (arXiv and/or OpenReview) once available.

## Citation

If you find this work useful, please cite it. A final BibTeX entry will be provided upon publication. For now, you can use the placeholder below and update fields later:

```bibtex
@inproceedings{
ruhrberg2025timely,
title={Timely Clinical Diagnosis through Active Test Selection},
author={Silas {Ruhrberg Estevez} and Nicolás Astorga and Mihaela van der Schaar},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=lO7RGax6u9}
}
```

## License

MIT Opensource License



---


