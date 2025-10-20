# ACTMED: Timely Clinical Diagnosis through Active Test Selection (NeurIPS 2025)

<p align="center">
	<img src="./Overview.png" alt="ACT-Med overview diagram" width="850" />
</p>

<p align="center">
	<b>Authors:</b> Silas Ruhrberg Estevez · Nicolás Astorga · Mihaela van der Schaar
</p>

<p align="center">
	<a href="#paper">Paper (coming soon)</a> •
	<a href="#citation">BibTeX</a> •
	<a href="#getting-started">Getting Started</a> •
	<a href="#experiments">Experiments</a> •
	<a href="#results">Results</a>
</p>

---

There is growing interest in using machine learning (ML) to support clinical diagnosis, but most approaches rely on static, fully observed datasets and fail to reflect the sequential, resource-aware reasoning clinicians use in practice. Diagnosis remains complex and error prone, especially in high-pressure or resource-limited settings, underscoring the need for frameworks that help clinicians make timely and cost-effective decisions. We propose \name \ (Adaptive Clinical Test selection via Model-based Experimental Design), a diagnostic framework that integrates Bayesian Experimental Design (BED) with large language models (LLMs) to better emulate real-world diagnostic reasoning. At each step, \name \ selects the test expected to yield the greatest reduction in diagnostic uncertainty for a given patient. LLMs act as flexible simulators, generating plausible patient state distributions and supporting belief updates without requiring structured, task-specific training data. Clinicians can remain in the loop; reviewing test suggestions, interpreting intermediate outputs, and applying clinical judgment throughout. We evaluate \name \ on real-world datasets and show it can optimize test selection to improve diagnostic accuracy, interpretability, and resource use. This represents a step toward transparent, adaptive, and clinician-aligned diagnostic systems that generalize across settings with reduced reliance on domain-specific data.




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

### Environment variables (OpenAI)

This repo uses the standard OpenAI client and enforces.  To use GPT-4o or GPT-3o-mini Set the following in your shell (bash):

```bash
# Required
export OPENAI_API_KEY="..."


export OPENAI_BASE_URL="https://api.openai.com/v1"  
```

It is also possible to use other models, simply implement them using the other_chat function under `lib/model.py`.




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
- Link: coming soon

We will update this section with the camera-ready link (arXiv and/or OpenReview) once available.

## Citation

If you find this work useful, please cite it. A final BibTeX entry will be provided upon publication. For now, you can use the placeholder below and update fields later:

```bibtex
@inproceedings{ruhrberg2025timely,
	title     = {Timely Clinical Diagnosis},
	author    = {Silas RUhrberg Estevez and Nicolas Astorga and Mihaela van der Schaar},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	year      = {2025},
	note      = {To appear},
	url       = {https://arxiv.org/abs/TODO}
}
```

## License

MIT Opensource License



---


