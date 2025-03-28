# 🚌 Edinburgh School Bus Optimization

This project replicates and extends the work of Chu et al. (2020) to optimize school bus operations, with a focus on the City of Edinburgh. It includes deterministic and stochastic optimization models for efficient assignment of buses to school routes, addressing key challenges such as repositioning delays, resource usage, and route consistency.

## 📌 Overview

The project aims to:
- Minimize the number of buses required
- Reduce total repositioning mileage
- Account for uncertainty in travel times
- Offer a robust, flexible tool for planners

We developed and tested both Deterministic Integer Programming (D-IP) and Stochastic Integer Programming (S-IP) models. The models were customized to Edinburgh's school transport system and integrated into a Python-based decision support workflow.

## 🔧 Repository Structure

```
├── DIP_model.py         # Definition of the deterministic model
├── SIP_model.py         # Definition of the stochastic model
├── Solve_DIP.py         # Solver script for deterministic model
├── Solve_SIP.py         # Solver script for stochastic model
├── main.py              # Main entry point for running experiments
```

## 🧠 Methodology

We implement optimization models inspired by:
> Chu, A., Keskinocak, P. and Villarreal, M.C., 2020. *Empowering Denver Public Schools to optimize school bus operations*. INFORMS Journal on Applied Analytics, 50(5), pp.298–312.

Models were adapted to:
- Generate realistic bus routes using Edinburgh’s public transport map and catchment areas
- Estimate student demand from open data
- Support scenario analysis under uncertainty

## 🚀 Getting Started

1. Clone this repo:
```bash
git clone https://github.com/Dennisyyds/Topics-in-Applied-Operational-Researc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```


## 📜 License

This project is released for educational and research purposes.

## 🔗 Citation

If you use this work, please cite the original paper:

> Chu, A., Keskinocak, P. and Villarreal, M.C., 2020. *Empowering Denver Public Schools to optimize school bus operations*. INFORMS Journal on Applied Analytics, 50(5), pp.298–312. https://doi.org/10.1287/inte.2020.1042
