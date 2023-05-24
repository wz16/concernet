## ConCerNet: A Contrastive Learning Based Framework for Automated Conservation Law Discovery and Trustworthy Dynamical System Prediction
This is the official implementation of paper: ConCerNet: A Contrastive Learning Based Framework for Automated Conservation Law Discovery and Trustworthy Dynamical System Prediction

The ConCerNet framework includes two separate steps:
* using contrastive learning to automatically learn conservation law from trajectory data. 
* learning dynamical system augmented by a projection layer that ensures the conserved quantity in trajectory prediction. 

To notice, the 2nd step is independent of the 1st step, i.e., any conservation function can be used in the 2nd step.
## Dependencies

The codes should work with the common packages below. 

 * scipy
 * numpy
 * torch
 * seaborn
 * matplotlib
 * autograd

See requirements.txt for all prerequisites, and you can also install them using the following command.
```
pip install -r requirements.txt
```
## Usage

Ensure that the directories `simulations/`, `logs/`, `figs/` and `saved_models/` are writable.
```
mkdir simulations logs figs saved_models
```

### Running quick demonstration of the projection layer
Learn a simple chemical reaction system assuming the mass conservation function is known:

```
python simple_projection_layer_demo.py
```
The illustrative figs are saved to `figs/`

### Running all the experiments under ConCerNet framework vs baseline neural networks
```
bash scripts/run_all.sh
```

The result summary will be saved into `logs/`


## Cite

[ConCerNet paper](https://arxiv.org/abs/2302.05783):

```
@article{zhang2023concernet,
  title={ConCerNet: A Contrastive Learning Based Framework for Automated Conservation Law Discovery and Trustworthy Dynamical System Prediction
},
  author={Zhang, Wang and Weng, Tsui-Wei and Das, Subhro and Megretski, Alexandre and Daniel, Luca and Nguyen, Lam M.},
  journal={arXiv preprint arXiv:2302.05783},
  year={2023}
}
```