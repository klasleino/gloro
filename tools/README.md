# gloro-tools

Here we provide some tools and examples to supplement the `gloro` library.
The `lower_bounds` directory provides code for estimating lower bounds of the Lipschitz constant of a given network.
The `training` directory provides utilities and scripts for training GloRo Nets and for reproducing the results from the paper, *Globally-Robust Neural Networks* (Leino et al., ICML 2021).

For example, to reproduce the results for MNIST with a robustness radius of 1.58, the following could be run from the `training` directory:
```
python train_gloro.py \
    --dataset mnist \
    --architecture minmax_cnn_4C3F \
    --epsilon 1.58 \
    --epsilon_schedule [0.01]-log-[50%:1.1] \
    --epochs 300 \
    --batch_size 512 \
    --loss sparse_trades_kl.1.5 \
    --lr_schedule decay_after_half_to_0.000001 \
    --augmentation none
```