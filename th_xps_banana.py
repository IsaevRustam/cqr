from th_xps import *
activation = "relu" 
res = run_experiment(
    generate_banana, oracle_banana,
    name="Banana / Crescent (2D)", d=2,
    n_train=5000, n_cal=8000,
    hidden_dim=64, epochs=200,
    alpha=0.05, lr=0.01, bandwidth_scale=3.0,
    seed=42, activation=activation,
)
plot_banana(res, f"figures_cqr/banana_adaptivity_{activation}.pdf", show=True)