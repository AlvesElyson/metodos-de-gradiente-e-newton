import numpy as np
import matplotlib.pyplot as plt

from item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, passo_constante, armijo, backtracking
from problema_1.item_c import newton

# Problema 2 — Item d)  Comparação de iterações ──────────────────────────────────────────────

x0 = np.array([0.0, 0.0])
alpha_otimo = 2.0 / (2.0 + 18.0)

metodos = {
    "GD + Passo Constante": gradiente_descendente(
        f, grad_f, x0, passo_constante, alpha=alpha_otimo),
    "GD + Armijo":          gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "GD + Backtracking":    gradiente_descendente(
        f, grad_f, x0, backtracking, alpha0=1.0, rho=0.8, c=1e-4),
    "Newton + Armijo":      newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
}

# Tabela comparativa ──────────────────────────────────────────────

print("=" * 85)
print("Problema 2 — Item d)  Comparação do número de iterações  (x0 = [0, 0])")
print("=" * 85)
print(f"{'Método':<25} {'Iter':>6} {'f*':>14} {'‖∇f‖':>14} "
      f"{'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 85)
for nome, r in metodos.items():
    print(f"{nome:<25} {r['n_iter']:>6} {r['f_opt']:>14.6e} "
          f"{r['gnorm_final']:>14.6e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}")

# Figura 1 — f(xₖ) e ‖∇f‖ vs iteração ──────────────────────────────────────────────

cores = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Problema 2 — Item d)  Comparação de convergência", fontsize=13)

for (nome, r), cor in zip(metodos.items(), cores):
    it  = np.arange(len(r["hist_f"]))
    ms  = 8  if nome.startswith("Newton") else 3
    mev = 1  if nome.startswith("Newton") else max(1, len(it)//100)
    axes[0].semilogy(it[::mev], r["hist_f"][::mev],
                     color=cor, lw=1.5, marker="o", ms=ms, label=nome)
    axes[1].semilogy(it[::mev], r["hist_gnorm"][::mev],
                     color=cor, lw=1.5, marker="o", ms=ms, label=nome)

axes[0].set_title("f(xₖ) vs Iteração")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("f(xₖ)  [log]")
axes[0].legend(fontsize=9); axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].set_title("‖∇f(xₖ)‖ vs Iteração")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tol = 1e-6")
axes[1].legend(fontsize=9); axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob2_d_convergencia.png", dpi=150)
plt.close()

# Figura 2 — Barras: iterações e avaliações ──────────────────────────────────────────────

nomes   = list(metodos.keys())
iters   = [r["n_iter"]   for r in metodos.values()]
aval_f  = [r["n_evals_f"] for r in metodos.values()]
aval_g  = [r["n_evals_g"] for r in metodos.values()]

x_pos = np.arange(len(nomes))
width = 0.28

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Problema 2 — Item d)  Comparação: iterações e avaliações", fontsize=13)

# Iterações
bars = axes2[0].bar(x_pos, iters, color=cores, alpha=0.8, edgecolor="black")
axes2[0].bar_label(bars, fmt="%d", fontsize=10)
axes2[0].set_xticks(x_pos)
axes2[0].set_xticklabels(nomes, rotation=15, ha="right", fontsize=9)
axes2[0].set_title("Número de iterações")
axes2[0].set_ylabel("Iterações")
axes2[0].grid(True, axis="y", ls="--", alpha=0.4)

# Avaliações de f e g
b1 = axes2[1].bar(x_pos - width/2, aval_f, width, label="Aval. f", color=cores, alpha=0.7, edgecolor="black")
b2 = axes2[1].bar(x_pos + width/2, aval_g, width, label="Aval. ∇f", color=cores, alpha=0.4, edgecolor="black", hatch="//")
axes2[1].set_xticks(x_pos)
axes2[1].set_xticklabels(nomes, rotation=15, ha="right", fontsize=9)
axes2[1].set_title("Avaliações de f e ∇f")
axes2[1].set_ylabel("Número de avaliações")
axes2[1].legend(fontsize=9)
axes2[1].grid(True, axis="y", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob2_d_comparacao_barras.png", dpi=150)
plt.close()

# Figura 3 — Curvas de nível + todas as trajetórias ──────────────────────────────────────────────

X  = np.linspace(-0.5, 2.5, 400)
Y  = np.linspace(-0.5, 4.0, 400)
Xg, Yg = np.meshgrid(X, Y)
Z  = (Xg + 2*Yg - 7)**2 + (2*Xg + Yg - 5)**2
levels = np.logspace(-4, 2, 25)

fig3, ax = plt.subplots(figsize=(8, 7))
fig3.suptitle("Problema 2 — Item d)  Trajetórias (todos os métodos)", fontsize=12)
ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.45)

for (nome, r), cor in zip(metodos.items(), cores):
    traj = r["hist_x"]
    step = max(1, len(traj) // 300)
    ax.plot(traj[::step, 0], traj[::step, 1],
            "-o", color=cor, lw=1.3, ms=3,
            label=f"{nome}  ({r['n_iter']} it.)")

ax.plot(x0[0], x0[1], "ks", ms=9, zorder=5, label="x₀ = (0,0)")
ax.plot(1, 3, "r*", ms=13, zorder=5, label="x* = (1,3)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("prob2_d_trajetorias.png", dpi=150)
plt.close()