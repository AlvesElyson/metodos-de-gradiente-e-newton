import numpy as np
import matplotlib.pyplot as plt

from item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, passo_constante, armijo, backtracking
from problema_1.item_c import newton

# ──────────────────────────────────────────────
# Problema 2 — Item b)
# Ponto inicial x0 = (0, 0)
# ──────────────────────────────────────────────
x0 = np.array([0.0, 0.0])

# Passo constante ótimo: α* = 2/(λ_min + λ_max) = 2/(2+18) = 0.1
alpha_otimo = 2.0 / (2.0 + 18.0)

metodos = {
    "GD + Passo Constante": gradiente_descendente(
        f, grad_f, x0, passo_constante, alpha=alpha_otimo),
    "GD + Armijo":          gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Newton + Armijo":      newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
}

# ──────────────────────────────────────────────
# Tabela de resultados
# ──────────────────────────────────────────────
print("=" * 80)
print("Problema 2 — Item b)  (x0 = [0, 0])")
print("=" * 80)
print(f"{'Método':<25} {'Iter':>5} {'f*':>14} {'‖∇f‖':>14} "
      f"{'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 80)
for nome, r in metodos.items():
    print(f"{nome:<25} {r['n_iter']:>5} {r['f_opt']:>14.6e} "
          f"{r['gnorm_final']:>14.6e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}")

# ──────────────────────────────────────────────
# Figura 1 — Convergência
# ──────────────────────────────────────────────
cores = ["tab:blue", "tab:orange", "tab:green"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Problema 2 — Item b)  Convergência dos métodos", fontsize=13)

for (nome, r), cor in zip(metodos.items(), cores):
    it = np.arange(len(r["hist_f"]))
    axes[0].semilogy(it, r["hist_f"],     color=cor, lw=1.5, label=nome)
    axes[1].semilogy(it, r["hist_gnorm"], color=cor, lw=1.5, label=nome)

axes[0].set_title("f(xₖ) vs Iteração")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("f(xₖ)  [log]")
axes[0].legend(fontsize=9); axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].set_title("‖∇f(xₖ)‖ vs Iteração")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tolerância 1e-6")
axes[1].legend(fontsize=9); axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob2_b_convergencia.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Curvas de nível + trajetórias
# ──────────────────────────────────────────────
X  = np.linspace(-0.5, 2.5, 400)
Y  = np.linspace(-0.5, 4.0, 400)
Xg, Yg = np.meshgrid(X, Y)
Z  = (Xg + 2*Yg - 7)**2 + (2*Xg + Yg - 5)**2
levels = np.logspace(-4, 2, 25)

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Problema 2 — Item b)  Curvas de nível + Trajetórias", fontsize=13)

for ax, (nome, r), cor in zip(axes2, metodos.items(), cores):
    ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.55)
    traj = r["hist_x"]
    step = max(1, len(traj) // 500)
    ax.plot(traj[::step, 0], traj[::step, 1], "-o", color=cor, lw=1.2, ms=3)
    ax.plot(traj[0, 0],  traj[0, 1],  "ks", ms=7, label="x₀=(0,0)")
    ax.plot(traj[-1, 0], traj[-1, 1], "^",  color=cor, ms=7, label="x_final")
    ax.plot(1, 3, "r*", ms=12, label="x*=(1,3)")
    ax.set_title(f"{nome}\n({r['n_iter']} iterações)", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7); ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("prob2_b_trajetorias.png", dpi=150)
plt.close()