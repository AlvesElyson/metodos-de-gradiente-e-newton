import numpy as np
import matplotlib.pyplot as plt

from problema_4.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo
from problema_1.item_c import newton

# Problema 4 — Item b)
# Testando GD+Armijo e Newton+Armijo no ponto x0 = (0,0)
# (os demais pontos serão sistematicamente cobertos no item c)

x0 = np.array([0.0, 0.0])

metodos = {
    "GD + Armijo":     gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Newton + Armijo": newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
}

# Tabela de resultados

print("=" * 80)
print("Problema 4 — Item b)  (x0 = [0, 0])")
print("=" * 80)
print(f"{'Método':<20} {'Iter':>5} {'f*':>14} {'‖∇f‖':>14} "
      f"{'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 80)
for nome, r in metodos.items():
    print(f"{nome:<20} {r['n_iter']:>5} {r['f_opt']:>14.6e} "
          f"{r['gnorm_final']:>14.6e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}")
    print(f"  x_final = {r['x_opt']}   status: {r.get('status', 'N/A')}")

# Figura 1 — Convergência

cores = ["tab:blue", "tab:orange"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Problema 4 — Item b)  Convergência (x0 = [0, 0])", fontsize=13)

for (nome, r), cor in zip(metodos.items(), cores):
    it = np.arange(len(r["hist_f"]))
    ms = 6 if nome.startswith("Newton") else 2
    axes[0].plot(it, r["hist_f"], color=cor, lw=1.5, marker="o", ms=ms, label=nome)
    axes[1].semilogy(it, r["hist_gnorm"] + 1e-16, color=cor, lw=1.5, marker="o", ms=ms, label=nome)

axes[0].set_title("f(xₖ) vs Iteração")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("f(xₖ)")
axes[0].legend(fontsize=9); axes[0].grid(True, ls="--", alpha=0.4)

axes[1].set_title("‖∇f(xₖ)‖ vs Iteração")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tol = 1e-6")
axes[1].legend(fontsize=9); axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob4_b_convergencia.png", dpi=150)
plt.close()

# Figura 2 — Curvas de nível + trajetórias

X  = np.linspace(-6, 6, 500)
Y  = np.linspace(-6, 6, 500)
Xg, Yg = np.meshgrid(X, Y)
Zg = np.sin(Xg + Yg) + (Xg - Yg)**2 - 1.5*Xg + 2.5*Yg + 1

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6))
fig2.suptitle("Problema 4 — Item b)  Curvas de nível + Trajetórias (x0 = [0, 0])", fontsize=12)

for ax, (nome, r) in zip(axes2, metodos.items()):
    cor = cores[list(metodos.keys()).index(nome)]
    cs = ax.contour(Xg, Yg, Zg, levels=30, cmap="coolwarm", alpha=0.6)
    traj = r["hist_x"]
    step = max(1, len(traj) // 300)
    ax.plot(traj[::step, 0], traj[::step, 1], "-o", color=cor, lw=1.3, ms=4)
    ax.plot(traj[0, 0],  traj[0, 1],  "ks", ms=8, label="x₀=(0,0)")
    ax.plot(traj[-1, 0], traj[-1, 1], "^",  color=cor, ms=8, label="x_final")
    ax.set_title(f"{nome}\n({r['n_iter']} iterações, f*={r['f_opt']:.4f})", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("prob4_b_trajetorias.png", dpi=150)
plt.close()