import numpy as np
import matplotlib.pyplot as plt

from problema_3.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo
from problema_1.item_c import newton

# ──────────────────────────────────────────────
# Problema 3 — Item b)
# GD + Armijo e Newton, testados em x0 = (1,1)
# ──────────────────────────────────────────────
x0 = np.array([1.0, 1.0])

metodos = {
    "GD + Armijo": gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Newton + Armijo": newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
}

# ──────────────────────────────────────────────
# Tabela de resultados
# ──────────────────────────────────────────────
print("=" * 80)
print("Problema 3 — Item b)  Função de Beale  (x0 = [1, 1])")
print("=" * 80)
print(f"{'Método':<20} {'Iter':>5} {'f*':>14} {'‖∇f‖':>14} "
      f"{'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 80)
for nome, r in metodos.items():
    print(f"{nome:<20} {r['n_iter']:>5} {r['f_opt']:>14.6e} "
          f"{r['gnorm_final']:>14.6e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}")
    print(f"  x_final = {r['x_opt']}")

print("\n*** ATENÇÃO ***")
print("Newton parou prematuramente em x0=(1,1) pois a Hessiana é INDEFINIDA")
print("nesse ponto (autovalores: -9.83 e 78.33). A direção de Newton resultante")
print("é ortogonal ao gradiente (g·d ≈ 0), então a busca de Armijo falha em")
print("encontrar redução suficiente e alpha colapsa para um valor ínfimo,")
print("fazendo o algoritmo estagnar em x_final = (0, 1), que NÃO é o mínimo")
print("global. Isso ilustra uma limitação clássica do método de Newton puro")
print("em regiões não-convexas (será discutido no item e).")

# ──────────────────────────────────────────────
# Figura 1 — Convergência
# ──────────────────────────────────────────────
cores = ["tab:blue", "tab:orange"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Problema 3 — Item b)  Convergência (x0 = [1, 1])", fontsize=13)

for (nome, r), cor in zip(metodos.items(), cores):
    it = np.arange(len(r["hist_f"]))
    ms  = 6 if nome.startswith("Newton") else 2
    axes[0].semilogy(it, r["hist_f"],     color=cor, lw=1.5, marker="o", ms=ms, label=nome)
    axes[1].semilogy(it, r["hist_gnorm"], color=cor, lw=1.5, marker="o", ms=ms, label=nome)

axes[0].set_title("f(xₖ) vs Iteração")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("f(xₖ)  [log]")
axes[0].legend(fontsize=9); axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].set_title("‖∇f(xₖ)‖ vs Iteração")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tol = 1e-6")
axes[1].legend(fontsize=9); axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob3_b_convergencia.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Curvas de nível + trajetórias
# ──────────────────────────────────────────────
X  = np.linspace(-1, 4, 500)
Y  = np.linspace(-1, 2, 500)
Xg, Yg = np.meshgrid(X, Y)
Zg = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        Zg[i, j] = f([Xg[i, j], Yg[i, j]])

levels = np.logspace(-2, 3, 30)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6))
fig2.suptitle("Problema 3 — Item b)  Curvas de nível + Trajetórias (x0 = [1, 1])",
              fontsize=12)

for ax, (nome, r), cor in zip(axes2, metodos.items(), cores):
    ax.contour(Xg, Yg, Zg, levels=levels, cmap="coolwarm", alpha=0.6)
    traj = r["hist_x"]
    step = max(1, len(traj) // 300)
    ax.plot(traj[::step, 0], traj[::step, 1], "-o", color=cor, lw=1.3, ms=4)
    ax.plot(traj[0, 0],  traj[0, 1],  "ks", ms=8, label="x₀=(1,1)")
    ax.plot(traj[-1, 0], traj[-1, 1], "^",  color=cor, ms=8, label="x_final")
    ax.plot(3, 0.5, "r*", ms=14, label="x*=(3,0.5)")
    ax.set_title(f"{nome}\n({r['n_iter']} iterações)", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("prob3_b_trajetorias.png", dpi=150)
plt.close()