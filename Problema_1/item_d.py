import numpy as np
import matplotlib.pyplot as plt

from problema_1.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, passo_constante, armijo, backtracking
from problema_1.item_c import newton

# ──────────────────────────────────────────────
# Ponto inicial
# ──────────────────────────────────────────────
x0 = np.array([1.0, 1.0])

# ──────────────────────────────────────────────
# Execução de todos os métodos
# ──────────────────────────────────────────────
resultados = {
    "GD + Passo Constante": gradiente_descendente(
        f, grad_f, x0, passo_constante, alpha=9.99e-4),
    "GD + Armijo":          gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "GD + Backtracking":    gradiente_descendente(
        f, grad_f, x0, backtracking, alpha0=1.0, rho=0.8, c=1e-4),
    "Newton + Passo Const.": newton(
        f, grad_f, hess_f, x0, passo_constante, alpha=1.0),
    "Newton + Armijo":       newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Newton + Backtracking": newton(
        f, grad_f, hess_f, x0, backtracking, alpha0=1.0, rho=0.8, c=1e-4),
}

cores = {
    "GD + Passo Constante":  ("tab:blue",   "-"),
    "GD + Armijo":           ("tab:orange", "-"),
    "GD + Backtracking":     ("tab:green",  "-"),
    "Newton + Passo Const.": ("tab:blue",   "--"),
    "Newton + Armijo":       ("tab:orange", "--"),
    "Newton + Backtracking": ("tab:green",  "--"),
}

# ──────────────────────────────────────────────
# Figura 1 — f(xₖ) vs iteração (todos juntos)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Problema 1 — Item d)  Comparação de Trajetórias", fontsize=13)

for nome, r in resultados.items():
    cor, ls = cores[nome]
    it = np.arange(len(r["hist_f"]))
    axes[0].semilogy(it, r["hist_f"],     color=cor, ls=ls, lw=1.5, label=nome)
    axes[1].semilogy(it, r["hist_gnorm"], color=cor, ls=ls, lw=1.5, label=nome)

axes[0].set_title("f(xₖ) vs Iteração")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("f(xₖ)  [log]")
axes[0].legend(fontsize=8); axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].set_title("‖∇f(xₖ)‖ vs Iteração")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tolerância 1e-6")
axes[1].legend(fontsize=8); axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob1_d_convergencia_comparacao.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Curvas de nível + trajetórias (todos juntos)
# ──────────────────────────────────────────────
X  = np.linspace(-1.1, 1.1, 400)
Y  = np.linspace(-1.1, 1.1, 400)
Xg, Yg = np.meshgrid(X, Y)
Z  = 1000 * Xg**2 + Yg**2
levels = np.logspace(-6, 3, 30)

fig2, ax = plt.subplots(figsize=(8, 7))
fig2.suptitle("Problema 1 — Item d)  Curvas de nível + Trajetórias (todos os métodos)",
              fontsize=12)

ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.45)

for nome, r in resultados.items():
    cor, ls = cores[nome]
    traj = r["hist_x"]
    step = max(1, len(traj) // 500)
    ax.plot(traj[::step, 0], traj[::step, 1],
            color=cor, ls=ls, lw=1.2, label=f"{nome} ({r['n_iter']} it.)")

ax.plot(x0[0], x0[1], "ks", ms=8, zorder=5, label="x₀ = (1,1)")
ax.plot(0, 0, "r*", ms=12, zorder=5, label="x* = (0,0)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, ls="--", alpha=0.3)
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("prob1_d_trajetorias_comparacao.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 3 — Curvas de nível separadas por família
# ──────────────────────────────────────────────
familias = {
    "Gradiente Descendente": ["GD + Passo Constante", "GD + Armijo", "GD + Backtracking"],
    "Newton":                ["Newton + Passo Const.", "Newton + Armijo", "Newton + Backtracking"],
}

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Problema 1 — Item d)  Trajetórias por família de método", fontsize=13)

for ax, (familia, nomes) in zip(axes3, familias.items()):
    ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.45)
    for nome in nomes:
        r = resultados[nome]
        cor, ls = cores[nome]
        traj = r["hist_x"]
        step = max(1, len(traj) // 500)
        ax.plot(traj[::step, 0], traj[::step, 1],
                color=cor, ls="-", lw=1.3,
                label=f"{nome.split('+')[1].strip()} ({r['n_iter']} it.)")
    ax.plot(x0[0], x0[1], "ks", ms=8, zorder=5, label="x₀")
    ax.plot(0, 0, "r*", ms=12, zorder=5, label="x*")
    ax.set_title(familia, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("prob1_d_trajetorias_por_familia.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Tabela comparativa final
# ──────────────────────────────────────────────
print("=" * 85)
print("Problema 1 — Item d)  Comparação geral")
print("=" * 85)
print(f"{'Método':<28} {'Iter':>6} {'f*':>12} {'‖∇f‖':>12} {'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 85)
for nome, r in resultados.items():
    print(f"{nome:<28} {r['n_iter']:>6} {r['f_opt']:>12.4e} "
          f"{r['gnorm_final']:>12.4e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}")

print("\nGráficos salvos com sucesso.")