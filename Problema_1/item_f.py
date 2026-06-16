import numpy as np
import matplotlib.pyplot as plt

from item_a import f, grad_f, hess_f
from item_b import gradiente_descendente, passo_constante, armijo, backtracking
from item_c import newton

# ──────────────────────────────────────────────
# Item f) Comparação GD vs Newton
# ──────────────────────────────────────────────
x0 = np.array([1.0, 1.0])

resultados = {
    # Gradiente Descendente
    "GD + Passo Constante": gradiente_descendente(
        f, grad_f, x0, passo_constante, alpha=9.99e-4),
    "GD + Armijo":          gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "GD + Backtracking":    gradiente_descendente(
        f, grad_f, x0, backtracking, alpha0=1.0, rho=0.8, c=1e-4),
    # Newton
    "Newton + Passo Const.": newton(
        f, grad_f, hess_f, x0, passo_constante, alpha=1.0),
    "Newton + Armijo":        newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Newton + Backtracking":  newton(
        f, grad_f, hess_f, x0, backtracking, alpha0=1.0, rho=0.8, c=1e-4),
}

estilos = {
    "GD + Passo Constante":  ("tab:blue",   "-",  "o"),
    "GD + Armijo":           ("tab:orange", "-",  "s"),
    "GD + Backtracking":     ("tab:green",  "-",  "^"),
    "Newton + Passo Const.": ("tab:blue",   "--", "o"),
    "Newton + Armijo":       ("tab:orange", "--", "s"),
    "Newton + Backtracking": ("tab:green",  "--", "^"),
}

# ──────────────────────────────────────────────
# Tabela comparativa completa
# ──────────────────────────────────────────────
print("=" * 90)
print("Problema 1 — Item f)  Comparação: Gradiente Descendente vs Newton")
print("=" * 90)
print(f"{'Método':<28} {'Iter':>6} {'f*':>14} {'‖∇f‖':>14} "
      f"{'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 90)
for nome, r in resultados.items():
    print(f"{nome:<28} {r['n_iter']:>6} {r['f_opt']:>14.4e} "
          f"{r['gnorm_final']:>14.4e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}")

# Razão de iterações
iter_gd  = np.mean([r["n_iter"] for k, r in resultados.items() if k.startswith("GD")])
iter_nt  = np.mean([r["n_iter"] for k, r in resultados.items() if k.startswith("Newton")])
print(f"\nMédia de iterações — GD: {iter_gd:.0f}  |  Newton: {iter_nt:.0f}")
print(f"GD precisa de ~{iter_gd/iter_nt:.0f}x mais iterações que Newton.")

# ──────────────────────────────────────────────
# Figura 1 — Convergência: f(xₖ) e ‖∇f‖ vs iteração
#            (escala log no eixo x para ver Newton)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Problema 1 — Item f)  GD vs Newton: Convergência", fontsize=13)

for nome, r in resultados.items():
    cor, ls, mk = estilos[nome]
    it = np.arange(len(r["hist_f"]))
    # Newton tem só 2 pontos — destaca com marcador maior
    ms = 8 if nome.startswith("Newton") else 3
    mev = 1 if nome.startswith("Newton") else max(1, len(it) // 200)
    axes[0].semilogy(it[::mev], r["hist_f"][::mev],
                     color=cor, ls=ls, lw=1.5, marker=mk,
                     ms=ms, markevery=1 if nome.startswith("Newton") else 50,
                     label=nome)
    axes[1].semilogy(it[::mev], r["hist_gnorm"][::mev],
                     color=cor, ls=ls, lw=1.5, marker=mk,
                     ms=ms, markevery=1 if nome.startswith("Newton") else 50,
                     label=nome)

for ax, title, ylabel in zip(
    axes,
    ["f(xₖ) vs Iteração", "‖∇f(xₖ)‖ vs Iteração"],
    ["f(xₖ)  [log]",      "‖∇f(xₖ)‖  [log]"]
):
    ax.set_title(title)
    ax.set_xlabel("Iteração")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tol = 1e-6")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("prob1_f_convergencia_gd_vs_newton.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Zoom nas primeiras 20 iterações
#            para ver a diferença inicial
# ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Problema 1 — Item f)  GD vs Newton: Zoom nas primeiras 20 iterações",
              fontsize=13)

for nome, r in resultados.items():
    cor, ls, mk = estilos[nome]
    n_plot = min(20, len(r["hist_f"]))
    it = np.arange(n_plot)
    axes2[0].semilogy(it, r["hist_f"][:n_plot],
                      color=cor, ls=ls, lw=1.5, marker=mk, ms=6, label=nome)
    axes2[1].semilogy(it, r["hist_gnorm"][:n_plot],
                      color=cor, ls=ls, lw=1.5, marker=mk, ms=6, label=nome)

for ax, title, ylabel in zip(
    axes2,
    ["f(xₖ) — primeiras 20 iterações", "‖∇f(xₖ)‖ — primeiras 20 iterações"],
    ["f(xₖ)  [log]",                    "‖∇f(xₖ)‖  [log]"]
):
    ax.set_title(title)
    ax.set_xlabel("Iteração")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

axes2[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tol = 1e-6")
axes2[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("prob1_f_zoom_primeiras_iteracoes.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 3 — Curvas de nível: GD vs Newton lado a lado
# ──────────────────────────────────────────────
X  = np.linspace(-1.1, 1.1, 400)
Y  = np.linspace(-1.1, 1.1, 400)
Xg, Yg = np.meshgrid(X, Y)
Z  = 1000 * Xg**2 + Yg**2
levels = np.logspace(-6, 3, 30)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Problema 1 — Item f)  Trajetórias: GD vs Newton nas curvas de nível",
              fontsize=12)

familias = {
    "Gradiente Descendente": ["GD + Passo Constante", "GD + Armijo", "GD + Backtracking"],
    "Newton":                ["Newton + Passo Const.", "Newton + Armijo", "Newton + Backtracking"],
}
cores_familia = ["tab:blue", "tab:orange", "tab:green"]

for ax, (familia, nomes) in zip(axes3, familias.items()):
    ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.45)
    for nome, cor in zip(nomes, cores_familia):
        r = resultados[nome]
        traj = r["hist_x"]
        step = max(1, len(traj) // 500)
        label_step = nome.split("+")[1].strip()
        ax.plot(traj[::step, 0], traj[::step, 1],
                color=cor, lw=1.3,
                label=f"{label_step} ({r['n_iter']} it.)")
    ax.plot(x0[0], x0[1], "ks", ms=8, zorder=5, label="x₀ = (1,1)")
    ax.plot(0, 0, "r*", ms=12, zorder=5, label="x* = (0,0)")
    ax.set_title(familia, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("prob1_f_trajetorias_gd_vs_newton.png", dpi=150)
plt.close()

print("\nGráficos salvos com sucesso.")