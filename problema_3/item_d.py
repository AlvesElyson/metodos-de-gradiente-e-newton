import numpy as np
import matplotlib.pyplot as plt

from problema_3.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo
from problema_1.item_c import newton

# ──────────────────────────────────────────────
# Problema 3 — Item d)
# Compara a influência do ponto inicial.
#
# x0=(3,0.5) já É o ótimo (gradiente zero), o que não ilustra bem
# a influência do ponto inicial sobre a trajetória. Por isso,
# além dos dois pontos pedidos, adicionamos x0=(1, 4) como um
# terceiro ponto de teste, mais distante e revelador.
# ──────────────────────────────────────────────

pontos_iniciais = {
    "(1, 1)":   np.array([1.0, 1.0]),
    "(3, 0.5)": np.array([3.0, 0.5]),
    "(1, 4)":   np.array([1.0, 4.0]),   # ponto extra para ilustrar melhor
}

resultados = {}
for nome, x0 in pontos_iniciais.items():
    resultados[nome] = {
        "GD + Armijo": gradiente_descendente(
            f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
        "Newton + Armijo": newton(
            f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    }

# ──────────────────────────────────────────────
# Tabela comparativa
# ──────────────────────────────────────────────
x_star = np.array([3.0, 0.5])
print("=" * 95)
print("Problema 3 — Item d)  Influência do ponto inicial")
print("=" * 95)
print(f"{'x0':<12} {'Método':<18} {'Iter':>5} {'f*':>14} {'‖∇f‖':>12} "
      f"{'x_final':>16}  {'Resultado'}")
print("-" * 95)

for nome_x0, metodos in resultados.items():
    for metodo, r in metodos.items():
        dist = np.linalg.norm(r["x_opt"] - x_star)
        if r["gnorm_final"] <= 1e-6 and dist < 1e-3:
            status = "convergiu p/ mínimo global"
        elif r["gnorm_final"] <= 1e-6:
            status = "ponto crítico ERRADO"
        else:
            status = "não convergiu (limite iter.)"
        print(f"{nome_x0:<12} {metodo:<18} {r['n_iter']:>5} {r['f_opt']:>14.4e} "
              f"{r['gnorm_final']:>12.4e}   {str(np.round(r['x_opt'],3)):<14} {status}")
    print("-" * 95)

# ──────────────────────────────────────────────
# Figura — Curvas de nível com trajetórias dos 3 pontos iniciais
# ──────────────────────────────────────────────
X  = np.linspace(-2, 4.5, 500)
Y  = np.linspace(-1, 4.5, 500)
Xg, Yg = np.meshgrid(X, Y)
Zg = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        Zg[i, j] = f([Xg[i, j], Yg[i, j]])
levels = np.logspace(-2, 3.5, 35)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle("Problema 3 — Item d)  Influência do ponto inicial nas trajetórias", fontsize=13)
cores_pontos = {"(1, 1)": "tab:blue", "(3, 0.5)": "tab:green", "(1, 4)": "tab:purple"}

for ax, metodo in zip(axes, ["GD + Armijo", "Newton + Armijo"]):
    ax.contour(Xg, Yg, Zg, levels=levels, cmap="coolwarm", alpha=0.5)
    for nome_x0, x0 in pontos_iniciais.items():
        r = resultados[nome_x0][metodo]
        traj = r["hist_x"]
        step = max(1, len(traj) // 300)
        cor = cores_pontos[nome_x0]
        ax.plot(traj[::step, 0], traj[::step, 1], "-o",
                color=cor, lw=1.3, ms=3,
                label=f"x₀={nome_x0} ({r['n_iter']} it.)")
        ax.plot(x0[0], x0[1], "s", color=cor, ms=9, markeredgecolor="black")
    ax.plot(3, 0.5, "r*", ms=16, label="x* = (3, 0.5)", zorder=5)
    ax.set_title(metodo, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(-2, 4.5); ax.set_ylim(-1, 4.5)

plt.tight_layout()
plt.savefig("prob3_d_influencia_x0.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura — Convergência para cada ponto inicial (GD)
# ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Problema 3 — Item d)  Convergência do GD+Armijo por ponto inicial", fontsize=12)

for nome_x0, x0 in pontos_iniciais.items():
    r = resultados[nome_x0]["GD + Armijo"]
    it = np.arange(len(r["hist_f"]))
    cor = cores_pontos[nome_x0]
    axes2[0].semilogy(it, r["hist_f"] + 1e-16,     color=cor, lw=1.5, label=f"x₀={nome_x0}")
    axes2[1].semilogy(it, r["hist_gnorm"] + 1e-16, color=cor, lw=1.5, label=f"x₀={nome_x0}")

axes2[0].set_title("f(xₖ) vs Iteração"); axes2[0].set_xlabel("Iteração")
axes2[0].set_ylabel("f(xₖ)  [log]"); axes2[0].legend(fontsize=9)
axes2[0].grid(True, which="both", ls="--", alpha=0.4)

axes2[1].set_title("‖∇f(xₖ)‖ vs Iteração"); axes2[1].set_xlabel("Iteração")
axes2[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes2[1].axhline(1e-6, color="red", ls=":", label="tol=1e-6")
axes2[1].legend(fontsize=9); axes2[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob3_d_convergencia_por_x0.png", dpi=150)
plt.close()