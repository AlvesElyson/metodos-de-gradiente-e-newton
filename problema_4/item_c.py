import numpy as np
import matplotlib.pyplot as plt

from problema_4.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo
from problema_1.item_c import newton

# Problema 4 — Item c) ──────────────────────────────────────────────
# Pontos iniciais: (0,0), (2,-2), (-3,-3)

pontos_iniciais = {
    "(0, 0)":   np.array([0.0, 0.0]),
    "(2, -2)":  np.array([2.0, -2.0]),
    "(-3, -3)": np.array([-3.0, -3.0]),
}

resultados = {}
for nome, x0 in pontos_iniciais.items():
    resultados[nome] = {
        "GD + Armijo": gradiente_descendente(
            f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
        "Newton + Armijo": newton(
            f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    }

# Tabela de resultados ──────────────────────────────────────────────

print("=" * 95)
print("Problema 4 — Item c)  Resultados para os três pontos iniciais")
print("=" * 95)
print(f"{'x0':<10} {'Método':<18} {'Iter':>5} {'f*':>12} {'‖∇f‖':>12} "
      f"{'x_final':>18}  {'Status'}")
print("-" * 95)

for nome_x0, metodos in resultados.items():
    for metodo, r in metodos.items():
        status = r.get("status", "convergiu" if r["gnorm_final"] <= 1e-6 else "limite iter.")
        print(f"{nome_x0:<10} {metodo:<18} {r['n_iter']:>5} {r['f_opt']:>12.4e} "
              f"{r['gnorm_final']:>12.4e}   {str(np.round(r['x_opt'],3)):<16} {status}")
    print("-" * 95)

# Figura — Curvas de nível + trajetórias para cada x0 ──────────────────────────────────────────────

X  = np.linspace(-8, 6, 600)
Y  = np.linspace(-8, 6, 600)
Xg, Yg = np.meshgrid(X, Y)
Zg = np.sin(Xg + Yg) + (Xg - Yg)**2 - 1.5*Xg + 2.5*Yg + 1

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Problema 4 — Item c)  Trajetórias do GD+Armijo por ponto inicial", fontsize=13)

for ax, (nome_x0, x0) in zip(axes, pontos_iniciais.items()):
    cs = ax.contour(Xg, Yg, Zg, levels=30, cmap="coolwarm", alpha=0.6)
    r = resultados[nome_x0]["GD + Armijo"]
    traj = r["hist_x"]
    step = max(1, len(traj) // 300)
    ax.plot(traj[::step, 0], traj[::step, 1], "-o", color="tab:blue", lw=1.3, ms=4)
    ax.plot(x0[0], x0[1], "ks", ms=9, label=f"x₀={nome_x0}")
    ax.plot(traj[-1, 0], traj[-1, 1], "g^", ms=9, label=f"x_final (f={r['f_opt']:.3f})")
    ax.set_title(f"x₀={nome_x0}  ({r['n_iter']} it.)", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("prob4_c_trajetorias_gd.png", dpi=150)
plt.close()