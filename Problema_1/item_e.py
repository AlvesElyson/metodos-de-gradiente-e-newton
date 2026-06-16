import numpy as np
import matplotlib.pyplot as plt

from item_b import gradiente_descendente, armijo

# ──────────────────────────────────────────────
# Item e) Efeito do condicionamento da Hessiana
#
# Consideramos a família de funções quadráticas:
#   f_κ(x, y) = κ * x² + y²
# onde κ é o número de condicionamento da Hessiana.
#
# Hessiana: H = [[2κ, 0], [0, 2]]
# Autovalores: λ_min = 2, λ_max = 2κ
# Condicionamento: κ(H) = λ_max / λ_min = κ
#
# Taxa de convergência teórica do GD com passo ótimo:
#   q = (κ - 1) / (κ + 1)
# Quanto maior κ, mais próximo q está de 1 → convergência mais lenta.
# ──────────────────────────────────────────────

x0 = np.array([1.0, 1.0])
kappas = [1, 5, 10, 50, 100, 500, 1000]

resultados_kappa = {}

for kappa in kappas:
    # Define funções específicas para este κ
    def make_funcs(k):
        def f_k(x):
            return k * x[0]**2 + x[1]**2
        def grad_k(x):
            return np.array([2 * k * x[0], 2 * x[1]])
        return f_k, grad_k

    f_k, grad_k = make_funcs(kappa)

    # Passo ótimo teórico: α* = 2 / (λ_min + λ_max) = 1 / (1 + κ)
    alpha_opt = 1.0 / (1.0 + kappa)

    r = gradiente_descendente(f_k, grad_k, x0, armijo,
                              alpha0=1.0, sigma=0.5, c=1e-4,
                              tol=1e-6, max_iter=1000)

    # Taxa teórica de convergência
    q_teorico = (kappa - 1) / (kappa + 1)

    resultados_kappa[kappa] = {
        **r,
        "q_teorico": q_teorico,
        "alpha_opt": alpha_opt,
    }

# ──────────────────────────────────────────────
# Tabela de resultados
# ──────────────────────────────────────────────
print("=" * 75)
print("Problema 1 — Item e)  Efeito do condicionamento na convergência (GD + Armijo)")
print("=" * 75)
print(f"{'κ(H)':>8} {'q teórico':>12} {'Iterações':>10} {'f*':>14} {'‖∇f‖ final':>14}")
print("-" * 75)
for kappa, r in resultados_kappa.items():
    print(f"{kappa:>8} {r['q_teorico']:>12.6f} {r['n_iter']:>10} "
          f"{r['f_opt']:>14.4e} {r['gnorm_final']:>14.4e}")

# ──────────────────────────────────────────────
# Figura 1 — ‖∇f(xₖ)‖ vs iteração para cada κ
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Problema 1 — Item e)  Efeito do condicionamento na velocidade de convergência",
             fontsize=12)

cmap = plt.cm.plasma
cores = [cmap(i / (len(kappas) - 1)) for i in range(len(kappas))]

for (kappa, r), cor in zip(resultados_kappa.items(), cores):
    it = np.arange(len(r["hist_gnorm"]))
    axes[0].semilogy(it, r["hist_gnorm"], color=cor, lw=1.5, label=f"κ = {kappa}")
    axes[1].semilogy(it, r["hist_f"],     color=cor, lw=1.5, label=f"κ = {kappa}")

for ax, ylabel, title in zip(
    axes,
    ["‖∇f(xₖ)‖  [log]", "f(xₖ)  [log]"],
    ["‖∇f(xₖ)‖ vs Iteração", "f(xₖ) vs Iteração"]
):
    ax.axhline(1e-6, color="red", ls=":", lw=1.2, label="tol = 1e-6")
    ax.set_title(title)
    ax.set_xlabel("Iteração")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob1_e_efeito_kappa.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Iterações necessárias vs κ
# ──────────────────────────────────────────────
kappas_arr  = np.array(list(resultados_kappa.keys()))
iters_arr   = np.array([r["n_iter"] for r in resultados_kappa.values()])
q_arr       = np.array([r["q_teorico"] for r in resultados_kappa.values()])

# Estimativa teórica de iterações para reduzir ‖∇f‖ por fator 1e6
# usando q: N ≈ log(1e6) / log(1/q)
iters_teo = np.where(q_arr > 0,
                     np.log(1e6) / np.log(1.0 / np.maximum(q_arr, 1e-12)),
                     1)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Problema 1 — Item e)  Iterações vs número de condicionamento", fontsize=12)

axes2[0].plot(kappas_arr, iters_arr,  "o-", color="tab:blue",   lw=2, ms=7, label="Iterações (GD + Armijo)")
axes2[0].plot(kappas_arr, iters_teo,  "s--", color="tab:orange", lw=1.5, ms=6, label="Estimativa teórica")
axes2[0].set_xlabel("κ(H)  (número de condicionamento)")
axes2[0].set_ylabel("Número de iterações")
axes2[0].set_title("Iterações necessárias para convergir")
axes2[0].legend(fontsize=9)
axes2[0].grid(True, ls="--", alpha=0.4)

axes2[1].plot(kappas_arr, q_arr, "o-", color="tab:red", lw=2, ms=7)
axes2[1].set_xlabel("κ(H)  (número de condicionamento)")
axes2[1].set_ylabel("q = (κ−1)/(κ+1)")
axes2[1].set_title("Taxa de convergência teórica q\n(quanto mais próximo de 1, mais lento)")
axes2[1].grid(True, ls="--", alpha=0.4)
axes2[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("prob1_e_iters_vs_kappa.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 3 — Curvas de nível para κ = 1, 10, 100, 1000
# ──────────────────────────────────────────────
kappas_plot = [1, 10, 100, 1000]
fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4))
fig3.suptitle("Problema 1 — Item e)  Curvas de nível f_κ(x,y) = κx² + y²  para diferentes κ",
              fontsize=11)

X = np.linspace(-1.1, 1.1, 400)
Y = np.linspace(-1.1, 1.1, 400)
Xg, Yg = np.meshgrid(X, Y)

for ax, kappa in zip(axes3, kappas_plot):
    Z = kappa * Xg**2 + Yg**2
    levels = np.logspace(-3, 3, 20)
    ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.7)

    r = resultados_kappa[kappa]
    traj = r["hist_x"]
    step = max(1, len(traj) // 300)
    ax.plot(traj[::step, 0], traj[::step, 1], "-", color="black", lw=0.8, alpha=0.7)
    ax.plot(traj[0, 0], traj[0, 1], "ks", ms=6, label="x₀")
    ax.plot(0, 0, "r*", ms=10, label="x*")
    ax.set_title(f"κ = {kappa}\n({r['n_iter']} iterações)", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("prob1_e_curvas_nivel_kappa.png", dpi=150)
plt.close()

print("\nGráficos salvos com sucesso.")