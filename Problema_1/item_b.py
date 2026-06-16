import numpy as np
import matplotlib.pyplot as plt

from item_a import f, grad_f, hess_f

# ──────────────────────────────────────────────
# Estratégias de passo
# ──────────────────────────────────────────────

def passo_constante(x, d, f, grad, alpha=9.99e-4, **kwargs):
    """Passo constante α ≈ 2/(λ_min + λ_max) = 1/1001 ≈ 9.99e-4 (valor ótimo teórico)."""
    return alpha, 1

def armijo(x, d, f, grad, alpha0=1.0, sigma=0.5, c=1e-4, **kwargs):
    """Condição de Armijo (sufficient decrease)."""
    alpha = alpha0
    fx    = f(x)
    slope = grad(x) @ d
    evals = 0
    while True:
        evals += 1
        if f(x + alpha * d) <= fx + c * alpha * slope:
            break
        alpha *= sigma
        if alpha < 1e-16:
            break
    return alpha, evals

def backtracking(x, d, f, grad, alpha0=1.0, rho=0.8, c=1e-4, **kwargs):
    """Backtracking line search (redução geométrica com fator rho)."""
    alpha = alpha0
    fx    = f(x)
    slope = grad(x) @ d
    evals = 0
    while True:
        evals += 1
        if f(x + alpha * d) <= fx + c * alpha * slope:
            break
        alpha *= rho
        if alpha < 1e-16:
            break
    return alpha, evals

# ──────────────────────────────────────────────
# Gradiente Descendente genérico
# ──────────────────────────────────────────────

def gradiente_descendente(f, grad, x0, step_strategy,
                          tol=1e-6, max_iter=1000, **step_kwargs):
    x = x0.copy().astype(float)
    hist_f     = [f(x)]
    hist_gnorm = [np.linalg.norm(grad(x))]
    hist_x     = [x.copy()]
    n_evals_f  = 1
    n_evals_g  = 1

    for k in range(max_iter):
        g = grad(x);  n_evals_g += 1
        if np.linalg.norm(g) <= tol:
            break
        d = -g
        alpha, ls_evals = step_strategy(x, d, f=f, grad=grad, **step_kwargs)
        n_evals_f += ls_evals
        x = x + alpha * d
        n_evals_f += 1;  n_evals_g += 1
        hist_f.append(f(x))
        hist_gnorm.append(np.linalg.norm(grad(x)))
        hist_x.append(x.copy())

    return {
        "x_opt":       x,
        "f_opt":       f(x),
        "gnorm_final": np.linalg.norm(grad(x)),
        "n_iter":      len(hist_f) - 1,
        "n_evals_f":   n_evals_f,
        "n_evals_g":   n_evals_g,
        "n_evals_H":   0,
        "hist_f":      np.array(hist_f),
        "hist_gnorm":  np.array(hist_gnorm),
        "hist_x":      np.array(hist_x),
    }

# ──────────────────────────────────────────────
# Execução
# ──────────────────────────────────────────────
x0 = np.array([1.0, 1.0])

metodos = {
    "Passo Constante (α≈1/1001)": gradiente_descendente(
        f, grad_f, x0, passo_constante, alpha=9.99e-4),
    "Armijo (σ=0.5)":            gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Backtracking (ρ=0.8)":      gradiente_descendente(
        f, grad_f, x0, backtracking, alpha0=1.0, rho=0.8, c=1e-4),
}

# ──────────────────────────────────────────────
# Tabela de resultados
# ──────────────────────────────────────────────
print("=" * 75)
print("Problema 1 — Item b)  Gradiente Descendente  (x0 = [1, 1])")
print("=" * 75)
print(f"{'Método':<30} {'Iter':>6} {'f*':>12} {'‖∇f‖':>12} {'Aval f':>8} {'Aval g':>8}")
print("-" * 75)
for nome, r in metodos.items():
    print(f"{nome:<30} {r['n_iter']:>6} {r['f_opt']:>12.4e} "
          f"{r['gnorm_final']:>12.4e} {r['n_evals_f']:>8} {r['n_evals_g']:>8}")

# ──────────────────────────────────────────────
# Figura 1 — Convergência (f e ‖∇f‖ vs iteração)
# ──────────────────────────────────────────────
cores = ["tab:blue", "tab:orange", "tab:green"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Problema 1 — Item b)  Gradiente Descendente", fontsize=13)

for (nome, r), cor in zip(metodos.items(), cores):
    it = np.arange(len(r["hist_f"]))
    axes[0].semilogy(it, r["hist_f"],     color=cor, label=nome, lw=1.5)
    axes[1].semilogy(it, r["hist_gnorm"], color=cor, label=nome, lw=1.5)

axes[0].set_title("f(xₖ) vs Iteração")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("f(xₖ)  [log]")
axes[0].legend(fontsize=8); axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].set_title("‖∇f(xₖ)‖ vs Iteração")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].axhline(1e-6, color="red", ls=":", lw=1.5, label="tolerância 1e-6")
axes[1].legend(fontsize=8); axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob1_b_convergencia.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Curvas de nível + trajetórias
# ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Problema 1 — Item b)  Curvas de nível + Trajetórias", fontsize=13)

X  = np.linspace(-1.1, 1.1, 400)
Y  = np.linspace(-1.1, 1.1, 400)
Xg, Yg = np.meshgrid(X, Y)
Z  = 1000 * Xg**2 + Yg**2
levels = np.logspace(-6, 3, 30)

for ax, (nome, r), cor in zip(axes2, metodos.items(), cores):
    ax.contour(Xg, Yg, Z, levels=levels, cmap="coolwarm", alpha=0.55)
    traj = r["hist_x"]
    step = max(1, len(traj) // 500)
    ax.plot(traj[::step, 0], traj[::step, 1], "-", color=cor, lw=0.8, alpha=0.8)
    ax.plot(traj[0, 0],  traj[0, 1],  "ks", ms=7, label="x₀=(1,1)")
    ax.plot(traj[-1, 0], traj[-1, 1], "^",  color=cor, ms=7, label="x_final")
    ax.plot(0, 0, "r*", ms=10, label="x*=(0,0)")
    ax.set_title(nome, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7); ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("prob1_b_trajetorias.png", dpi=150)
plt.close()

print("\nGráficos salvos com sucesso.")