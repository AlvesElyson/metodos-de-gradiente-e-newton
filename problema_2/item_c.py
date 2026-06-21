import numpy as np

from item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, passo_constante, armijo
from problema_1.item_c import newton

# Problema 2 — Item c) ──────────────────────────────────────────────
# Ponto inicial: x0 = (0, 0)^T

x0 = np.array([0.0, 0.0])

alpha_otimo = 2.0 / (2.0 + 18.0)  # 2/(λ_min + λ_max)

metodos = {
    "GD + Passo Constante": gradiente_descendente(
        f, grad_f, x0, passo_constante, alpha=alpha_otimo),
    "GD + Armijo":          gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    "Newton + Armijo":      newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
}

print("=" * 80)
print("Problema 2 — Item c)  Ponto inicial x0 = (0, 0)^T")
print("=" * 80)
print(f"\nf(x0)    = {f(x0):.4f}")
print(f"∇f(x0)  = {grad_f(x0)}")
print(f"‖∇f(x0)‖ = {np.linalg.norm(grad_f(x0)):.4f}")
print(f"\nMínimo esperado: x* = (1, 3),  f(x*) = 0\n")
print(f"{'Método':<25} {'Iter':>5} {'f*':>14} {'‖∇f‖':>14} "
      f"{'Aval f':>8} {'Aval g':>8} {'Aval H':>8}")
print("-" * 80)
for nome, r in metodos.items():
    status = "✓" if r["gnorm_final"] <= 1e-6 else "✗ (não convergiu)"
    print(f"{nome:<25} {r['n_iter']:>5} {r['f_opt']:>14.6e} "
          f"{r['gnorm_final']:>14.6e} {r['n_evals_f']:>8} "
          f"{r['n_evals_g']:>8} {r['n_evals_H']:>8}  {status}")
    print(f"  x_final = {r['x_opt']}")