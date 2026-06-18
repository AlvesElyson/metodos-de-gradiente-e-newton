import numpy as np

from problema_3.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo
from problema_1.item_c import newton

# ──────────────────────────────────────────────
# Problema 3 — Item c)
# Pontos iniciais: (1,1)^T  e  (3, 0.5)^T
# ──────────────────────────────────────────────
pontos_iniciais = {
    "x0 = (1, 1)":     np.array([1.0, 1.0]),
    "x0 = (3, 0.5)":   np.array([3.0, 0.5]),
}

resultados = {}

print("=" * 85)
print("Problema 3 — Item c)  Testando diferentes pontos iniciais")
print("=" * 85)

for nome_x0, x0 in pontos_iniciais.items():
    print(f"\n{'─'*85}")
    print(f"{nome_x0}   →   f(x0) = {f(x0):.6f},  ‖∇f(x0)‖ = {np.linalg.norm(grad_f(x0)):.6f}")

    H0 = hess_f(x0)
    autoval = np.linalg.eigvalsh(H0)
    pd = "definida positiva" if autoval.min() > 0 else "INDEFINIDA"
    print(f"Hessiana em x0: autovalores = {autoval}  →  {pd}")
    print(f"{'─'*85}")

    r_gd = gradiente_descendente(
        f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4)
    r_nt = newton(
        f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4)

    resultados[nome_x0] = {"GD + Armijo": r_gd, "Newton + Armijo": r_nt}

    print(f"{'Método':<20} {'Iter':>5} {'f*':>14} {'‖∇f‖':>14} {'x_final':>20}")
    x_star = np.array([3.0, 0.5])
    for metodo, r in resultados[nome_x0].items():
        dist_para_otimo = np.linalg.norm(r["x_opt"] - x_star)
        if r["gnorm_final"] <= 1e-6 and dist_para_otimo < 1e-3:
            status = "✓ convergiu para o mínimo global"
        elif r["gnorm_final"] <= 1e-6:
            status = "⚠ estacionou em PONTO CRÍTICO ERRADO (sela/máx. local)"
        else:
            status = "✗ não convergiu (limite de iterações)"
        print(f"{metodo:<20} {r['n_iter']:>5} {r['f_opt']:>14.6e} "
              f"{r['gnorm_final']:>14.6e}   {str(np.round(r['x_opt'],4)):<15} {status}")