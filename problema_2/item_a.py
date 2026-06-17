import numpy as np

# ──────────────────────────────────────────────
# Problema 2 — Item a)
# f(x, y) = (x + 2y - 7)² + (2x + y - 5)²
# ──────────────────────────────────────────────

def f(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def grad_f(x):
    df_dx = 2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5)
    df_dy = 4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5)
    return np.array([df_dx, df_dy])

def hess_f(x):
    # Hessiana constante (função quadrática)
    return np.array([[10.0, 8.0],
                     [ 8.0, 10.0]])

# ──────────────────────────────────────────────
# Avaliação no ponto inicial x0 = (0, 0)
# ──────────────────────────────────────────────
x0 = np.array([0.0, 0.0])

g = grad_f(x0)
H = hess_f(x0)
autovalores = np.linalg.eigvalsh(H)
kappa = autovalores.max() / autovalores.min()

print("=" * 55)
print("Problema 2 — Item a)")
print("=" * 55)
print(f"\nFunção: f(x,y) = (x + 2y - 7)² + (2x + y - 5)²")
print(f"\nPonto x0 = {x0}")
print(f"f(x0)    = {f(x0):.4f}")
print(f"\nGradiente em x0:")
print(f"  ∇f(x0)   = {g}")
print(f"  ‖∇f(x0)‖ = {np.linalg.norm(g):.4f}")
print(f"\nHessiana (constante):")
print(f"  H =\n{H}")
print(f"\nAutovalores: λ_min = {autovalores[0]:.1f}, λ_max = {autovalores[1]:.1f}")
print(f"Número de condicionamento: κ(H) = {kappa:.1f}")
print(f"\nMínimo analítico: resolver ∇f = 0")
# Resolve H x = b onde b = [34, 38]
b = np.array([34.0, 38.0])
x_star = np.linalg.solve(H, b)
print(f"  x* = {x_star}  →  f(x*) = {f(x_star):.6e}")