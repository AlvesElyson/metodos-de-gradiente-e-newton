import numpy as np

# Problema 1 — Item a) ──────────────────────────────────────────────
# f(x, y) = 1000x² + y²

def f(x):
    """Função objetivo."""
    return 1000 * x[0]**2 + x[1]**2

def grad_f(x):
    """Gradiente analítico de f."""
    return np.array([2000 * x[0], 2 * x[1]])

def hess_f(x):
    """Hessiana analítica de f (constante para funções quadráticas)."""
    return np.array([[2000, 0],
                     [0,    2]])

# Ponto de avaliação ──────────────────────────────────────────────

if __name__ == "__main__":
    x0 = np.array([1.0, 1.0])

    g  = grad_f(x0)
    H  = hess_f(x0)

    # Número de condicionamento
    autovalores = np.linalg.eigvalsh(H)
    kappa = autovalores.max() / autovalores.min()

    # Resultados ──────────────────────────────────────────────

    print("=" * 50)
    print("Problema 1 — Item a)")
    print("=" * 50)
    print(f"\nFunção:    f(x, y) = 1000x² + y²")
    print(f"\nPonto x0 = {x0}")
    print(f"\nf(x0)    = {f(x0):.4f}")
    print(f"\nGradiente em x0:")
    print(f"  ∇f(x0) = {g}")
    print(f"  ‖∇f(x0)‖ = {np.linalg.norm(g):.4f}")
    print(f"\nHessiana (constante):")
    print(f"  H =\n{H}")
    print(f"\nAutovalores de H: λ_min = {autovalores[0]:.1f}, λ_max = {autovalores[1]:.1f}")
    print(f"Número de condicionamento: κ(H) = {kappa:.1f}")
    print(f"\nConclusão: κ = {kappa:.0f} >> 1 → função mal condicionada.")
    print("O gradiente descendente converge lentamente; Newton converge em 1 passo.")