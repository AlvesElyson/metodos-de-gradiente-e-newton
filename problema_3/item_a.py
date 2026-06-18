import numpy as np

# ──────────────────────────────────────────────
# Problema 3 — Item a)
# f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
# (Função de Beale — não-convexa, mínimo global em (3, 0.5))
# ──────────────────────────────────────────────

def f(p):
    x, y = p[0], p[1]
    g1 = 1.5   - x + x*y
    g2 = 2.25  - x + x*y**2
    g3 = 2.625 - x + x*y**3
    return g1**2 + g2**2 + g3**2

def grad_f(p):
    x, y = p[0], p[1]
    g1 = 1.5   - x + x*y
    g2 = 2.25  - x + x*y**2
    g3 = 2.625 - x + x*y**3

    df_dx = 2*g1*(y - 1) + 2*g2*(y**2 - 1) + 2*g3*(y**3 - 1)
    df_dy = 2*g1*x + 4*g2*x*y + 6*g3*x*y**2

    return np.array([df_dx, df_dy])

def hess_f(p):
    x, y = p[0], p[1]
    g1 = 1.5   - x + x*y
    g2 = 2.25  - x + x*y**2
    g3 = 2.625 - x + x*y**3

    # d2f/dx2
    d2f_dx2 = 2*(y - 1)**2 + 2*(y**2 - 1)**2 + 2*(y**3 - 1)**2

    # d2f/dxdy
    d2f_dxdy = (2*g1 + 2*x*(y - 1)
               + 4*g2*y + 4*x*y*(y**2 - 1)
               + 6*g3*y**2 + 6*x*y**2*(y**3 - 1))

    # d2f/dy2
    d2f_dy2 = (2*x**2 + 4*g2*x + 8*x**2*y**2
              + 6*g3*x*2*y + 18*x**2*y**4)
    # versão simplificada/verificada via sympy:
    d2f_dy2 = x*(30.0*x*y**4 + 12.0*x*y**2 - 12.0*x*y - 2.0*x + 31.5*y + 9.0)
    d2f_dxdy = (12.0*x*y**5 + 8.0*x*y**3 - 12.0*x*y**2
               - 4.0*x*y - 4.0*x + 15.75*y**2 + 9.0*y + 3.0)

    return np.array([[d2f_dx2,  d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# ──────────────────────────────────────────────
# Validação numérica (diferenças finitas)
# ──────────────────────────────────────────────
def grad_numerico(func, p, h=1e-6):
    n = len(p)
    g = np.zeros(n)
    for i in range(n):
        p_mais = p.copy(); p_mais[i] += h
        p_menos = p.copy(); p_menos[i] -= h
        g[i] = (func(p_mais) - func(p_menos)) / (2*h)
    return g

def hess_numerico(func, p, h=1e-5):
    n = len(p)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pij = p.copy(); pij[i] += h; pij[j] += h
            pi  = p.copy(); pi[i]  += h
            pj  = p.copy(); pj[j]  += h
            p0  = p.copy()
            H[i, j] = (func(pij) - func(pi) - func(pj) + func(p0)) / (h*h)
    return H

# ──────────────────────────────────────────────
# Avaliação em x0 = (1, 1) e x0 = (3, 0.5)
# ──────────────────────────────────────────────
print("=" * 65)
print("Problema 3 — Item a)  Função de Beale")
print("=" * 65)

for x0 in [np.array([1.0, 1.0]), np.array([3.0, 0.5])]:
    print(f"\n--- Ponto x0 = {x0} ---")
    print(f"f(x0) = {f(x0):.6f}")

    g_anal = grad_f(x0)
    g_num  = grad_numerico(f, x0)
    print(f"∇f analítico = {g_anal}")
    print(f"∇f numérico  = {g_num}")
    print(f"Diferença max = {np.max(np.abs(g_anal - g_num)):.2e}")

    H_anal = hess_f(x0)
    H_num  = hess_numerico(f, x0)
    print(f"H analítica =\n{H_anal}")
    print(f"H numérica  =\n{H_num}")
    print(f"Diferença max = {np.max(np.abs(H_anal - H_num)):.2e}")

    if np.linalg.det(H_anal) != 0:
        autovalores = np.linalg.eigvalsh(H_anal)
        print(f"Autovalores de H: {autovalores}")
        if autovalores.min() > 0:
            kappa = autovalores.max() / autovalores.min()
            print(f"κ(H) = {kappa:.2f}  (Hessiana definida positiva)")
        else:
            print("Hessiana NÃO é definida positiva neste ponto (indefinida).")

print(f"\nMínimo global conhecido: x* = (3, 0.5),  f(x*) = 0")