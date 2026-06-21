import numpy as np

# Problema 4 — Item a)
# f(x, y) = sin(x+y) + (x-y)² - 1.5x + 2.5y + 1

def f(p):
    x, y = p[0], p[1]
    return np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

def grad_f(p):
    x, y = p[0], p[1]
    s = x + y
    df_dx = np.cos(s) + 2*(x - y) - 1.5
    df_dy = np.cos(s) - 2*(x - y) + 2.5
    return np.array([df_dx, df_dy])

def hess_f(p):
    x, y = p[0], p[1]
    s = x + y
    sin_s = np.sin(s)
    d2f_dx2  = 2 - sin_s
    d2f_dy2  = 2 - sin_s
    d2f_dxdy = -sin_s - 2
    return np.array([[d2f_dx2,  d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# Validação numérica (diferenças finitas)

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

# Avaliação nos pontos iniciais do item c) ──────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("Problema 4 — Item a)  f(x,y) = sin(x+y) + (x-y)² - 1.5x + 2.5y + 1")
    print("=" * 70)

    pontos = [np.array([0.0, 0.0]), np.array([2.0, -2.0]), np.array([-3.0, -3.0])]

    for x0 in pontos:
        print(f"\n--- Ponto x0 = {x0} ---")
        print(f"f(x0) = {f(x0):.6f}")

        g_anal, g_num = grad_f(x0), grad_numerico(f, x0)
        print(f"∇f analítico = {g_anal}")
        print(f"Diferença vs. numérico = {np.max(np.abs(g_anal - g_num)):.2e}")

        H_anal, H_num = hess_f(x0), hess_numerico(f, x0)
        print(f"H analítica =\n{H_anal}")
        print(f"Diferença vs. numérico = {np.max(np.abs(H_anal - H_num)):.2e}")

        autovalores = np.linalg.eigvalsh(H_anal)
        tipo = "definida positiva" if autovalores.min() > 0 else \
               "definida negativa" if autovalores.max() < 0 else "INDEFINIDA"
        print(f"Autovalores: {autovalores}  →  {tipo}")

    print("\nObs.: os autovalores de H têm forma fechada:")
    print("  λ1 = 4          (sempre positivo)")
    print("  λ2 = -2 sin(x+y) (varia conforme x+y)")
    print("\n→ A Hessiana é definida positiva quando sin(x+y) < 0,")
    print("  e INDEFINIDA quando sin(x+y) > 0 (pois λ2 < 0 nesse caso).")
    print("  Isso significa que a convexidade local da função MUDA")
    print("  dependendo da região do plano (x,y) — diferente do Problema 3,")
    print("  onde a indefinição era um caso isolado.")