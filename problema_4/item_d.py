import numpy as np
import matplotlib.pyplot as plt

from problema_4.item_a import f, grad_f, hess_f

# ──────────────────────────────────────────────
# Problema 4 — Item d)
# Verificar experimentalmente se a direção de Newton
#   d = -H^{-1} g
# é sempre uma direção de descida, isto é, se
#   g^T d < 0
# ──────────────────────────────────────────────

def direcao_newton(x):
    """Retorna a direção de Newton e o produto g^T d em x, ou None se H for singular."""
    g = grad_f(x)
    H = hess_f(x)
    if abs(np.linalg.det(H)) < 1e-12:
        return None, None, g, H
    d = np.linalg.solve(H, -g)
    return d, g @ d, g, H

# ──────────────────────────────────────────────
# 1) Teste nos três pontos do item c)
# ──────────────────────────────────────────────
pontos_c = {
    "(0, 0)":   np.array([0.0, 0.0]),
    "(2, -2)":  np.array([2.0, -2.0]),
    "(-3, -3)": np.array([-3.0, -3.0]),
}

print("=" * 80)
print("Problema 4 — Item d)  A direção de Newton é sempre de descida?")
print("=" * 80)
print(f"\n{'Ponto':<12} {'g^T d':>12} {'Autovalores H':>22} {'É descida?'}")
print("-" * 80)

for nome, x0 in pontos_c.items():
    d, gd, g, H = direcao_newton(x0)
    autoval = np.linalg.eigvalsh(H)
    if d is None:
        print(f"{nome:<12} {'N/A (H singular)':>12} {str(np.round(autoval,3)):>22}   N/A")
    else:
        descida = "SIM" if gd < 0 else "NÃO"
        print(f"{nome:<12} {gd:>12.4f} {str(np.round(autoval,3)):>22}   {descida}")

# ──────────────────────────────────────────────
# 2) Teste sistemático: varrer uma grade de pontos e verificar
#    a proporção de pontos onde Newton é direção de descida
# ──────────────────────────────────────────────
X_test = np.linspace(-8, 6, 80)
Y_test = np.linspace(-8, 6, 80)

resultados_grid = []
for x in X_test:
    for y in Y_test:
        p = np.array([x, y])
        d, gd, g, H = direcao_newton(p)
        if d is not None and np.linalg.norm(g) > 1e-8:
            resultados_grid.append((x, y, gd))

resultados_grid = np.array(resultados_grid)
n_total    = len(resultados_grid)
n_descida  = np.sum(resultados_grid[:, 2] < 0)
n_subida   = np.sum(resultados_grid[:, 2] > 0)

print(f"\n\nVarredura em grade {len(X_test)}x{len(Y_test)} pontos "
      f"(excluindo H singular e ∇f≈0):")
print(f"  Total de pontos válidos: {n_total}")
print(f"  Direção de DESCIDA (g^T d < 0): {n_descida}  ({100*n_descida/n_total:.1f}%)")
print(f"  Direção de SUBIDA  (g^T d > 0): {n_subida}  ({100*n_subida/n_total:.1f}%)")
print(f"\n→ CONCLUSÃO: a direção de Newton NÃO é sempre de descida nesta função.")
print(f"  Isso ocorre porque a Hessiana é indefinida em regiões onde sin(x+y) > 0")
print(f"  (autovalor λ2 = -2sin(x+y) < 0 nesses casos).")

# ──────────────────────────────────────────────
# Figura 1 — Mapa de g^T d no plano (x,y)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Problema 4 — Item d)  Mapa de g^T·d (direção de Newton)", fontsize=12)

sc = ax.scatter(resultados_grid[:, 0], resultados_grid[:, 1],
                c=np.sign(resultados_grid[:, 2]),
                cmap="RdYlGn_r", s=8, vmin=-1, vmax=1)

for nome, x0 in pontos_c.items():
    ax.plot(x0[0], x0[1], "ks", ms=10, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(nome, x0, textcoords="offset points", xytext=(8, 8), fontsize=9)

ax.set_title("Verde = direção de descida (g^Td<0)  |  Vermelho = direção de subida (g^Td>0)",
             fontsize=9)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("prob4_d_mapa_descida.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Curva sin(x+y)=0 que separa as regiões
# ──────────────────────────────────────────────
X  = np.linspace(-8, 6, 500)
Y  = np.linspace(-8, 6, 500)
Xg, Yg = np.meshgrid(X, Y)
Zg = np.sin(Xg + Yg) + (Xg - Yg)**2 - 1.5*Xg + 2.5*Yg + 1
sin_term = np.sin(Xg + Yg)

fig2, ax2 = plt.subplots(figsize=(8, 7))
fig2.suptitle("Problema 4 — Item d)  Regiões de convexidade local", fontsize=12)
ax2.contour(Xg, Yg, Zg, levels=30, cmap="coolwarm", alpha=0.4)
cs = ax2.contourf(Xg, Yg, sin_term, levels=[-1, 0, 1],
                  colors=["lightgreen", "lightcoral"], alpha=0.4)
ax2.contour(Xg, Yg, sin_term, levels=[0], colors="black", linewidths=2)

for nome, x0 in pontos_c.items():
    ax2.plot(x0[0], x0[1], "ks", ms=10, markeredgecolor="white", markeredgewidth=1.5)
    ax2.annotate(nome, x0, textcoords="offset points", xytext=(8, 8), fontsize=9)

ax2.set_title("Verde: sin(x+y)<0 → H definida positiva  |  Vermelho: sin(x+y)>0 → H indefinida",
              fontsize=9)
ax2.set_xlabel("x"); ax2.set_ylabel("y")

plt.tight_layout()
plt.savefig("prob4_d_regioes_convexidade.png", dpi=150)
plt.close()