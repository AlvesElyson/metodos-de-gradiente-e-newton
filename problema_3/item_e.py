import numpy as np
import matplotlib.pyplot as plt

from problema_3.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo

# ──────────────────────────────────────────────
# Problema 3 — Item e)
# Dificuldades encontradas pelo método do gradiente
# na função de Beale.
#
# Hipóteses a investigar:
# 1) Regiões muito "planas" (gradiente pequeno) longe do ótimo
#    fazem o GD avançar lentamente, mesmo sem mal condicionamento
#    no sentido clássico.
# 2) A função tem vales estreitos e curvos — diferente das
#    quadráticas, a direção de descida "ótima" muda a cada passo.
# 3) Não-convexidade: existem regiões com Hessiana indefinida,
#    o que não afeta diretamente o GD (que sempre usa -gradiente),
#    mas pode gerar passos de Armijo muito pequenos perto de
#    platôs/inflexões.
# ──────────────────────────────────────────────

x0 = np.array([1.0, 1.0])
r = gradiente_descendente(f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4)

print("=" * 75)
print("Problema 3 — Item e)  Dificuldades do método do gradiente (Beale)")
print("=" * 75)
print(f"\nGD + Armijo a partir de x0=(1,1): {r['n_iter']} iterações até convergir.")
print(f"(Para comparação, no Problema 1 com κ=1000, Armijo precisou de 7304 iterações")
print(f" SEM exigir nada de não-convexidade — aqui a dificuldade é de outra natureza.)")

# ──────────────────────────────────────────────
# 1) Evolução do tamanho do passo (alpha) ao longo das iterações
#    -> revela platôs onde alpha precisa cair muito
# ──────────────────────────────────────────────
def gd_com_historico_alpha(f, grad, x0, tol=1e-6, max_iter=1000,
                           alpha0=1.0, sigma=0.5, c=1e-4):
    x = x0.copy().astype(float)
    hist_alpha = []
    hist_gnorm = [np.linalg.norm(grad(x))]
    hist_x = [x.copy()]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            break
        d = -g
        alpha = alpha0
        fx, slope = f(x), g @ d
        while f(x + alpha*d) > fx + c*alpha*slope:
            alpha *= sigma
            if alpha < 1e-16:
                break
        x = x + alpha * d
        hist_alpha.append(alpha)
        hist_gnorm.append(np.linalg.norm(grad(x)))
        hist_x.append(x.copy())
    return np.array(hist_alpha), np.array(hist_gnorm), np.array(hist_x)

alphas, gnorms, traj = gd_com_historico_alpha(f, grad_f, x0)

# ──────────────────────────────────────────────
# 2) Identificar "platôs" - trechos onde ‖∇f‖ varia pouco
#    por muitas iterações (estagnação)
# ──────────────────────────────────────────────
janela = 50
variacao = np.array([
    abs(gnorms[i+janela] - gnorms[i]) / (gnorms[i] + 1e-12)
    for i in range(0, len(gnorms)-janela, janela)
])
n_platos = np.sum(variacao < 0.05)
print(f"\nNúmero de janelas de {janela} iterações com <5% de redução no gradiente: "
      f"{n_platos} de {len(variacao)}")
if n_platos > 0:
    print("→ Indica presença de regiões quase-planas (platôs) na trajetória.")
else:
    print("→ Não há platôs longos nesta trajetória específica (x0=(1,1));")
    print("  a dificuldade observada é antes a OSCILAÇÃO do passo α (veja Figura 1),")
    print(f"  que varia entre {alphas.min():.4f} e {alphas.max():.4f} a cada iteração,")
    print("  pois a curvatura da função muda de direção ao longo do vale sinuoso.")

# ──────────────────────────────────────────────
# Figura 1 — alpha e ‖∇f‖ ao longo das iterações
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Problema 3 — Item e)  Dificuldades do GD: passo e estagnação", fontsize=12)

axes[0].plot(alphas, color="tab:blue", lw=1)
axes[0].set_title("Tamanho do passo α aceito por Armijo")
axes[0].set_xlabel("Iteração"); axes[0].set_ylabel("α")
axes[0].set_yscale("log")
axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].semilogy(gnorms, color="tab:red", lw=1.2)
axes[1].axhline(1e-6, color="black", ls=":", label="tol=1e-6")
axes[1].set_title("‖∇f(xₖ)‖ — note trechos quase planos (platôs)")
axes[1].set_xlabel("Iteração"); axes[1].set_ylabel("‖∇f(xₖ)‖  [log]")
axes[1].legend(fontsize=9)
axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("prob3_e_dificuldades_passo.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Figura 2 — Trajetória completa sobre a superfície de níveis,
#            destacando zonas de baixo gradiente (relevo "plano")
# ──────────────────────────────────────────────
X  = np.linspace(-1, 4.5, 600)
Y  = np.linspace(-1, 4.5, 600)
Xg, Yg = np.meshgrid(X, Y)
Zg = np.zeros_like(Xg)
Gnorm_grid = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        p = [Xg[i, j], Yg[i, j]]
        Zg[i, j] = f(p)
        Gnorm_grid[i, j] = np.linalg.norm(grad_f(p))

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Problema 3 — Item e)  Relevo da função de Beale", fontsize=12)

levels = np.logspace(-2, 3.5, 35)
cs = axes2[0].contour(Xg, Yg, Zg, levels=levels, cmap="coolwarm", alpha=0.7)
axes2[0].plot(traj[:, 0], traj[:, 1], "-", color="black", lw=1.2)
axes2[0].plot(traj[0, 0], traj[0, 1], "ks", ms=8, label="x₀=(1,1)")
axes2[0].plot(3, 0.5, "r*", ms=14, label="x*=(3,0.5)")
axes2[0].set_title("Curvas de nível de f(x,y)")
axes2[0].set_xlabel("x"); axes2[0].set_ylabel("y")
axes2[0].legend(fontsize=9); axes2[0].grid(True, ls="--", alpha=0.3)

im = axes2[1].pcolormesh(Xg, Yg, np.log10(Gnorm_grid + 1e-10), cmap="viridis", shading="auto")
axes2[1].plot(traj[:, 0], traj[:, 1], "-", color="white", lw=1.3)
axes2[1].plot(traj[0, 0], traj[0, 1], "ks", ms=8, label="x₀=(1,1)")
axes2[1].plot(3, 0.5, "r*", ms=14, label="x*=(3,0.5)")
plt.colorbar(im, ax=axes2[1], label="log10 ‖∇f‖")
axes2[1].set_title("Mapa de log‖∇f(x,y)‖\n(regiões escuras = quase-planas)")
axes2[1].set_xlabel("x"); axes2[1].set_ylabel("y")
axes2[1].legend(fontsize=9, loc="lower left")

plt.tight_layout()
plt.savefig("prob3_e_relevo.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# Discussão textual
# ──────────────────────────────────────────────
print("\n" + "=" * 75)
print("DISCUSSÃO — Dificuldades do método do gradiente na função de Beale")
print("=" * 75)
print("""
1) Função não-quadrática e "vale curvo":
   Diferente do Problema 1/2 (quadráticas), aqui o GD não segue um vale
   reto. A cada iteração a direção -∇f muda de orientação, fazendo a
   trajetória curvar-se ao longo de um vale estreito e sinuoso —
   isso aumenta o número de iterações mesmo sem mal condicionamento
   constante.

2) Passo instável (oscilação de α):
   Mesmo usando Armijo, o passo aceito oscila bastante ao longo da
   trajetória (entre ~0.03 e 0.5, ver Figura 1) em vez de estabilizar.
   Isso ocorre porque a curvatura local da função muda de direção a
   cada iteração — diferente das quadráticas do Problema 1/2, onde o
   passo converge para um valor estável.

3) Múltiplos pontos críticos / não-convexidade:
   A função possui pontos de sela e regiões com Hessiana indefinida
   (ex: x=(1,1) e x=(0,1), ver item a). O mapa de log‖∇f‖ (Figura 2)
   mostra ainda regiões mais distantes do ótimo onde o gradiente é
   pequeno sem estar perto do mínimo — um risco geral em funções
   não-convexas, mesmo que a trajetória específica analisada aqui
   não tenha cruzado essas regiões.

4) Sensibilidade ao ponto inicial:
   Com x0=(1,4) o GD não converge em 1000 iterações — ilustrando que,
   em funções não-convexas, a robustez do método depende fortemente
   de onde se inicia a busca.

5) Comparação com Newton:
   Newton resolve problemas quadráticos em 1 passo, mas na função de
   Beale ele pode CONVERGIR PARA O PONTO ERRADO quando a Hessiana é
   indefinida (caso x0=(1,1) e x0=(1,4), ambos terminam em (0,1)).
   O GD, embora mais lento, é mais "seguro" pois sempre segue uma
   direção de descida.
""")