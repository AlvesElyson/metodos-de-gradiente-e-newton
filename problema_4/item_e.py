import numpy as np

from problema_4.item_a import f, grad_f, hess_f
from problema_1.item_b import gradiente_descendente, armijo
from problema_1.item_c import newton

# Problema 4 — Item e) ──────────────────────────────────────────────
# Discussão consolidada dos resultados observados
# nos itens a) a d).

pontos_iniciais = {
    "(0, 0)":   np.array([0.0, 0.0]),
    "(2, -2)":  np.array([2.0, -2.0]),
    "(-3, -3)": np.array([-3.0, -3.0]),
}

resultados = {}
for nome, x0 in pontos_iniciais.items():
    resultados[nome] = {
        "GD + Armijo": gradiente_descendente(
            f, grad_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
        "Newton + Armijo": newton(
            f, grad_f, hess_f, x0, armijo, alpha0=1.0, sigma=0.5, c=1e-4),
    }

print("=" * 90)
print("Problema 4 — Item e)  Discussão dos resultados observados")
print("=" * 90)

print("""
RESUMO DOS RESULTADOS (itens a-d):
""")
print(f"{'x0':<10} {'GD: iter / f*':<22} {'Newton: iter / f* / status'}")
print("-" * 90)
for nome_x0, metodos in resultados.items():
    gd = metodos["GD + Armijo"]
    nt = metodos["Newton + Armijo"]
    gd_str = f"{gd['n_iter']} / {gd['f_opt']:.4f}"
    nt_str = f"{nt['n_iter']} / {nt['f_opt']:.4f} / {nt.get('status','-')}"
    print(f"{nome_x0:<10} {gd_str:<22} {nt_str}")

print("""
─────────────────────────────────────────────────────────────────────────────
DISCUSSÃO
─────────────────────────────────────────────────────────────────────────────

1) A Hessiana desta função NÃO é constante e tem autovalores:
       λ1 = 4               (sempre positivo)
       λ2 = -2·sin(x+y)      (pode ser positivo, negativo ou nulo)

   Logo, a convexidade local de f muda conforme a região do plano (x,y):
   H é definida positiva quando sin(x+y) < 0, e indefinida quando
   sin(x+y) > 0. Isso contrasta com os Problemas 1-2 (Hessiana constante,
   sempre def. positiva) e até com o Problema 3 (indefinição pontual,
   mas em região limitada).

2) Falhas do método de Newton observadas experimentalmente:

   a) Hessiana SINGULAR — ocorre exatamente quando sin(x+y) = 0,
      como em x0=(0,0) e x0=(2,-2) (ambos têm x+y=0). Nesses casos,
      o sistema linear H·d = -g não tem solução única e o método
      para imediatamente (0 iterações), sem produzir direção alguma.

   b) Hessiana INDEFINIDA mas não singular — ocorre em x0=(-3,-3),
      onde sin(x+y) = sin(-6) > 0. Aqui H ainda é invertível, mas a
      direção de Newton resultante satisfaz g^T d > 0 (direção de
      SUBIDA). Como a busca de Armijo exige decréscimo suficiente,
      ela reduz alpha repetidamente até um valor ínfimo sem nunca
      encontrar redução — o algoritmo fica "preso" no mesmo ponto
      por todas as 1000 iterações permitidas.

   c) Verificação sistemática (item d): numa varredura de 6400 pontos
      da grade, 6.3% apresentaram g^T d > 0 — ou seja, a direção de
      Newton NÃO é sempre de descida nesta função. Isso é uma
      propriedade teórica conhecida: a garantia g^T d < 0 só vale
      quando H é definida positiva.

3) Comportamento do Gradiente Descendente:

   Diferente de Newton, o GD (que usa d = -∇f) é SEMPRE uma direção de
   descida, pois g^T d = -‖g‖² ≤ 0 por construção, independente da
   Hessiana. Por isso o GD convergiu nos 3 pontos testados, embora para
   mínimos LOCAIS diferentes — função não-convexa, e o ponto alcançado
   depende de onde se inicia a busca:
     • x0=(0,0)   e x0=(2,-2)  →  convergem para o mesmo mínimo local
                                    (f* ≈ -1.913)
     • x0=(-3,-3) →  converge para um mínimo local diferente
                                    (f* ≈ -5.055, melhor que o outro!)

4) Lição geral:

   O método de Newton "puro" (sem modificações) é poderoso perto de
   mínimos locais com Hessiana definida positiva (convergência
   quadrática), mas é FRÁGIL em regiões não-convexas: pode falhar
   (H singular), estagnar (direção de subida + Armijo nunca aceita
   passo), ou convergir para pontos de sela. Na prática, versões
   robustas de Newton (Newton modificado, regularização da Hessiana,
   ex: somar λI até garantir def. positividade, ou métodos
   quase-Newton como BFGS) são usadas para mitigar esse problema.
   O Gradiente Descendente, embora mais lento, é estruturalmente mais
   robusto por sempre seguir uma direção de descida garantida.
""")