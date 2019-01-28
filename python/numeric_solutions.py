# Por: Adeildo Soares Ramos Jr
# Criação: 17/12/2018
# Versão: 30/12/2003

import numpy as np


def newmark_linear(M, C, K, fext, uo, vo, tf, dt, gama, beta):
    """
    Função para integração de sistemas de equações não lineares
    de segunda ordem através do método implícito de Newmark (formulação em
    deslocamentos)

    Parâmetros:
        M {array} - Matriz de massa
        C {array} - Matriz de amortecimento
        K {array} - Matriz de rigidez
        uo {array} - Vetor dos deslocamentos iniciais.
        vo {array} - Vetor das velocidades iniciais.
        tf {array} - Tempo de análise.
        dt {array} - Incremento do tempo.
        fext {func} - Função que modela a força externa.
        tf {float} - Tempo total da análise.
        dt {float} - Incremento do tempo.
        gama {float} - Coeficiente do algorítimo de Newmark.
        beta {float} - Coeficiente do algorítimo de Newmark.
    
    Retornos:
        t {array} - Vetor dos tempos da análise.
        u {array} - Matriz dos deslocamentos da análise.
        v {array} - Matriz das velocidades da análise.
        a {array} - Matriz das acelerações da análise.

    Os parâmetros da família de algoritmos Newmark definem alguns métodos
    conhecidos,  a depender dos valores dos parâmetros alpha e beta
    escolhidos.

    Método:				                  Tipo:		  Beta:	  Gama:
    Aceleração média (regra trapezoidal)  implícito	  1/4	  1/2
    Aceleração linear                     implícito   1/6     1/2
    Fox-Goodwin                           implícito   1/12    1/2
    """

    # Integração no tempo
    t = np.arange(start=0, stop=tf, step=dt)
    a = np.zeros((len(vo), len(t)))
    v = np.zeros((len(vo), len(t)))
    u = np.zeros((len(vo), len(t)))

    # Parâmetros do algoritmo (regra trapezoidal)
    a0 = 1 / (dt ** 2 * beta)
    a1 = gama / (dt * beta)
    a2 = 1 / (dt * beta)
    a3 = 1 / (2 * beta) - 1
    a4 = gama / beta - 1
    a5 = dt * (gama / (2 * beta) - 1)
    a6 = (1 - gama) * dt
    a7 = gama * dt
    # Inicialização das variáveis
    # Calcula o vetor de forças externas (f)
    f = fext(0)
    #  Aceleração inicial
    ao = np.linalg.inv(M) @ (f - C @ vo - K @ uo)
    #  Atribui os valores iniciais às variáveis
    a = np.empty((len(uo), len(t)))
    a[:, 0] = ao
    v = np.empty((len(uo), len(t)))
    a[:, 0] = vo
    u = np.empty((len(uo), len(t)))
    a[:, 0] = uo
    # Matriz de rigidez efetiva
    Kb = K + a1 * C + a0 * M
    # Loop para o cálculo de u, v e a para cada instante de tempo
    for i in range(1, len(t)):
        # Calcula o vetor de forças externas (f)
        f = fext(t[i])
        fk1 = (
            f
            + M @ (a0 * u[:, i - 1] + a2 * v[:, i - 1] + a3 * a[:, i - 1])
            + C @ (a1 * u[:, i - 1] + a4 * v[:, i - 1] + a5 * a[:, i - 1])
        )
        # Resolve o sistema e calcula o deslocamento em t+1
        uk1 = np.linalg.inv(Kb) @ fk1
        # Atualizações das acelerações, velocidades e deslocamentos
        u[:, i] = uk1
        a[:, i] = a0 * (uk1 - u[:, i - 1]) - a2 * v[:, i - 1] - a3 * a[:, i - 1]
        v[:, i] = v[:, i - 1] + a6 * a[:, i - 1] + a7 * a[:, i]
    return t, u, v, a


def diferencacentral(M, C, K, fext, uo, vo, tf, dt):
    """
    Função para integração de sistemas de equações não lineares
    de segunda ordem através do método implícito de Newmark

    Parâmetros:
        M {array} - Matriz de massa
        C {array} - Matriz de amortecimento
        K {array} - Matriz de rigidez
        uo {array} - Vetor dos deslocamentos iniciais.
        vo {array} - Vetor das velocidades iniciais.
        tf {array} - Tempo de análise.
        dt {array} - Incremento do tempo.
        fext {func} - Função que modela a força externa.
        tf {float} - Tempo total da análise.
        dt {float} - Incremento do tempo.

    Returnos:
        t {array} - Vetor dos tempos da análise.
        u {array} - Matriz dos deslocamentos da análise.
        v {array} - Matriz das velocidades da análise.
        a {array} - Matriz das acelerações da análise.
    """

    # Integração no tempo
    t = np.arange(start=0, stop=tf, step=dt)
    a = np.zeros((len(vo), len(t)))
    v = np.zeros((len(vo), len(t)))
    u = np.zeros((len(vo), len(t)))
    #  Parâmetros (Ver Busby pg 523)
    a0 = 1 / (dt ** 2)
    a1 = 1 / (2 * dt)
    a2 = 2 * a0
    a3 = 1 / a2
    # Inicialização das variáveis
    # Calcula o vetor de forças externas (f)
    f = fext(0)
    #  Aceleração inicial
    ao = np.linalg.inv(M) @ (f - C @ vo - K @ uo)
    #  Atribui os valores iniciais às variáveis
    a = np.empty((len(uo), len(t)))
    a[:, 0] = ao
    v = np.empty((len(uo), len(t)))
    v[:, 0] = vo
    u = np.empty((len(uo), len(t)))
    u[:, 0] = uo
    u_1 = uo - dt * vo + a3 * ao
    # Matriz de rigidez efetiva
    Kb = a1 * C + a0 * M
    # Loop para o cálculo de u, v e a para cada instante de tempo
    for i in range(1, len(t)):
        # Calcula o vetor de forças externas (f)
        f = fext(t[i])
        fk1 = f - (a0 * M - a1 * C) @ u_1 - (K - a2 * M) @ u[:, i - 1]
        # Resolve o sistema e calcula o deslocamento em t+1
        uk1 = np.linalg.solve(Kb, fk1)
        # Atualizações das acelerações, velocidades e deslocamentos
        u[:, i] = uk1
        v[:, i] = a1 * (u[:, i] - u_1)
        a[:, i] = a0 * (u[:, i] - 2 * u[:, i - 1] + u_1)
        u_1 = u[:, i - 1]
    return t, u, v, a
