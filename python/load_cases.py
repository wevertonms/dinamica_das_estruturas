import numpy as np
import scipy.linalg as LA


class Harmonic(object):
    def __init__(self, fo, omega):
        """
        Arguments:
            fo {float} -- Amplitude da carga.
            omega {float} -- Frequência angular da carga.
        """
        self.fo = fo
        self.omega = omega

    def load(self, t):
        """Função que modela uma carga harmônica.

        Arguments:
            t {float} -- Instante de tempo.

        Returns:
            float -- Valor da carga.
        """
        return self.fo * np.cos(self.omega * t)

    def response(self, omega_i, ni_i, up_i, vp_i, Fpi, t):
        """Função que fornece a resposta dinâmica de um extrutura a uma carga
        harmônica nas coordenadas principais.
        """
        omegad = np.sqrt(omega_i ** 2 - ni_i ** 2)
        Qi = (
            (omega_i ** 2 - self.omega ** 2)
            * Fpi
            / ((omega_i ** 2 - self.omega ** 2) ** 2 + 4 * ni_i ** 2 * self.omega ** 2)
        )
        Ri = (
            (2 * ni_i * self.omega)
            * Fpi
            / ((omega_i ** 2 - self.omega ** 2) ** 2 + 4 * ni_i ** 2 * self.omega ** 2)
        )
        C1 = up_i - Qi
        C2 = (vp_i - (Ri * self.omega) + (ni_i * C1)) / omegad
        Dp = (
            np.exp(-ni_i * t) * (C1 * np.cos(omegad * t) + C2 * np.sin(omegad * t))
            + Qi * np.cos(self.omega * t)
            + Ri * np.sin(self.omega * t)
        )
        return Dp

    def modal_superposition(self, M, C, K, u0, v0, fo, sim_time, num_steps):
        """Função computa a repsosta dinâmica de um sistema por superposição modal.

        Arguments:
            M {array} -- Matriz de massa global dos elementos.
            C {array} --  Matriz de amortecimento global dos elementos.
            K {array} -- Matriz de rigidez global dos elementos.
            u0 {array} -- Vetor dos deslocamentos iniciais.
            v0 {array} -- Vetor das velocidades iniciais.
            fo {array} -- Vetor das forças externas.
            sim_time {float} -- Tempo total de simulação.
            num_steps {int} -- Número de incrementos de tempo.
            num_modes {int} -- Números de modos que participarão da solução.

        Returns:
            [list] -- Vetor com os tempos de simulação.
            [array] -- Vetor com os deslocamento.
        """
        omega2, phi = LA.eigh(K, M)
        #  Freqüências naturais
        omega_i = np.sqrt(omega2)
        #  Mudança de base
        #  Matriz de massa
        Mp = phi.T @ M @ phi
        #  Matriz de amortecimento
        Cp = phi.T @ C @ phi
        #  Matriz de rigidez
        #  Kp_i = phi.T @ K @ phi
        # Vetor de forças externas
        Fp_i = phi.T @ fo
        #  Deslocamentos iniciais
        up = phi.T @ M @ u0
        #  Velocidades iniciais
        vp = phi.T @ M @ v0
        # ni_i = zeta*omega_i
        ni_i = Cp.diagonal() / (2 * Mp.diagonal())
        # Vetor com os tempos de simulação.
        t = np.arange(start=0, stop=sim_time, step=(sim_time / num_steps))
        # Cálculo do deslocamento (na base provisória)
        step = np.empty((M.shape[0], t.size))
        for i in range(M.shape[0]):
            step[i, :] = self.response(omega_i[i], ni_i[i], up[i], vp[i], Fp_i[i], t)
        # Mudança de base (volta à base original)
        # u = phi*dp
        u = phi @ step
        return t, u


class Impulsive(object):
    def __init__(self, fo, td):
        """
        Arguments:
            fo {float} -- Amplitude da carga.
            td {float} -- Tempo de aplicação da carga.
        """
        self.fo = fo
        self.td = td

    def load(self, t):
        """Função que modela uma carga impulsiva.

        Arguments:
            t {float} -- Instante de tempo.

        Returns:
            float -- Valor da carga.
        """
        return self.fo if t <= self.td else 0 * self.fo

    def response(self, Kp_i, Fp_i, omega_i, t):
        """Função que fornece a resposta dinâmica de um extrutura a uma carga
        impulsiva nas coordenadas principais.

        Arguments:
            t {float} -- Instante de tempo.
            K {matrix} -- Matrix de rigidez da estrutura.
            omega {float} -- Fe

        Returns:
            float -- Valor da carga.
        """
        # Primeira parte t <= td
        t1 = t[t <= self.td]
        Dp1 = (Fp_i / Kp_i) * (1 - np.cos(omega_i * t1))
        # Segunda parte: t >> self.
        t2 = t[t > self.td]
        Dp2 = (Fp_i / Kp_i) * (
            (1 - np.cos(omega_i * self.td)) * np.cos(omega_i * (t2 - self.td))
            + np.sin(omega_i * self.td) * np.sin(omega_i * (t2 - self.td))
        )
        return [*Dp1, *Dp2]

    def modal_superposition(self, M, K, u0, fo, sim_time, num_steps):
        """Função computa a repsosta dinâmica de um sistema por superposição modal.

        Arguments:
            M {array} -- Matriz de massa global dos elementos.
            K {array} -- Matriz de rigidez global dos elementos.
            u0 {array} -- Vetor dos deslocamentos iniciais.
            fo {array} -- Vetor das forças externas.
            sim_time {float} -- Tempo total de simulação.
            num_steps {int} -- Número de incrementos de tempo.

        Returns:
            [list] -- Vetor com os tempos de simulação.
            [array] -- Vetor com os deslocamento.
        """
        omega2, phi = LA.eigh(K, M)
        #  Freqüências naturais
        omega_i = np.sqrt(omega2)
        #  Mudança de base
        #  Matriz de rigidez
        Kp_i = phi.T @ K @ phi
        # Vetor de forças externas
        Fp_i = phi.T @ fo
        #  Deslocamentos iniciais
        # up = phi.T @ M @ u0
        # Vetor com os tempos de simulação.
        t = np.arange(start=0, stop=sim_time, step=(sim_time / num_steps))
        # Cálculo do deslocamento (na base provisória)
        step = np.empty((M.shape[0], t.size))
        for i in range(M.shape[0]):
            step[i, :] = self.response(Kp_i[i, i], Fp_i[i], omega_i[i], t)
        # Mudança de base (volta à base original)
        u = phi @ step
        return t, u


class Ramp(object):
    def __init__(self, fo, tr):
        """
        Arguments:
            fo {float} -- Amplitude da carga.
            tr {float} -- Tempo de amplificação da carga.
        """
        self.fo = fo
        self.tr = tr

    def load(self, t):
        """Função que modela uma carga rampa.

        Arguments:
            t {float} -- Instante de tempo.

        Returns:
            float -- Valor da carga.
        """
        return (self.fo * t / self.tr) if t <= self.tr else self.fo

    def response(self, Kp_i, Fp_i, omega_i, t):
        """Função que fornece a resposta dinâmica de um extrutura a uma carga
        de rampa nas coordenadas principais.

        Arguments:
            Fp_i {float} -- Amplitude da da carga.
            Kp_i {matrix} -- Rigidez da estrutura.
            omega_i {float} -- Frequência da carga externa.
            t {float} -- Instante de tempo.

        Returns:
            float -- Valor da carga.
        """
        # Primeira parte t <= tr
        t1 = t[t <= self.tr]
        Dp1 = (Fp_i / Kp_i) * (
            (t1 / self.tr) - (1 / (omega_i * self.tr)) * np.sin(omega_i * t1)
        )
        # Segunda parte: t > tr
        t2 = t[t > self.tr]
        Dp2 = (Fp_i / Kp_i) * (
            1
            + (1 / (omega_i * self.tr))
            * (np.sin(omega_i * (t2 - self.tr)) - np.sin(omega_i * t2))
        )
        return [*Dp1, *Dp2]

    def modal_superposition(self, M, K, fo, sim_time, num_steps):
        """Função computa a repsosta dinâmica de um sistema por superposição modal.

        Arguments:
            M {array} -- Matriz de massa global dos elementos.
            K {array} -- Matriz de rigidez global dos elementos.
            u0 {array} -- Vetor dos deslocamentos iniciais.
            fo {array} -- Vetor das forças externas.
            sim_time {float} -- Tempo total de simulação.
            num_steps {int} -- Número de incrementos de tempo.

        Returns:
            [list] -- Vetor com os tempos de simulação.
            [array] -- Vetor com os deslocamento.
        """
        omega2, phi = LA.eigh(K, M)
        #  Freqüências naturais
        omega_i = np.sqrt(omega2)
        #  Mudança de base
        #  Matriz de rigidez
        Kp_i = phi.T @ K @ phi
        # Vetor de forças externas
        Fp_i = phi.T @ fo
        #  Deslocamentos iniciais
        # up = phi.T @ M @ u0
        # Vetor com os tempos de simulação.
        t = np.arange(start=0, stop=sim_time, step=(sim_time / num_steps))
        # Cálculo do deslocamento (na base provisória)
        step = np.empty((M.shape[0], t.size))
        for i in range(M.shape[0]):
            step[i, :] = self.response(Kp_i[i, i], Fp_i[i], omega_i[i], t)
        # Mudança de base (volta à base original)
        u = phi @ step
        return t, u
