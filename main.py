import mpmath as mp
import scipy.integrate as integrate
import numpy as np
import cmath
import math
from multiprocessing import Pool
import time


class Sig_Calculate:
    mev = 0.511
    d = 3.6
    df = 0.0
    z1 = 48.0
    z2 = 1.0
    a1 = 112.0
    a2 = 1.0

    mn = 1.835E3
    gv = 2.9899e-12
    aplha_e = 1/137.03

    # substitute
    ksi_b = 1.0
    plank = 1.0
    e = 1.0

    m = mn*a1*a2/(a1+a2)

    d = d/mev
    df = df/mev

    # global
    lambda_i = 0.0
    k_i = 0.0

    def __init__(self, a1, z1, d, df, beta):
        self.a1 = a1
        self.z1 = z1
        self.d = d/self.mev
        self.df = df/self.mev
        self.m = self.mn*a1*self.a2/(a1+self.a2)
        self.ksi_b = 6250.0 / 10.0**beta

    def Phi(self, E):
        return 1.0/60.0 * (E**2.0 - 1.0)**0.5 * (2.0*E**4.0 - 9.0*E**2.0 - 8.0) \
            + 1.0/4.0 * E * np.log(E + (E**2.0 - 1.0)**0.5)

    def Get_k_f(self, ef):
        return math.sqrt(2.0*self.m*ef)

    def Lambda_f(self, ef):
        return (self.z1+1)*self.z2*self.aplha_e*np.sqrt(self.m/2.0/ef)

    def Lambda_i(self, ei):
        return self.z1*self.z2*self.aplha_e*np.sqrt(self.m/2.0/ei)

    def Get_k_i(self, ei):
        return np.sqrt(2.0*self.m*ei)

    def Inner_func(self, x, eps_f):
        lambda_f = self.Lambda_f(eps_f)
        return abs(mp.hyp2f1(-self.lambda_i*1.0j, -lambda_f*1.0j, 1.0, x))**2.0 / (1.0 - x)**2.0

    def E_f(self, eps_f, eps_i):
        return eps_i - eps_f - self.d - self.df

    def Outer_func(self, eps_f, eps_i):
        e_f = self.E_f(eps_f, eps_i)
        k_f = self.Get_k_f(eps_f)
        result = self.Phi(e_f) / (np.exp(2.0*cmath.pi*float(self.Lambda_f(eps_f)) - 1.0)
                                  * k_f*(self.k_i - k_f)**4.0 * (self.k_i+k_f)**2.0)
        return result

    def X_0(self, eps_f):
        k_f = self.Get_k_f(eps_f)
        result = -4.0*self.k_i*k_f / (self.k_i - k_f)**2.0
        return result

    def ResultFunc(self, x, eps_f, args):
        eps_i = args
        result = self.Outer_func(eps_f, eps_i)*self.Inner_func(x, eps_f)
        return float(result.real)

    def sig(self, eps_i):
        self.k_i = self.Get_k_i(eps_i)
        self.lambda_i = self.Lambda_i(eps_i)
        top_limit = eps_i - self.d - self.df
        integral = integrate.dblquad(self.ResultFunc, 0.0, top_limit, self.X_0, lambda eps_f: 0.0,
                           args=[eps_i])
        mltiple = (4.0 * 2.0**0.5 / cmath.pi) * \
            ((self.gv**2.0 * self.aplha_e**4.0 * self.z1 * (self.z1 + 1.0) * self.z2**4.0 * self.m**(9.0/2.0))) \
            / (eps_i**1.5 * (1.0 - np.exp(-2.0 * cmath.pi * float(self.lambda_i)))) * self.ksi_b
        return mltiple * integral[0]

    def fun3(self, eps_i, tt):
        return math.exp(-eps_i/tt)*eps_i*self.sig(eps_i)

    def sigv(self, tt):
        tt = tt * 1e-10/1.16/self.mev
        res = integrate.quad(self.fun3, self.d+self.df+1.0, np.inf, args=tt)
        return math.sqrt(8/math.pi/self.m/tt/tt/tt)*res[0] * 0.19448 * 44.7e-12


def thread_work(input_array):
    tt = input_array[0]
    a1 = input_array[1]
    z1 = input_array[2]
    d = input_array[3]
    df = input_array[4]
    beta = input_array[5]
    calc = Sig_Calculate(a1, z1, d, df, beta)
    return calc.sigv(tt)


def build_full_range(a1, z1, d, df, beta, nuc_name1, nuc_name2):
    t_array = np.linspace(1.0e8, 1.0e10, 24)
    input_array = []
    for i in range(0, len(t_array)):
        input_array.append([t_array[i], a1, z1, d, df, beta])
    f = open('out-sig/{0}{1}-{2}{1}.txt'.format(nuc_name1,
                                                int(a1), nuc_name2), 'w')
    p = Pool(12)
    start_time = time.time()
    y_array = p.map(thread_work, input_array)
    p.close()
    p.join()
    for i in range(0, len(y_array)):
        print(str(t_array[i]) + ' ' + str(y_array[i]), file=f)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(y_array)
    return y_array


# if __name__ == "__main__":
#     result = build_full_range(78.0, 35.0, 3.5737, 0.0, 4.8, "br", "kr")
#     result = build_full_range(80.0, 35.0, 1.8703, 0.0, 4.6, "br", "kr")
#     result = build_full_range(84.0, 37.0, 2.6800, 0.0, 9.6, "rb", "sr")
#     result = build_full_range(106.0, 47.0, 2.9830, 0.0, 4.9, "ag", "cd")
#     result = build_full_range(106.0, 47.0, 2.4710, 0.0, 5.3, "ag-2-", "cd")
#     result = build_full_range(108.0, 47.0, 1.9210, 0.0, 4.8, "ag", "cd")
#     result = build_full_range(108.0, 47.0, 1.4870, 0.0, 5.5, "ag-2-", "cd")
#     result = build_full_range(112.0, 49.0, 2.5780, 0.0, 4.7, "in", "sn")
#     result = build_full_range(112.0, 49.0, 1.9610, 0.0, 5.3, "in-2-", "sn")
#     result = build_full_range(114.0, 49.0, 1.9846, 0.0, 4.8, "in", "sn")
#     result = build_full_range(114.0, 49.0, 1.4266, 0.0, 5.3, "in-2-", "sn")
#     result = build_full_range(120.0, 51.0, 2.6810, 0.0, 4.5, "sb", "te")
#     result = build_full_range(124.0, 53.0, 3.1570, 0.0, 9.3, "i", "xe")
#     result = build_full_range(124.0, 53.0, 2.5550, 0.0, 7.5, "i-2-", "xe")
#     result = build_full_range(126.0, 53.0, 2.1560, 0.0, 9.2, "i", "xe")
#     result = build_full_range(126.0, 53.0, 1.4900, 0.0, 7.4, "i-2-", "xe")
#     result = build_full_range(130.0, 55.0, 3.0190, 0.0, 5.1, "cs", "ba")
#     result = build_full_range(130.0, 55.0, 2.4830, 0.0, 6.4, "cs-2-", "ba")
#     result = build_full_range(132.0, 55.0, 1.4430, 0.0, 6.0, "cs", "ba")
#     result = build_full_range(136.0, 58.0, 2.8700, 0.0, 4.6, "la", "ce")
#     result = build_full_range(164.0, 67.0, 1.0292, 0.0, 4.6, "ho", "er")
#     result = build_full_range(164.0, 67.0, 0.9558, 0.0, 4.9, "ho-2-", "er")
if __name__ == "__main__":
    result = build_full_range(78.0, 34.0, 3.5737, 0.0, 4.8, "se", "br")
    result = build_full_range(80.0, 34.0, 1.8703, 0.0, 4.6, "se", "br")
    result = build_full_range(84.0, 36.0, 2.6800, 0.0, 9.6, "kr", "rb")
    result = build_full_range(106.0, 46.0, 2.9830, 0.0, 4.9, "pd", "ag")
    result = build_full_range(106.0, 46.0, 2.4710, 0.0, 5.3, "pd-2-", "ag")
    result = build_full_range(108.0, 46.0, 1.9210, 0.0, 4.8, "pd", "ag")
    result = build_full_range(108.0, 46.0, 1.4870, 0.0, 5.5, "pd-2-", "ag")
    result = build_full_range(112.0, 48.0, 2.5780, 0.0, 4.7, "cd", "in")
    result = build_full_range(112.0, 48.0, 1.9610, 0.0, 5.3, "cd-2-", "in")
    result = build_full_range(114.0, 48.0, 1.9846, 0.0, 4.8, "cd", "in")
    result = build_full_range(114.0, 48.0, 1.4266, 0.0, 5.3, "cd-2-", "in")
    result = build_full_range(120.0, 50.0, 2.6810, 0.0, 4.5, "sn", "sb")
    result = build_full_range(124.0, 52.0, 3.1570, 0.0, 9.3, "te", "i")
    result = build_full_range(124.0, 52.0, 2.5550, 0.0, 7.5, "te-2-", "i")
    result = build_full_range(126.0, 52.0, 2.1560, 0.0, 9.2, "te", "i")
    result = build_full_range(126.0, 52.0, 1.4900, 0.0, 7.4, "te-2-", "i")
    result = build_full_range(130.0, 54.0, 3.0190, 0.0, 5.1, "xe", "cs")
    result = build_full_range(130.0, 54.0, 2.4830, 0.0, 6.4, "xe-2-", "cs")
    result = build_full_range(132.0, 54.0, 1.4430, 0.0, 6.0, "xe", "cs")
    result = build_full_range(136.0, 57.0, 2.8700, 0.0, 4.6, "ba", "la")
    result = build_full_range(164.0, 66.0, 1.0292, 0.0, 4.6, "dy", "ho")
    result = build_full_range(164.0, 66.0, 0.9558, 0.0, 4.9, "dy-2-", "ho")