import numpy as np
import math as m
from numba import jit, njit
import time



# householder's method
@njit
def householder(T,labda,labda_squared, ini, abs_div=1e-10):
    x0 = ini
    f0, df, d2f, d3f = deltaStar0(T,x0,labda,labda_squared)
    while abs(f0) > abs_div:
        x1 = x0 - f0 * (df ** 2 - f0 * d2f / 2) / (
            df * (df ** 2 - f0 * d2f) + d3f * f0 ** 2 / 6
        )
        x0 = x1
        f0, df, d2f, d3f = deltaStar0(T,x0,labda,labda_squared)
    return x0


# slave function to hypergeometric calculation
@njit
def series(m, i, n):
    ret = 1.0
    for n in range(0, i):
        ret *= m + n
    return ret


@njit
def factorial(n):
    fact = 1.0
    for i in range(1, n + 1):
        fact *= i
    return fact


# evaluate hypergeometric function to the nominally tenth place
@njit
def hypergeometric(a, b, c, z, n=10):
    hyp = 0.0
    for i in range(0, n):
        hyp += (
            series(a, i, n)
            * series(b, i, n)
            / (series(c, i, n) * factorial(i))
            * z ** i
        )

    return hyp


G = 6.67430e-11  # SGP in SI
au = 1.496e11  # Astronomical unit in meters
Mj = 1.898e27  # Julian mass in kilograms
Me = 5.972e24  # Earth mass in kilograms

Msol = 1047.35 * Mj  # one solar mass
mu = Msol * G

mu1 = 1 * Me * G  # SGP for planet 1
mu2 = 0.107 * Me * G  # SGP for planet 2

rp1 = 6400 * 1000  # mean radius of planet 1
rp2 = 3400 * 1000  # mean radius of planet 2

hlo = 0 * 1000  # height of low orbit

vlo1 = m.sqrt(mu1 / (rp1 + hlo))  # low orbit velocity of planets
vlo2 = m.sqrt(mu2 / (rp2 + hlo))

ve1 = vlo1 * m.sqrt(2)  # escape velocity of planets
ve2 = vlo2 * m.sqrt(2)

# wrapper for f(x) = T(x)-T* using M = 0
@njit
def deltaStar0(T,x,labda,labda_squared):
    y = m.sqrt(1 - labda_squared * (1 - x ** 2))

    # use y and x to calculate auxiliary angle psi
    if abs(x) < 1:
        # elliptical
        psi = m.asin((y - x * labda) * m.sqrt(1 - x ** 2))
    else:
        # hyperbolical
        psi = m.asinh((y - x * labda) * m.sqrt(x ** 2 - 1))

    # non-dimensional TOF T(x)
    if x == 1:
        Tx = 2 / 3 * (1 - labda ** 3)
    # uses hypergeometric series expansion to evaluate when close to 1
    elif abs(x - 1) < 0.1:
        eta = y - labda * x
        s1 = (1 - labda - x * eta) / 2
        q = 4 / 3 * hypergeometric(3, 1, 2.5, s1)
        Tx = (eta ** 3 * q + 4 * labda * eta) / 2
    else:
        Tx = (
            1
            / (1 - x ** 2)
            * ((psi) / m.sqrt(abs(1 - x ** 2)) - x + labda * y)
        )

    if x == 1:
        # parabola
        dT = 2 / 5 * (labda ** 5 - 1)
        d2T = 0
        d3T = 0
    elif x == 0 and labda_squared == 1:
        # edge case x = 0
        dT = -2
        d2T = 0
        d3T = 0

    else:
        # valid for all cases: single rev, mult rev, elliptic and hyperbolic
        # except labda = 1 x = 0  and x = 1 all labda
        # first derivative
        dT = (3 * Tx * x - 2 + 2 * labda ** 3 * x / y) / (1 - x ** 2)
        # second derivative
        d2T = (
            3 * Tx + 5 * x * dT + 2 * (1 - labda ** 2) * labda ** 3 / y ** 3
        ) / (1 - x ** 2)
        # third derivative
        d3T = (
            7 * x * d2T
            + 8 * dT
            - 6 * (1 - labda ** 2) * labda ** 5 * x / y ** 5
        ) / (1 - x ** 2)

    return Tx-T, dT, d2T, d3T

# main lamberts solver implemented using python
# algorithm from Pykep in C++
@njit
def lambert(r1_arr, r2_arr, mu, t):

    # validates all inputs:
    assert t > 0
    assert mu > 0

    r1 = np.linalg.norm(r1_arr)
    r2 = np.linalg.norm(r2_arr)

    # vector pointing from r1 to r2
    c_arr = np.subtract(r2_arr, r1_arr)
    c = np.linalg.norm(c_arr)

    # semi perimeter
    s = (r1 + r2 + c) / 2

    # unit vectors
    ir1 = r1_arr / r1
    ir2 = r2_arr / r2

    ih = np.cross(ir1, ir2)
    h = np.linalg.norm(ih)
    if h != 0:
        ih = ih / h

    labda_squared = 1 - c / s
    labda = m.sqrt(labda_squared)

    r11, r12, r13 = r1_arr
    r21, r22, r23 = r2_arr

    # direction:
    if (r11 * r22 - r12 * r21) < 0:
        labda = -labda
        it1 = np.cross(ir1, ih)
        it2 = np.cross(ir2, ih)
    else:
        it1 = np.cross(ih, ir1)
        it2 = np.cross(ih, ir2)

    # non-dimensional travel time T
    # T = m.sqrt(2*mu/s**3)*(t2-t1)

    T = m.sqrt(2 * mu / s ** 3) * t

    # black magic happens here.

    # calculations associated with T
    # with additional parameter of M (revs)
    # target flight time T*, alias for T.

    assert abs(labda) < 1
    assert T > 0  # !!!?


    T0 = m.acos(labda) + labda * m.sqrt(1 - labda_squared)
    T1 = 2 / 3 * (1 - labda ** 3)

    # compute x0 from Eq.30:
    if T >= T0:
        x0 = (T0 / T) ** (2 / 3) - 1
    elif T < T1:
        x0 = 5 / 2 * T1 * (T1 - T) / (T * (1 - labda ** 5)) + 1
    else:
        x0 = (T0 / T) ** np.log2(T1 / T0)



    # start householder iterations from x0 and find x,y
    # for the first revolution, so presumably M = 0
    x = householder(T,labda,labda_squared,x0)
    # x-y transform using equaiton 16-17
    y = m.sqrt(1 - labda_squared * (1 - x ** 2))

    gamma = m.sqrt(mu * s / 2)
    rho = (r1 - r2) / c
    sigma = m.sqrt(1 - rho ** 2)

    vr1 = gamma * ((labda * y - x) - rho * (labda * y + x)) / r1
    vr2 = -gamma * ((labda * y - x) + rho * (labda * y + x)) / r2
    vt1 = gamma * sigma * (y + labda * x) / r1
    vt2 = gamma * sigma * (y + labda * x) / r2

    v1_arr = vr1 * ir1 + vt1 * it1
    v2_arr = vr2 * ir2 + vt2 * it2

    return v1_arr,v2_arr


# return velocity vector of planet in a circular orbit
# velocity vector direction is counterclockwise (ir x ih)
# defaults onto X-Y plane
@njit
def circularOrbit(r_arr, ih=np.array([0, 0, -1])):
    R = np.linalg.norm(r_arr)
    # unit vector
    iR = r_arr / R
    iV = np.cross(iR, ih)

    vel = m.sqrt(mu / R)

    return iV * vel


# solve lamberts over the entire synodic period.
# and scan for solution.
# r1 is fixed at [r1,0,0]
# returns numpy array of departure and arrival true dV, lists of travel time and angle.


def scan(mu, a1, a2, tlow, thigh, dAng=1, dT=1 * 86400):
    ysize = thigh // dT
    xsize = 360 // dAng
    dv_dep = np.empty(shape=(xsize, ysize), dtype=object)
    dv_arr = np.empty(shape=(xsize, ysize), dtype=object)
    dv_dep[:] = np.NaN
    dv_arr[:] = np.NaN
    ang_ls = []

    p1 = m.sqrt(a1 ** 3 / mu) * 2 * m.pi
    p2 = m.sqrt(a2 ** 3 / mu) * 2 * m.pi
    syn_p = p1 * p2 / abs(p1 - p2)
    ang = 0
    while ang < 360:
        ang_ls.append(syn_p * ang / 360)
        ang += dAng

    t_ls = []
    t = tlow
    while t < thigh:
        ang = 0
        r1_arr = np.array([a1, 0, 0])
        while ang < 360:
            angRad = ang * m.pi / 180
            r2_arr = np.array([m.cos(angRad) * a2, m.sin(angRad) * a2, 0])
            v1_arr,v2_arr = lambert(r1_arr, r2_arr, mu, t)

            # define all relevent velocities
            vr1_arr = circularOrbit(r1_arr)
            vr2_arr = circularOrbit(r2_arr)

            # hyperbolic excess velocity
            dv1_arr = np.subtract(v1_arr, vr1_arr)
            dv2_arr = np.subtract(v2_arr, vr2_arr)

            # actual low orbit delta v:
            dv1 = m.sqrt(np.linalg.norm(dv1_arr) ** 2 + ve1 ** 2) - vlo1
            dv2 = m.sqrt(np.linalg.norm(dv2_arr) ** 2 + ve2 ** 2) - vlo2

            y = t // dT
            x = ang // dAng
            dv_dep[x, y] = dv1
            dv_arr[x, y] = dv2

            ang += dAng

        t_ls.append(t)
        t += dT
    return dv_dep, dv_arr, t_ls, ang_ls

starttime = time.time()
dvDep, dvArr, tls, als = scan(mu, 1.0 * au, 1.524 * au, 10 * 86400, 300 * 86400)
endtime = time.time()
print("calculation took {} seconds".format(endtime-starttime))

# code to fancy-print and visualize data.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

deltaVcutoff = 50  # cutoff for contour plotting

fdep_arr = dvDep
farr_arr = dvArr

# sum up
fsum_arr = fdep_arr + farr_arr
# convert unit into km/s
fsum_arr /= 1000
summin = np.nanmin(fsum_arr)
minloc = np.where(fsum_arr == summin)

level = np.linspace(summin, deltaVcutoff, 10)
fig, ax = plt.subplots()
sumcontour = ax.contour(np.swapaxes(fsum_arr, 0, 1), levels=level)
ax.clabel(sumcontour, inline=1, inline_spacing=0.1, fontsize=8, fmt="%1.1f")
thisx, thisy = minloc
ax.scatter(thisx, thisy, marker="x")
ax.annotate(str("{0:.3f}".format(summin)), minloc, c="navy")

scalex = max(als) / len(als)
scaley = max(tls) / len(tls)

scaley /= 86400  # converts travel time into days
scalex /= 86400
ticks = ticker.FuncFormatter(lambda x, pos: "{0:.1f}".format(x * scalex))
ax.xaxis.set_major_formatter(ticks)
ticks = ticker.FuncFormatter(lambda y, pos: "{0:.1f}".format(y * scaley))
ax.yaxis.set_major_formatter(ticks)
plt.show()
