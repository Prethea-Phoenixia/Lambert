import numpy as np
import math as m
from numba import jit, njit


# halley's method to find root of f(x) = 0
def halley(func, ini, abs_div=1e-13):
    # initializing solver
    x0 = ini
    f0, df, d2f = func(x0)
    while abs(f0) > abs_div:
        # sequential iteration
        x1 = x0 - 2 * f0 * df / (2 * df ** 2 - f0 * d2f)
        x0 = x1
        f0, df, d2f = func(x0)
    return x0


# householder's method
def householder(func, ini, abs_div=1e-13):
    x0 = ini
    f0, df, d2f, d3f = func(x0)
    while abs(f0) > abs_div:
        x1 = x0 - f0 * (df ** 2 - f0 * d2f / 2) / (
            df * (df ** 2 - f0 * d2f) + d3f * f0 ** 2 / 6
        )
        x0 = x1
        f0, df, d2f, d3f = func(x0)
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

hlo = 100 * 1000  # height of low orbit

vlo1 = m.sqrt(mu1 / (rp1 + hlo))  # low orbit velocity of planets
vlo2 = m.sqrt(mu2 / (rp2 + hlo))

ve1 = vlo1 * m.sqrt(2)  # escape velocity of planets
ve2 = vlo2 * m.sqrt(2)


# main lamberts solver implemented using python
# algorithm from Pykep in C++
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

    def findxy(labda, T):
        # define calculations associated with T
        # with additional parameter of M (revs)
        # target flight time T*, alias for T.
        Tstar = T

        assert abs(labda) < 1
        assert T > 0  # !!!?
        Mmax = m.floor(T / m.pi)
        T00 = m.acos(labda) + labda * m.sqrt(1 - labda_squared)

        def fT(x, M):
            # calculates y
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
                T = 2 / 3 * (1 - labda ** 3)
            # uses hypergeometric series expansion to evaluate when close to 1
            elif abs(x - 1) < 0.1:
                eta = y - labda * x
                s1 = (1 - labda - x * eta) / 2
                q = 4 / 3 * hypergeometric(3, 1, 2.5, s1)
                T = (eta ** 3 * q + 4 * labda * eta) / 2
            else:
                T = (
                    1
                    / (1 - x ** 2)
                    * ((psi + M * m.pi) / m.sqrt(abs(1 - x ** 2)) - x + labda * y)
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
                dT = (3 * T * x - 2 + 2 * labda ** 3 * x / y) / (1 - x ** 2)
                # second derivative
                d2T = (
                    3 * T + 5 * x * dT + 2 * (1 - labda ** 2) * labda ** 3 / y ** 3
                ) / (1 - x ** 2)
                # third derivative
                d3T = (
                    7 * x * d2T
                    + 8 * dT
                    - 6 * (1 - labda ** 2) * labda ** 5 * x / y ** 5
                ) / (1 - x ** 2)

            return T, dT, d2T, d3T

        # x-y transform using equaiton 16-17
        def xytransform(x):
            return m.sqrt(1 - labda_squared * (1 - x ** 2))

        if T < T00 + Mmax * m.pi and Mmax > 0:
            # more than 1 rev possible
            # start halley iterations from x = 0, T = T0 and find Tmin(Mmax)
            # M = Mmax
            T0 = T00 + Mmax * m.pi

            # wrapper for calculating derivative of T
            def dT(x):
                t, dt, d2t, d3t = fT(x,Mmax)
                return dt, d2t, d3t

            # using halley's method to solve for root
            # then using root to find Tmin
            # dummy variables to preserve formatting.
            Tmin, dTmin, d2Tmin, d3Tmin = fT(halley(dT, 0),Mmax)

            if Tmin > T:
                Mmax = Mmax - 1
        else:
            T0 = T00

        T1 = 2 / 3 * (1 - labda ** 3)

        # compute x0 from Eq.30:
        if T >= T0:
            x0 = (T0 / T) ** (2 / 3) - 1
        elif T < T1:
            x0 = 5 / 2 * T1 * (T1 - T) / (T * (1 - labda ** 5)) + 1
        else:
            x0 = (T0 / T) ** m.log2(T1 / T0)

        # wrapper for f(x) = T(x)-T* using M = 0
        def deltaStar0(x):
            t, dt, d2t, d3t = fT(x, 0)
            return t - Tstar, dt, d2t, d3t

        # start householder iterations from x0 and find x,y
        # for the first revolution, so presumably M = 0
        x = householder(deltaStar0, x0)
        y = xytransform(x)

        xlist = [x]
        ylist = [y]

        while Mmax > 0:
            # x0r and x0l from Eq.31,M = Mmax
            M = Mmax
            # wrapper for f(x) = T(x)-T* using M = Mmax
            def deltaStar(x):
                t, dt, d2t, d3t = fT(x,M)
                return t - Tstar, dt, d2t, d3t

            x0l = (((M * m.pi + m.pi) / (8 * T)) ** (2 / 3) - 1) / (
                ((M * m.pi + m.pi) / (8 * T)) ** (2 / 3) + 1
            )
            x0r = (((8 * T) / (M * m.pi)) ** (2 / 3) - 1) / (
                ((8 * T) / (M * m.pi)) ** (2 / 3) + 1
            )

            xr = householder(deltaStar, x0l)
            yr = xytransform(xr)
            xlist.append(xr)
            ylist.append(yr)

            xl = householder(deltaStar, x0r)
            yl = xytransform(xl)
            xlist.append(xl)
            ylist.append(yl)

            Mmax -= 1

        return xlist, ylist

    xlist, ylist = findxy(labda, T)

    gamma = m.sqrt(mu * s / 2)
    rho = (r1 - r2) / c
    sigma = m.sqrt(1 - rho ** 2)

    vlist = []
    for x in xlist:
        y = ylist[xlist.index(x)]

        vr1 = gamma * ((labda * y - x) - rho * (labda * y + x)) / r1
        vr2 = -gamma * ((labda * y - x) + rho * (labda * y + x)) / r2
        vt1 = gamma * sigma * (y + labda * x) / r1
        vt2 = gamma * sigma * (y + labda * x) / r2

        v1_arr = vr1 * ir1 + vt1 * it1
        v2_arr = vr2 * ir2 + vt2 * it2
        vlist.append([v1_arr, v2_arr])

    return vlist


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


def scan(mu, a1, a2, tlow, thigh, dAng=1, dT=10 * 86400):
    ysize = thigh // dT
    xsize = 360 // dAng
    dv_dep = np.empty(shape=(xsize, ysize), dtype=object)
    dv_arr = np.empty(shape=(xsize, ysize), dtype=object)
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
            vlist = lambert(r1_arr, r2_arr, mu, t)

            dvdepls = []
            dvarrls = []

            for vel in vlist:
                # define all relevent velocities
                v1_arr = vel[0]
                vr1_arr = circularOrbit(r1_arr)
                v2_arr = vel[1]
                vr2_arr = circularOrbit(r2_arr)

                # hyperbolic excess velocity
                dv1_arr = np.subtract(v1_arr, vr1_arr)
                dv2_arr = np.subtract(v2_arr, vr2_arr)

                # actual low orbit delta v:
                dv1 = m.sqrt(np.linalg.norm(dv1_arr) ** 2 + ve1 ** 2) - vlo1
                dv2 = m.sqrt(np.linalg.norm(dv2_arr) ** 2 + ve2 ** 2) - vlo2

                dvdepls.append(dv1)
                dvarrls.append(dv2)

            y = t // dT
            x = ang // dAng
            dv_dep[x, y] = dvdepls
            dv_arr[x, y] = dvarrls

            ang += dAng

        t_ls.append(t)
        t += dT
    return dv_dep, dv_arr, t_ls, ang_ls


dvDep, dvArr, tls, als = scan(mu, 1.0 * au, 1.7 * au, 10 * 86400, 300 * 86400)

# filter array of orbits to first revolution.
def filterfirstrev(orbits_arr):
    x_lim, y_lim = orbits_arr.shape
    first_pass = np.empty_like(orbits_arr)
    for x in range(0, x_lim):
        for y in range(0, y_lim):
            if orbits_arr[x, y] is not None:
                first_pass[x, y] = orbits_arr[x, y][0]
            else:
                first_pass[x, y] = np.nan

    return first_pass


# code to fancy-print and visualize data.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

deltaVcutoff = 50  # cutoff for contour plotting

fdep_arr = filterfirstrev(dvDep)
farr_arr = filterfirstrev(dvArr)

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
