from math import sqrt, pi, sin, cos, asin, acos
from time import time
from numba import njit


# import cProfile


def main():

    G = 6.67430e-11  # SGP in SI
    au = 1.496e11  # Astronomical unit in meters
    Mj = 1.898e27  # Julian mass in kilograms
    Me = 5.972e24  # Earth mass in kilograms

    M = 1047.35 * Mj  # one solar mass
    mu = M * G

    r1 = 1 * au
    r2 = 2 * au

    m1 = Me
    m2 = 1 * Me

    rorb = 100000  # 100 km initial and target orbit.
    h1 = 6700000  # radius of planet 1
    h2 = 6700000  # radius of planet 2

    search_low = -max(r1, r2) * 20
    search_high = max(r1, r2) * 20
    search_res = max(r1, r2) / 100

    unit = 86400  # time unit of plot, unit of seconds
    ttcutoff = 300 * unit  # unit of unit.chooping off data points taking too long

    dv_unit = 1000  # unit of m/s.
    dvcutoff = 12 * dv_unit

    vorb1 = sqrt(G * m1 / (rorb + h1))
    vorb2 = sqrt(G * m2 / (rorb + h2))

    vesc1 = sqrt(2 * G * m1 / (rorb + h1))
    vesc2 = sqrt(2 * G * m2 / (rorb + h1))

    w1 = sqrt(mu / r1 ** 3)
    w2 = sqrt(mu / r2 ** 3)  # assuming circular orbit.
    syn_period = 2 * pi / abs(w2 - w1)  # synodic period in seconds

    @njit(fastmath=True)
    def shooting_pblm(alpha):

        result = []

        y = search_low

        while y <= search_high:
            """ hyperbola parameters"""

            rm = (r1 + r2) / 2
            d = sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * cos(alpha * pi / 180)) / 2
            A = (r2 - r1) / 2  # semi-major axis
            B = sqrt(d ** 2 - A ** 2)  # semi-minor axis
            E = d / A  # eccentricity

            """ coordinates of center-of-attraction (F1)"""

            x0 = -1 * rm / E
            y0 = B * sqrt((x0 / A) ** 2 - 1)

            """ coordinate of free foci"""

            x = A * sqrt(1 + (y / B) ** 2)

            """ semi-major axis of the ellipse defined
            by free-foci and passing through r1,r2"""

            a = (rm + E * x) / 2

            """eccentricity of ellipse """

            e = sqrt((x0 - x) ** 2 + (y0 - y) ** 2) / (2 * a)

            """ eccentric anomaly of start and end points"""

            ea1 = acos((1 - r1 / a) * 1 / e)
            ea2 = acos((1 - r2 / a) * 1 / e)

            """ mean anomaly of start and end points"""

            ma1 = ea1 - e * sin(ea1)
            ma2 = ea2 - e * sin(ea2) + alpha // 360 * 2 * pi

            """period from mean anomaly different:"""

            periods = (ma2 - ma1) / (2 * pi)

            """ direction of unit-vector pointing from free foci to F1"""

            fx = (x0 - x) / sqrt((x0 - x) ** 2 + (y0 - y) ** 2)
            fy = (y0 - y) / sqrt((x0 - x) ** 2 + (y0 - y) ** 2)

            """ calculating initial true anomaly theta1"""
            if sin(alpha * pi / 180) > 0:
                sintheta1 = ((x0 + d) * fy - y0 * fx) / r1
            else:
                sintheta1 = -1 * ((x0 + d) * fy - y0 * fx) / r1

            theta1 = asin(sintheta1)
            theta2 = theta1 + alpha

            """true anomaly"""

            oneperiod = sqrt(a ** 3 / mu) * 2 * pi

            """rejects long orbit"""
            if (periods * oneperiod) > ttcutoff:
                y += search_res
                continue

            """calculates the shooting angle measured from first to second"""

            sa = alpha * pi / 180 - periods * 2 * pi * sqrt(a ** 3 / mu) * w2

            """reduces sa to between (0,2pi)"""

            sa = sa % (2 * pi)

            """flight path angle"""

            cosphi1 = (1 + e * cos(theta1)) / sqrt(1 + e ** 2 + 2 * e * cos(theta1))
            cosphi2 = (1 + e * cos(theta2)) / sqrt(1 + e ** 2 + 2 * e * cos(theta2))
            phi1 = acos(cosphi1)
            phi2 = acos(cosphi2)

            """velocity"""

            v1 = sqrt(mu * (2 / r1 - 1 / a))
            v2 = sqrt(mu * (2 / r2 - 1 / a))

            """delta velocity (hyperbolic excess)"""

            dv1 = sqrt((cosphi1 * v1 - r1 * w1) ** 2 + (sin(phi1) * v1) ** 2)
            dv2 = sqrt((cosphi2 * v2 - r2 * w2) ** 2 + (sin(phi2) * v2) ** 2)

            """delta velocity (true)"""
            dv1 = sqrt(vesc1 ** 2 + dv1 ** 2) - vorb1
            dv2 = sqrt(vesc2 ** 2 + dv2 ** 2) - vorb2

            """rejects costly orbits"""

            if (dv1 + dv2) > dvcutoff:
                y += search_res
                continue
            else:
                result.append([sa, dv1, dv2, periods * oneperiod])

            y += search_res

        return result

    start_time = time()

    ini_alp = 0
    ang_res = 1 / 100

    @njit
    def firstelement(lists):
        return lists[0]

    @njit
    def traj_comp():
        trajectory = []
        alpha = ini_alp
        """if only want trajectory in the direction of r1 X r2 then (0,180)"""
        """else 360 for full coverage"""
        while alpha <= 180:
            if alpha % 180 == 0:
                alpha += ang_res
            trajectory.extend(shooting_pblm(alpha))
            alpha += ang_res
        trajectory.sort(key=firstelement)
        return trajectory

    """sorts the list according to shooting angle"""

    trajectory = traj_comp()

    end_time = time()

    print("Calculating took {} seconds".format(round(end_time - start_time, 2)))

    def draw(trajectory):
        import numpy as np

        """calculates size of grid"""

        xsize = int(syn_period // unit + 1)

        ysize = int(ttcutoff // unit + 1)

        """mapping orbits result to specific grid-point."""

        Z = np.zeros(shape=(xsize, ysize), dtype=float)
        cman = np.zeros(
            shape=(xsize, ysize), dtype=int
        )  # N used to calculate cumulative moving average

        for traj in trajectory:

            sa_rad = traj[0]
            dv1 = traj[1]
            dv2 = traj[2]
            time_s = traj[3]

            sa_t = sa_rad * syn_period / (2 * pi)

            """calculates the corresponding point"""

            x = int(sa_t // unit)
            y = int(time_s // unit)

            """cutoffs"""
            tdv = (dv1 + dv2) / dv_unit

            if cman[x, y] == 0:
                Z[x, y] = tdv
            else:
                """rolling average in the computation of dv required"""
                Z[x, y] += (tdv - Z[x, y]) / (cman[x, y] + 1 + 1)

            """increment the rolling count by 1"""
            cman[x, y] = cman[x, y] + 1

        from matplotlib import pyplot as plt
        from matplotlib.colors import BoundaryNorm
        from matplotlib.ticker import MaxNLocator

        fig = plt.figure()
        """enumerate to find peak and bottom of chart"""
        zmax = Z.max()
        zmin = zmax
        for indexz, z in np.ndenumerate(Z):
            if z != 0:
                if z < zmin:
                    zmin = z

        """initialize plot with 1 subplot"""
        ax1 = fig.add_subplot(111)
        ax1.title.set_text("dV Variation During 1 Synodic Period")
        ax1.set_ylim(top=ysize)

        """black magic happens here"""
        levels = MaxNLocator(nbins=8).tick_values(zmin, zmax)
        cmap = plt.get_cmap("hot")
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        sigmadv = ax1.pcolormesh(np.swapaxes(Z, 0, 1), norm=norm, cmap=cmap)
        plt.colorbar(sigmadv)
        ax1.set_ylabel("Travel Time / x {} days".format(round(unit / 86400), 1))

        """plot isochorno line"""

        for i in range(0, xsize, 10):
            ax1.plot([0, i], [i, 0], linewidth=0.25, color="white", linestyle="dashed")

        plt.show()

    save = True
    """save the available trajectories as .txt"""
    if save:
        """sort trajectory according to sum dv"""

        def sum12(lists):
            return lists[1] + lists[2]

        trajectory.sort(key=sum12)
        from tabulate import tabulate

        f = open("trajs.txt", "w")
        f.write(
            tabulate(
                trajectory,
                tablefmt="pipe",
                headers=[
                    "shooting angle rad",
                    "ejecion dv m/s",
                    "injecting dv m/s",
                    "travel time s",
                ],
            )
        )
        f.close()

    draw(trajectory)


# cProfile.run('main()')
main()
