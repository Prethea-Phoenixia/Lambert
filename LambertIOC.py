from math import sqrt, pi, sin, cos, asin, acos
from time import time
#import cProfile


def main():

    G = 6.67430e-11  # SGP in SI
    au = 1.496e11  # Astronomical unit in meters
    Mj = 1.898e27  # Julian mass in kilograms

    M = 1047.35 * Mj  # one solar mass
    mu = M * G

    r1 = 1 * au
    r2 = 1.7 * au
    w1 = sqrt(mu / r1 ** 3)
    w2 = sqrt(mu / r2 ** 3)  # assuming circular orbit.
    syn_period = 2 * pi / abs(w2 - w1)  # synodic period in seconds

    def shooting_pblm(alpha):

        search_low = -max(r1, r2) * 4
        search_high = max(r1, r2) * 4
        search_res = max(r1, r2) / 100

        def solve_lambert():
            result = []

            def free_param(y):

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

                return theta1, periods, a, e

            y = search_low
            while y <= search_high:
                theta, period, sma, e = free_param(y)
                result.append([theta, period, sma, e])
                y += search_res

            return result

        shooting_result = []

        result = solve_lambert()


        for i in result:
            """true anomaly"""

            theta1 = i[0]
            theta2 = theta1 + alpha
            periods = i[1]
            sma = i[2]
            ecc = i[3]

            oneperiod = sqrt(sma ** 3 / mu) * 2 * pi

            """calculates the shooting angle measured from first to second"""

            sa = alpha * pi / 180 - periods * 2 * pi * sqrt(sma ** 3 / mu) * w2

            """reduces sa to between (0,2pi)"""

            sa = sa % (2 * pi)

            """flight path angle"""

            cosphi1 = (1 + ecc * cos(theta1)) / sqrt(
                1 + ecc ** 2 + 2 * ecc * cos(theta1)
            )
            cosphi2 = (1 + ecc * cos(theta2)) / sqrt(
                1 + ecc ** 2 + 2 * ecc * cos(theta2)
            )
            phi1 = acos(cosphi1)
            phi2 = acos(cosphi2)

            """velocity"""

            v1 = sqrt(mu * (2 / r1 - 1 / sma))
            v2 = sqrt(mu * (2 / r2 - 1 / sma))

            """delta velocity (hyperbolic excess)"""

            dv1 = sqrt((cosphi1 * v1 - r1 * w1) ** 2 + (sin(phi1) * v1) ** 2)
            dv2 = sqrt((cosphi2 * v2 - r2 * w2) ** 2 + (sin(phi2) * v2) ** 2)

            shooting_result.append([sa, dv1, dv2, periods * oneperiod])

        return shooting_result

    start_time = time()

    ini_alp = 0
    ang_res = 0.25

    """must not take the value of 180 deg or 0 deg"""

    trajectory = []
    alpha = ini_alp
    while alpha <= 360:
        if alpha % 180 == 0:
            alpha += ang_res
        for i in shooting_pblm(alpha):
            trajectory.append(i)
        alpha += ang_res

    """sorts the list according to shooting angle"""

    def firstelement(lists):
        return lists[0]

    trajectory.sort(key=firstelement)
    end_time = time()

    print("Calculating took {} seconds".format(round(end_time - start_time, 2)))

    def tabulate(trajectory):
        from matplotlib import pyplot as plt
        from matplotlib.colors import BoundaryNorm
        from matplotlib.ticker import MaxNLocator
        import numpy as np

        res = 86400  # time resolution of plot, unit of seconds
        ttcutoff = 300  # unit of res.chooping off data points taking too long to arrive

        ax = plt.subplot(
            111, ylabel="Travel Time / {} days".format(round(res / 86400), 1)
        )
        ax.set_title(
            "Travel time vs Departure during Synodic Cycle with dV plotted in Colors"
        )

        dv_unit = 1000  # unit of m/s.
        dvcutoff = 10
        xsize = int(syn_period // res + 1)

        y_max = 0
        for traj in trajectory:
            if y_max > traj[3]:
                pass
            else:
                y_max = traj[3]
        ysize = min(int(y_max // res), ttcutoff) + 1

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

            x = int(sa_t // res)
            y = int(time_s // res)

            """cutoffs"""

            if y > ttcutoff:
                continue
            tdv = (dv1 + dv2) / dv_unit
            if tdv > dvcutoff:
                continue

            if cman[x, y] == 0:
                Z[x, y] = tdv
            else:
                Z[x, y] = Z[x, y] + (tdv - Z[x, y]) / (cman[x, y] + 1 + 1)

            cman[x, y] = cman[x, y] + 1

        levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap("PiYG")
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = ax.pcolormesh(np.swapaxes(Z, 0, 1), norm=norm)
        plt.colorbar(im)
        plt.show()

    tabulate(trajectory)


# cProfile.run('main()')
main()
