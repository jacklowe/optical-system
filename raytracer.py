import numpy as np

"""Ray tracer, to test optical ray and element configurations."""


class Ray(object):
    def __init__(self, p=np.array([0, 0, 0]), k=np.array([0, 0, 0])):
        self.p = p
        self.k = k
        self.m = [p]
        self.n = [k]

    def vertices(self):
        return self.m

    def p1(self):
        return self.m[len(self.m) - 1]    # Most recently appended point

    def k1(self):
        return self.n[len(self.n) - 1]  # Most recently appended direction

    def append(self, newp=np.array([0, 0, 0]), newk=np.array([0, 0, 0])):
        return self.n.append(newk), self.m.append(newp)


class SphericalRefraction:
    """An optical element with a spherical surface."""

    def __init__(self, z0, curvature, n1, n2):
        self.curvature = curvature
        self.z0 = z0
        self.n1 = n1
        self.n2 = n2

        if self.curvature == 0:
            self.dirVector = np.array([0, 0, z0])
        else:
            self.dirVector = np.array([0, 0, z0 + 1 / self.curvature])

    def intercept(self, Ray):
        khat = Ray.k1() / np.linalg.norm(Ray.k1())

        if self.curvature == 0:
            li = self.z0 - Ray.p1()[2] / khat[2]
            return Ray.p1() + li * khat
        else:
            radiusofcurvature = 1 / self.curvature
            r = Ray.p1() - self.dirVector
            modr = np.linalg.norm(r)
            rdotk = np.dot(r, khat)
            arg = rdotk * rdotk - (modr * modr - (
                radiusofcurvature * radiusofcurvature))
            if arg < 0:
                return None
            elif self.curvature < 0:
                li = -rdotk + np.sqrt(arg)
                return Ray.p1() + li * khat
            else:
                li = -rdotk - np.sqrt(arg)
                return Ray.p1() + li * khat

    def normal_surface(self, Ray):
        if self.curvature == 0:
            return np.array([0, 0, -1])
        else:
            normalvector = np.array(
                [self.intercept(Ray)[0], self.intercept(Ray)[1], self.intercept
                 (Ray)[2] - self.dirVector[2]])
            return normalvector / np.linalg.norm(normalvector)

    def refract(self, Ray):
        """Using Snell's law to calculate the new direction of a ray upon
        intersection with an optical element."""
        eta = self.n1 / self.n2
        nhat = self.normal_surface(Ray)
        dhat = Ray.k1() / np.linalg.norm(Ray.k1())

        if np.dot(nhat, dhat) > 0:
            nhat = -nhat
        nxd = np.cross(nhat, dhat)
        if np.linalg.norm(np.cross(nhat, dhat)) > 1. / eta:
            return None
        else:
            return eta * (-np.cross(nhat, nxd)) - nhat * np.sqrt(
                1 - eta * eta * np.dot(nxd, nxd))

    def propagate_ray(self, Ray):
        if self.intercept(Ray) is None or self.refract(Ray) is None:
            return "Ray does not pass through optical element!"
        else:
            newp = self.intercept(Ray)
            newk = self.refract(Ray)
            Ray.append(newp, newk)
        return Ray.p1(), Ray.k1()


class OutputPlane:
    """The end of the optical system"""

    def __init__(self, z0):
        self.curvature = 0
        self.z0 = z0

    def intercept(self, Ray):
        khat = Ray.k1() / np.linalg.norm(Ray.k1())
        li = (self.z0 - Ray.p1()[2]) / khat[2]
        return Ray.p1() + li * khat

    def propagate_ray(self, Ray):
        if self.intercept(Ray) is None:
            return "Ray does not pass through optical element!"
        else:
            newp = self.intercept(Ray)
            Ray.append(newp, Ray.k1())
        return Ray.p1(), Ray.k1()


class Bundle(object):
    """Produce a bundle of rays uniformly spread in x and y, travelling in z"""

    def __init__(self, n=0, rmax=0, m=0):
        self.n = n
        self.rmax = rmax
        self.m = m

    def rt_uniform(self):
        for i in range(0, self.n):
            if i == 0:
                r = 0
                t = 0
                yield r, t
            else:
                r = i * self.rmax / self.n
                p = 2 * np.pi / (self.m * i)
                for j in range(1, self.m * i + 1):
                    t = p * j
                    yield r, t

    def make_list_of_rays(self):
        bundle_list = []
        for r, t in self.rt_uniform():
            rays = Ray(np.array([r * np.cos(t), r * np.sin(t), 0]),
                       [0, 0, 1.0])
            bundle_list.append(rays)
        return bundle_list
