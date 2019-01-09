"""Optical system module to generate optical systems and plot them."""


import matplotlib.pyplot as plt

import numpy as np

import raytracer as rt

# from scipy.optimize import fmin


class OpSystem:
    """
    c1 : curvature of leftmost spherical surface of the lens, default 0.01.

    c2 : curvature of rightmost spherical surface of lens, default 0.01.

    num_radial : number of rays in a bundle in the radial direction.

    num_angular : number of rays in a bundle in the angular direction.

    total number of rays in a given bundle is then nRadial * nAngular.

    n : refractive index of lens, assumed glass 1.5168.

    screen_position : position of screen i.e. end of optical system.
    """

    def __init__(self, lens_position=100, c1=0.02, c2=0.00, lens_thickness=10,
                 bundle_radius=10, num_radial=10, num_angular=10, n=1.5168,
                 screen_position=240):
        self.lens_position = lens_position
        self.c1 = c1
        self.c2 = c2
        self.lens_thickness = lens_thickness
        self.bundle_radius = bundle_radius
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.n = n
        self.screen_position = screen_position
        self.sphsurface = rt.SphericalRefraction(
            (self.lens_position - self.lens_thickness * 0.5),
            self.c1, 1, self.n)
        self.sphsurface2 = rt.SphericalRefraction(
            (self.lens_position + self.lens_thickness * 0.5),
            self.c2, self.n, 1)
        self.outputplane = rt.OutputPlane(self.screen_position)
        self.bundl = rt.Bundle(self.num_radial, self.bundle_radius,
                               self.num_angular)

    def propagate_bundle(self):
        """
        Method that propagates the bundle of rays through the optical
        system and returns in list form a tensor object T_ijk where the i
        index refers to the specific ray in the bundle, j refers to the
        ray-surface interaction index and k refers to the dimension index
        i.e. x, y or z.
        """
        boundaries = []
        for ray in self.bundl.make_list_of_rays():
            self.sphsurface.propagate_ray(ray)
            self.sphsurface2.propagate_ray(ray)
            self.outputplane.propagate_ray(ray)
            boundaries.append(ray.vertices())
        t_ijk = np.asarray(boundaries)
        return t_ijk

    def x_hat(self):
        """Return unit vector in the x-direction."""
        return np.array([1, 0, 0])

    def y_hat(self):
        """Return unit vector in y-direction."""
        return np.array([0, 1, 0])

    def z_hat(self):
        """Return unit vector in z-direction."""
        return np.array([0, 0, 1])

    def ray_diagram(self):
        """
        Plot a ray diagram of the bundle passing through the optical
        system.
        """
        for ray in self.propagate_bundle():
            ray_x = np.dot(ray, self.x_hat())
            ray_z = np.dot(ray, self.z_hat())
            plt.plot(ray_z, ray_x)
            plt.plot(ray_z, ray_x, 'bo')
        plt.show()

    def rms(self, c=None):
        """Return rms spread of ray bundle at output plane location."""
        t_i3x = self.propagate_bundle()[::, 3, 0]
        t_i3y = self.propagate_bundle()[::, 3, 1]
        d_i = np.sqrt(t_i3x * t_i3x + t_i3y * t_i3y)
        return np.std(d_i)

    def spot_diagram(self):
        """
        Plot a spot diagram of the bundle at the output plane. i.e. an x-y
        scatter plot. This shows the spherical symmetry of the ray bundle.
        """
        t_i3x = self.propagate_bundle()[::, 3, 0]
        t_i3y = self.propagate_bundle()[::, 3, 1]
        plt.plot(t_i3x, t_i3y, 'go')
        plt.show()

# def optimise_curvatures(self, c=None):
#     """
#     Optimise the curvature of the surfaces of the lens by minimising
#     the rms spread at the output plane.  Note that our initial guess x0
#     for curvature of front and back of lensis taken as the argument for
#     rms function 'c' which is the curvature initial guess
#     """
#     if c is not None:
#         self.c1 = c[0]
#         self.c2 = c[1]
#     a = fmin(self.rms, x0=[self.c1, self.c2])
#     print("Optimised curvature combination is", a, "from initial guess",
#           [self.c1, self.c2])
