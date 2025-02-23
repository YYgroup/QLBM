# !=============================================================================================
# ! Version 1.0
# ! Author: Boyuan Wang
# ! Solving the classic D2Q9 lattice with one relaxation time, LGCA collision model.
# ! Solving the quantum D2Q9 lattice with qiskit function and our inhouse code.
# ! Corresponding to incompressible viscous flows with constant density rho_0=1.
# ! Solution domain: a periodic square of side 2pi.
# ! 
# ! Update logs:
# !   [v1][250223]: Implement the overall simulation process;
# !=============================================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
import random
import os
import copy
import sys
import resource
import time
from scipy.optimize import root, fsolve

class square:

    def __init__(self, nx, ny, xmin=-1, xmax=1, ymin=-1, ymax=1):
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Lx = xmax - xmin
        self.Ly = ymax - ymin
        self.dx = (self.xmax - self.xmin) / nx
        self.dy = (self.ymax - self.ymin) / ny
        self.name = ''
        self.step = 1
        self.ex = []
        self.ey = []

    def __str__(self):
        return f'Square lattice with space grid: {self.nx}*{self.ny}'

    def print_vel(self, u, v, cs2):
        cs = np.sqrt(cs2)
        magnitude = np.sqrt(u ** 2 + v ** 2)
        max_magnitude = np.max(magnitude)
        print('Mach number is:', max_magnitude / np.sqrt(self.dx ** 2 + self.dy ** 2) / cs)

    def init_Gauss(self, vnx, vny, Gamma, sigma):
        nx = self.nx
        ny = self.ny
        lx = self.xmax - self.xmin
        ly = self.ymax - self.ymin
        X = np.linspace(-1 + lx / nx, 1, nx)
        Y = np.linspace(-1 + ly / ny, 1, ny)
        x, y = np.meshgrid(X, Y)
        pi = np.pi
        x0 = vnx * lx / nx - 1 + 1.5 * lx / nx
        y0 = vny * ly / ny - 1 + 1.5 * ly / ny
        xx = x - x0
        yy = y - y0
        r = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        utheta = Gamma / 2 / pi / r * (1 - np.exp(-r ** 2 / 2 / sigma ** 2))
        v = utheta * np.cos(theta)
        u = -utheta * np.sin(theta)
        return (X, Y, u, v)

    def init_TG(self, gamma):
        coef = gamma / 8
        ny = self.ny
        nx = self.nx
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))
        for i in range(nx):
            for j in range(ny):
                x = self.xmin + self.Lx / self.nx * (i + 1)
                y = self.ymin + self.Ly / self.ny * (j + 1)
                u[j][i] = coef * math.sin(x) * math.cos(y)
                v[j][i] = -coef * math.sin(y) * math.cos(x)
        return (u, v)

    def Gauss_vortex(self, vnx, vny, Gamma, sigma):
        ny = self.ny
        nx = self.nx
        pi = np.pi
        lx = self.xmax - self.xmin
        ly = self.ymax - self.ymin
        X = np.linspace(self.xmin + lx / nx, self.xmax, nx)
        Y = np.linspace(self.ymin + ly / ny, self.ymax, ny)
        x, y = np.meshgrid(X, Y)
        x0 = vnx * lx / nx + self.xmin + 1.5 * lx / nx
        y0 = vny * ly / ny + self.ymin + 1.5 * ly / ny
        xx = x - x0
        yy = y - y0
        r = np.sqrt(xx ** 2 + yy ** 2)
        return Gamma / 2 / pi / sigma ** 2 * np.exp(-r ** 2 / 2 / sigma ** 2)

    def cross_vortex(self, vnx, vny, amp, width):
        ny = self.ny
        nx = self.nx
        nnx = width / (self.xmax - self.xmin) * self.nx
        nny = width / (self.ymax - self.ymin) * self.ny
        vor = np.zeros((ny, nx))
        vor[:, round(vnx - 1 - nnx / 2):round(vnx - 1 + nnx / 2)] = amp
        vor[round(vny - 1 - nny / 2):round(vny - 1 + nny / 2), :] = amp
        return vor

    def vor_periodic(self, vor):
        fft_data = np.fft.fft2(vor)
        fft_data[0, 0] = 0
        return np.real(ifft2(fft_data))

    def vor_to_psi(self, vor):
        nx = self.nx
        ny = self.ny
        if vor.shape != (ny, nx):
            raise ValueError(f'Error: The size must be ({ny}, {nx}).')
        vor_filter = self.vor_periodic(vor)
        ny, nx = (self.ny, self.nx)
        kx = np.fft.fftfreq(nx, (self.xmax - self.xmin) / nx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, (self.ymax - self.ymin) / ny) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)
        k_squared = kx ** 2 + ky ** 2
        k_squared[0, 0] = 1
        psi_hat = fft2(vor_filter) / k_squared
        psi = np.real(ifft2(psi_hat))
        return psi

    def psi_to_vel(self, psi):
        nx = self.nx
        ny = self.ny
        dx = self.dx
        dy = self.dy
        if psi.shape != (ny, nx):
            raise ValueError(f'Error: The size must be ({ny}, {nx}).')
        u_ext = np.pad(psi, ((1, 1), (1, 1)), mode='wrap')
        v = -(u_ext[1:-1, 2:] - u_ext[1:-1, :-2]) / (2 * dx)
        u = (u_ext[2:, 1:-1] - u_ext[:-2, 1:-1]) / (2 * dy)
        return (u, v)

    def plot(self, vor, i=None, vmin=None, vmax=None, str='RdBu_r'):
        nx = self.nx
        ny = self.ny
        lx = self.xmax - self.xmin
        ly = self.ymax - self.ymin
        if i is None:
            i = 0
        if vmin is None:
            vmin = np.min(vor)
        if vmax is None:
            vmax = np.max(vor)
        X = np.linspace(self.xmin + lx / nx, self.xmax, nx)
        Y = np.linspace(self.ymin + ly / ny, self.ymax, ny)
        x, y = np.meshgrid(X, Y)
        plt.figure(i)
        plt.pcolormesh(x, y, vor, cmap=str, vmin=vmin, vmax=vmax)
        plt.gca().set_aspect(1)
        plt.colorbar(label='Vorticity', shrink=0.9)
        plt.show()

    def streamline_plot(self, u, v):
        nx = self.nx
        ny = self.ny
        speed = np.sqrt(u ** 2 + v ** 2)
        lx = self.xmax - self.xmin
        ly = self.ymax - self.ymin
        X = np.linspace(-1 + lx / nx, 1, nx)
        Y = np.linspace(-1 + ly / ny, 1, ny)
        x, y = np.meshgrid(X, Y)
        plt.figure()
        strm = plt.streamplot(X, Y, u, v, color=speed, cmap='viridis')
        plt.colorbar(strm.lines)
        plt.gca().set_aspect(1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Streamline Plot')
        plt.show()

    def gaussian_filter(self, data, cutoff):
        self.plot(data)
        fft_data = np.fft.fftshift(np.fft.fft2(data))
        x = np.linspace(-0.5, 0.5, self.ny)
        y = np.linspace(-0.5, 0.5, self.nx)
        X, Y = np.meshgrid(x, y)
        d = np.sqrt(X ** 2 + Y ** 2)
        filter_ = np.exp(-d ** 2 / (2 * cutoff ** 2))
        filtered_fft_data = fft_data * filter_
        return np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_data)))

    def poisson_solver_fft(self, w):
        ny, nx = (self.ny, self.nx)
        kx = np.fft.fftfreq(nx).reshape(1, nx)
        ky = np.fft.fftfreq(ny).reshape(ny, 1)
        kx, ky = np.meshgrid(kx, ky)
        k_squared = kx ** 2 + ky ** 2
        k_squared[0, 0] = 1
        psi_hat = fft2(w) / -k_squared
        psi_hat[0, 0] = 0
        psi = np.real(ifft2(psi_hat))
        return psi

    def vorcitity(self, u, v):
        nx = self.nx
        ny = self.ny
        dx = self.dx
        dy = self.dy
        if u.shape != (ny, nx) or v.shape != (ny, nx):
            raise ValueError(f'Error: The size of u and v must be ({ny}, {nx}). Got u: {u.shape}, v: {v.shape}')
        u_ext = np.pad(u, ((1, 1), (1, 1)), mode='wrap')
        v_ext = np.pad(v, ((1, 1), (1, 1)), mode='wrap')
        dvdx = (v_ext[:, 2:] - v_ext[:, :-2]) / (2 * dx)
        dudy = (u_ext[2:, :] - u_ext[:-2, :]) / (2 * dy)
        vorticity = dvdx[1:-1, :] - dudy[:, 1:-1]
        return vorticity

    def write_scalar(self, vor, time, basename):
        name = self.name
        np.save(basename + str(time) + name + '.npy', vor)

    def write_file(self, vor, rho, time, name1=None, name2=None):
        name = self.name
        if name1 is None:
            name1 = 'vor'
        if name2 is None:
            name2 = 'rho'
        if name1 + name + '.npz' in os.listdir('.'):
            data = np.load(name1 + name + '.npz')
            data = dict(data)
        else:
            data = {}
        data[f'vor_{time}'] = vor
        np.savez(name1 + name + '.npz', **data)
        if name2 + name + '.npz' in os.listdir('.'):
            data = np.load(name2 + name + '.npz')
            data = dict(data)
        else:
            data = {}
        data[f'rho_{time}'] = rho
        np.savez(name2 + name + '.npz', **data)

class D2Q9(square):

    def __init__(self, nx, ny, xmin=None, xmax=None, ymin=None, ymax=None):
        super().__init__(nx, ny, xmin=xmin if xmin is not None else -1, xmax=xmax if xmax is not None else 1, ymin=ymin if ymin is not None else -1, ymax=ymax if ymax is not None else 1)
        self.step = 1
        self.cs2 = 1 / 3
        self.Re = 10
        self.ex = [0, 1, 0, -1, 0, 1, -1, -1, 1]
        self.ey = [0, 0, 1, 0, -1, 1, 1, -1, -1]
        self.ea2 = [0, 1, 1, 1, 1, 2, 2, 2, 2]
        t0 = 4 / 9
        t1 = 1 / 9
        t2 = 1 / 36
        self.w = [t0, t1, t1, t1, t1, t2, t2, t2, t2]
        self.set_viscosity()

    def set_viscosity(self, Re=None):
        if Re is None:
            Re = self.Re
        self.nu = 1 / Re
        self.tau = 0.5 + self.nu / self.cs2
        self.omega = 1.0 / self.tau
        self.tau_p = self.tau
        self.lambda_trt = 1.0 / 4.0
        self.tau_m = self.lambda_trt / (self.tau_p - 0.5) + 0.5
        self.omega_p = 1.0 / self.tau_p
        self.omega_m = 1.0 / self.tau_m

    def get_equil(self, rho, u, v):
        feq = np.zeros((9, self.ny, self.nx))
        cdot = np.zeros((9, self.ny, self.nx))
        w = self.w
        ex = self.ex
        ey = self.ey
        cs2 = self.cs2
        for i in range(9):
            cdot[i, :, :] = ex[i] * u + ey[i] * v
            feq[i, :, :] = rho * w[i] * (1 + cdot[i, :, :] / cs2 + cdot[i, :, :] * cdot[i, :, :] / 2 / cs2 ** 2 - (u * u + v * v) / 2 / cs2)
            check_array(feq[i, :, :], 0, 1)
        return feq

    def macro_to_f_LGA(self, rho, u, v, epsilon):
        ex = self.ex
        ey = self.ey
        ea2 = self.ea2
        alpha1 = 1 / (1 - epsilon)
        gamma1 = (1 - 3 * epsilon) / 4 / epsilon ** 2 / (1 - epsilon)
        beta0 = -1 / epsilon
        beta1 = (3 * epsilon - 1) / 4 / epsilon ** 3
        d0 = rho * (1 - epsilon) ** 2
        d1 = rho / 2 * (1 - epsilon) * epsilon
        d2 = rho / 4 * epsilon ** 2
        f = np.zeros((9, self.ny, self.nx))
        cdot = np.zeros((9, self.ny, self.nx))
        for i in range(9):
            cdot[i, :, :] = ex[i] * u + ey[i] * v
        d = d0
        for i in range(0, 1):
            f[i, :, :] = d - d * (1 - d) * beta0 * cdot[i, :, :] - d * (1 - d) * (alpha1 + gamma1 * ea2[i]) * (u ** 2 + v ** 2) + d / 2 * (1 - d) * (1 - 2 * d) * beta0 ** 2 * cdot[i, :, :] ** 2
        d = d1
        for i in range(1, 5):
            f[i, :, :] = d - d * (1 - d) * beta0 * cdot[i, :, :] - d * (1 - d) * (alpha1 + gamma1 * ea2[i]) * (u ** 2 + v ** 2) + d / 2 * (1 - d) * (1 - 2 * d) * beta0 ** 2 * cdot[i, :, :] ** 2
        d = d2
        for i in range(5, 9):
            f[i, :, :] = d - d * (1 - d) * beta0 * cdot[i, :, :] - d * (1 - d) * (alpha1 + gamma1 * ea2[i]) * (u ** 2 + v ** 2) + d / 2 * (1 - d) * (1 - 2 * d) * beta0 ** 2 * cdot[i, :, :] ** 2
        check_array(f, 0, 1)
        return f

    def macro_to_f_exact(self, rhoall, uall, vall, epsilonall):
        ex = self.ex
        ey = self.ey
        ea2 = self.ea2
        nx = self.nx
        ny = self.ny
        feq = np.zeros([9, ny, nx])
        for jy in range(ny):
            for ix in range(nx):
                rho = rhoall[jy, ix]
                u = uall[jy, ix]
                v = vall[jy, ix]
                epsilon = epsilonall[jy, ix]

                def equations(vars):
                    alpha, beta, gamma = vars
                    f = np.zeros(9)
                    for i in range(9):
                        cdot = ex[i] * u + ey[i] * v
                        f[i] = 1 / (1 + np.exp(alpha + beta * cdot + gamma * ea2[i]))
                    eq1 = np.sum(f) - rho
                    eq2 = np.dot(f, ex) - rho * u
                    eq3 = np.dot(f, ey) - rho * v
                    return [eq1, eq2, eq3]
                initial_guess = [1 / (1 - epsilon), -1 / epsilon, (1 - 3 * epsilon) / 4 / epsilon ** 2 / (1 - epsilon)]
                [alpha, beta, gamma] = fsolve(equations, initial_guess)
                for i in range(9):
                    cdot = ex[i] * u + ey[i] * v
                    feq[i, jy, ix] = 1 / (1 + np.exp(alpha + beta * cdot + gamma * ea2[i]))
        return feq

    def f_to_macro_thermal(self, f):
        rho = np.sum(f, axis=0)
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))
        epsilon = np.zeros((self.ny, self.nx))
        for i in range(9):
            u += self.ex[i] * f[i, :, :]
            v += self.ey[i] * f[i, :, :]
        u = u / rho
        v = v / rho
        for i in range(9):
            epsilon += ((self.ex[i] - u) ** 2 + (self.ey[i] - v) ** 2) * f[i, :, :] / 2
        return (rho, u, v, epsilon)

    def f_to_macro(self, f):
        rho = np.sum(f, axis=0)
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))
        for i in range(9):
            u += self.ex[i] * f[i, :, :]
            v += self.ey[i] * f[i, :, :]
        u = u / rho
        v = v / rho
        return (rho, u, v)

    def convect(self, f):
        for i in range(1, 9):
            f[i, :, :] = copy.deepcopy(np.roll(f[i, :, :], shift=self.step * self.ex[i], axis=1))
            f[i, :, :] = copy.deepcopy(np.roll(f[i, :, :], shift=self.step * self.ey[i], axis=0))
        return f

    def collide_ORT(self, f, feq):
        ff = np.zeros((9, self.ny, self.nx))
        om_p = self.omega
        for i in range(9):
            ff[i, :, :] = (1.0 - om_p) * f[i, :, :] + om_p * feq[i, :, :]
        return ff

    def collide_TRT(self, f, feq):
        om_p = self.omega_p
        om_m = self.omega_m
        ns = [0, 2, 1, 4, 3, 6, 5, 8, 7]
        g_up = np.zeros((9, self.ny, self.nx))
        for q in range(1, 9):
            qb = ns[q]
            g_up[q, :, :] = (1.0 - 0.5 * (om_p + om_m)) * f[q, :, :] - 0.5 * (om_p - om_m) * f[qb, :, :] + 0.5 * (om_p + om_m) * feq[q, :, :] + 0.5 * (om_p - om_m) * feq[qb, :, :]
        g_up[0, :, :] = (1.0 - om_p) * f[0, :, :] + om_p * feq[0, :, :]
        return g_up

    def collide_LGA(self, f, coll_coef_list):
        ff = f
        i1 = 1
        i2 = 3
        i3 = 2
        i4 = 4
        t1, t2 = (x * coll_coef_list[0] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 5
        i2 = 7
        i3 = 6
        i4 = 8
        t1, t2 = (x * coll_coef_list[1] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0
        i2 = 5
        i3 = 1
        i4 = 2
        t1, t2 = (x * coll_coef_list[2] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0
        i2 = 6
        i3 = 2
        i4 = 3
        t1, t2 = (x * coll_coef_list[2] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0
        i2 = 7
        i3 = 3
        i4 = 4
        t1, t2 = (x * coll_coef_list[2] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0
        i2 = 8
        i3 = 4
        i4 = 1
        t1, t2 = (x * coll_coef_list[2] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 4
        i2 = 5
        i3 = 8
        i4 = 2
        t1, t2 = (x * coll_coef_list[3] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 1
        i2 = 6
        i3 = 5
        i4 = 3
        t1, t2 = (x * coll_coef_list[3] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 2
        i2 = 7
        i3 = 6
        i4 = 4
        t1, t2 = (x * coll_coef_list[3] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 3
        i2 = 8
        i3 = 7
        i4 = 1
        t1, t2 = (x * coll_coef_list[3] for x in f_HPP(f, i1, i2, i3, i4))
        ff[i1, :] = ff[i1, :] - t1 + t2
        ff[i2, :] = ff[i2, :] - t1 + t2
        ff[i3, :] = ff[i3, :] + t1 - t2
        ff[i4, :] = ff[i4, :] + t1 - t2
        return ff

    def collide_LGA_commute(self):
        self.f[1, :, :], self.f[3, :, :], self.f[2, :, :], self.f[4, :, :] = copy.deepcopy(self.HPP_coll(self.f[1, :, :], self.f[3, :, :], self.f[2, :, :], self.f[4, :, :]))
        self.f[5, :, :], self.f[7, :, :], self.f[6, :, :], self.f[8, :, :] = copy.deepcopy(self.HPP_coll(self.f[5, :, :], self.f[7, :, :], self.f[6, :, :], self.f[8, :, :]))
        quadrant = [1, 2, 3, 4]
        random.shuffle(quadrant)
        for i in quadrant:
            self.f[0, :, :], self.f[i + 4, :, :], self.f[i, :, :], self.f[i % 4 + 1, :, :] = copy.deepcopy(self.HPP_coll(self.f[0, :, :], self.f[i + 4, :, :], self.f[i, :, :], self.f[i % 4 + 1, :, :]))
        quadrant = [1, 2, 3, 4]
        random.shuffle(quadrant)
        for i in quadrant:
            self.f[(i - 2) % 4 + 1, :, :], self.f[i + 4, :, :], self.f[i, :, :], self.f[(i - 2) % 4 + 5, :, :] = copy.deepcopy(self.HPP_coll(self.f[(i - 2) % 4 + 1, :, :], self.f[i + 4, :, :], self.f[i, :, :], self.f[(i - 2) % 4 + 5, :, :]))

    def HPP_coll(self, a, b, c, d):
        if a.shape != b.shape or b.shape != c.shape or c.shape != d.shape:
            raise ValueError(f'HPP dimension Error')
        check_array(a)
        check_array(b)
        check_array(c)
        check_array(d)
        fp = a * b * (1 - c) * (1 - d)
        fr = (1 - a) * (1 - b) * c * d
        a = a - fp + fr
        b = b - fp + fr
        c = c + fp - fr
        d = d + fp - fr
        return (a, b, c, d)

    def f_H(self, A):
        f = A.reshape(9, self.ny, self.nx)
        H = np.zeros([self.ny, self.nx])
        w = self.w
        for i in range(9):
            H = H - f[i, :, :] * np.log(f[i, :, :] / w[i])
        return H

    def f_H1(self, A):
        f = A.reshape(9, self.ny, self.nx)
        H = np.zeros([self.ny, self.nx])
        w = self.w
        for i in range(9):
            H = H - f[i, :, :] * np.log(f[i, :, :]) - (1 - f[i, :, :]) * np.log(1 - f[i, :, :])
        return H

    def config_H(self, A):
        config = A.reshape(512, self.ny * self.nx)
        H = np.zeros([self.ny * self.nx])
        for i in range(512):
            H = H - config[i, :] * np.log(config[i, :])
        return H

class qD2Q9(D2Q9):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import SwapGate, HGate, MCXGate
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    def __init__(self, nx, ny, xmin=None, xmax=None, ymin=None, ymax=None):
        super().__init__(nx, ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        self.rho_weight = [4, 2, 2, 2, 2, 1, 1, 1, 1]

    def macro_to_config():
        pass

    def frho_to_config(self):
        nx = self.nx
        ny = self.ny
        f_lattice = self.frho.reshape(9 * self.nx * self.ny)
        import itertools
        config = []
        combinations = list(itertools.product([0, 1], repeat=9))
        for combination in combinations:
            product = np.ones(nx * ny)
            for i, bit in enumerate(combination):
                if bit == 0:
                    product = product * (1 - f_lattice[(8 - i) * nx * ny:(9 - i) * nx * ny])
                else:
                    product = product * f_lattice[(8 - i) * nx * ny:(9 - i) * nx * ny]
            config.append(product)
        config = np.concatenate(config)
        self.config = config

    def config_to_frho(self):
        n = self.nx * self.ny
        f9 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 256))
        f8 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 128))
        f7 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 64))
        f6 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 32))
        f5 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 16))
        f4 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 8))
        f3 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 4))
        f2 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 2))
        f1 = np.sum((self.config[i * n:(i + 1) * n] for i in range(512) if i & 1))
        self.frho = np.reshape(np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8, f9]), (9, self.ny, self.nx))

    def f_to_config(self, A):
        nx = self.nx
        ny = self.ny
        f_lattice = A.reshape(9 * self.nx * self.ny)
        import itertools
        config = []
        combinations = list(itertools.product([0, 1], repeat=9))
        for combination in combinations:
            product = np.ones(nx * ny)
            for i, bit in enumerate(combination):
                if bit == 0:
                    product = product * (1 - f_lattice[(8 - i) * nx * ny:(9 - i) * nx * ny])
                else:
                    product = product * f_lattice[(8 - i) * nx * ny:(9 - i) * nx * ny]
            config.append(product)
        config = np.concatenate(config)
        return config

    def config_to_f(self, config):
        n = self.nx * self.ny
        config = config.ravel()
        f9 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 256))
        f8 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 128))
        f7 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 64))
        f6 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 32))
        f5 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 16))
        f4 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 8))
        f3 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 4))
        f2 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 2))
        f1 = np.sum((config[i * n:(i + 1) * n] for i in range(512) if i & 1))
        f_lattice = np.reshape(np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8, f9]), (9, self.ny, self.nx))
        return f_lattice

    def config_to_quantum(self, rho, config):
        config = config / np.sum(rho)
        return np.sqrt(config)

    def quantum_to_config(self, rho, quantum, an_num):
        length = self.nx * self.ny * 512
        statevector = np.asarray(quantum.get_statevector())
        full_Hilbert = statevector.real.astype(np.float64)
        all_lattice = np.zeros(length)
        for i in range(0, 2 ** an_num):
            all_lattice = all_lattice + full_Hilbert[length * i:length * (1 + i)] ** 2
        all_lattice = all_lattice * np.sum(rho)
        return all_lattice

    def q_convect(self, config, m_th):
        n = self.nx * self.ny
        B = config
        counter = 0
        for iii in range(0, 512):
            spin_index = format(iii, '09b')
            bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i]) for i in range(9))
            bit = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9]
            m = np.sum(np.array(bit))
            if m > m_th or m == 0:
                continue
            iiif = config[iii * n:(iii + 1) * n]
            f_temp = np.reshape(iiif, (self.ny, self.nx))
            A = self.convect_diffuse(f_temp, iii)
            B[iii * n:(iii + 1) * n] = A.flatten()
            counter = counter + 1
        return B

    def config_convect_noise(self, config, m_th, p):
        depth = int(np.log2(self.nx * self.ny)) ^ 2 * 40
        if p * depth > 1:
            print('p<1')
        noise = np.ones_like(config) / config.size * self.nx * self.ny
        A = self.q_convect(config, m_th)
        return A * (1 - p * depth) + noise * p * depth

    def q_convect_double(self, config, m_th):
        config = config.reshape(512, 512, self.ny, self.nx)
        counter = 0
        for iii in range(0, 512):
            spin_index = format(iii, '09b')
            bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i]) for i in range(9))
            bit = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9]
            m = np.sum(np.array(bit))
            if m > m_th or m == 0:
                continue
            iiif = config[iii, :, :, :]
            config[iii, :, :, :] = self.convect_diffuse_double(iiif, iii)
            counter = counter + 1
        return config

    def convect_diffuse_double(self, data, iii):
        lattice_temp = np.zeros((512, self.ny, self.nx))
        spin_index = format(iii, '09b')
        bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i]) for i in range(9))
        bit = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9]
        m = np.sum(np.array(bit))
        if m == 0:
            print('m=0!!')
        for ii in range(9):
            if bit[ii] == 1:
                temp = data / m
                temp = np.roll(temp, shift=self.step * self.ex[ii], axis=2)
                temp = np.roll(temp, shift=self.step * self.ey[ii], axis=1)
                lattice_temp = lattice_temp + temp
        return lattice_temp

    def convect_diffuse(self, data, iii):
        lattice_temp = np.zeros((self.ny, self.nx))
        if data.shape != lattice_temp.shape:
            print('convect size error!')
            sys.exit(6)
        spin_index = format(iii, '09b')
        bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i]) for i in range(9))
        bit = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8, bit9]
        m = np.sum(np.array(bit))
        if m == 0:
            print('m=0!!')
        for ii in range(9):
            if bit[ii] == 1:
                temp = data / m
                temp = np.roll(temp, shift=self.step * self.ex[ii], axis=1)
                temp = np.roll(temp, shift=self.step * self.ey[ii], axis=0)
                lattice_temp = lattice_temp + temp
        return lattice_temp

    @staticmethod
    def q_HPP(A, B, ratio):
        if A.shape != B.shape:
            print('q_HPP shape error')
            sys.exit(4)
        return (A * (1 - ratio) + B * ratio, A * ratio + B * (1 - ratio))

    def q_collide_Chen(self, A, coll_coef_list):
        config = np.reshape(A, (2, 2, 2, 2, 2, 2, 2, 2, 2, self.nx * self.ny))
        config[0, 0, 0, 0, 0, 1, 0, 1, 0, :], config[0, 0, 0, 0, 1, 0, 1, 0, 0, :] = self.q_HPP(config[0, 0, 0, 0, 0, 1, 0, 1, 0, :], config[0, 0, 0, 0, 1, 0, 1, 0, 0, :], coll_coef_list[0])
        config[0, 1, 0, 1, 0, 0, 0, 0, 0, :], config[1, 0, 1, 0, 0, 0, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 1, 0, 0, 0, 0, 0, :], config[1, 0, 1, 0, 0, 0, 0, 0, 0, :], coll_coef_list[1])
        config[0, 0, 0, 1, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 0, 1, 1, 0, :] = self.q_HPP(config[0, 0, 0, 1, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 0, 1, 1, 0, :], coll_coef_list[2])
        config[0, 0, 1, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 1, 1, 0, 0, :] = self.q_HPP(config[0, 0, 1, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 1, 1, 0, 0, :], coll_coef_list[2])
        config[0, 1, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 1, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 1, 0, 0, 0, :], coll_coef_list[2])
        config[1, 0, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 0, 0, 1, 0, :] = self.q_HPP(config[1, 0, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 0, 0, 1, 0, :], coll_coef_list[2])
        config[0, 0, 0, 1, 1, 0, 0, 0, 0, :], config[1, 0, 0, 0, 0, 0, 1, 0, 0, :] = self.q_HPP(config[0, 0, 0, 1, 1, 0, 0, 0, 0, :], config[1, 0, 0, 0, 0, 0, 1, 0, 0, :], coll_coef_list[3])
        config[0, 0, 1, 0, 0, 0, 0, 1, 0, :], config[0, 0, 0, 1, 0, 1, 0, 0, 0, :] = self.q_HPP(config[0, 0, 1, 0, 0, 0, 0, 1, 0, :], config[0, 0, 0, 1, 0, 1, 0, 0, 0, :], coll_coef_list[3])
        config[0, 1, 0, 0, 0, 0, 1, 0, 0, :], config[0, 0, 1, 0, 1, 0, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 0, 0, 0, 1, 0, 0, :], config[0, 0, 1, 0, 1, 0, 0, 0, 0, :], coll_coef_list[3])
        config[1, 0, 0, 0, 0, 1, 0, 0, 0, :], config[0, 1, 0, 0, 0, 0, 0, 1, 0, :] = self.q_HPP(config[1, 0, 0, 0, 0, 1, 0, 0, 0, :], config[0, 1, 0, 0, 0, 0, 0, 1, 0, :], coll_coef_list[3])
        config = config.flatten()
        return config

    def config_collide_noise(self, A, coll_coef_list, p):
        depth = 80
        if p * depth > 1:
            print('p<1')
        noise_o = np.reshape(A, (512, self.nx * self.ny))
        partial_trace = np.sum(noise_o, axis=0)
        arrays = [partial_trace for _ in range(512)]
        noise = np.stack(arrays, axis=0) / 512
        noise = noise.flatten()
        config = self.q_collide_Chen(A, coll_coef_list)
        return config * (1 - p * depth) + noise * p * depth

    def q_collide_double_1(self, config, coll_coef_list):
        config = config.reshape(2, 2, 2, 2, 2, 2, 2, 2, 2, 512 * self.nx * self.ny)
        config[0, 0, 0, 0, 0, 1, 0, 1, 0, :], config[0, 0, 0, 0, 1, 0, 1, 0, 0, :] = self.q_HPP(config[0, 0, 0, 0, 0, 1, 0, 1, 0, :], config[0, 0, 0, 0, 1, 0, 1, 0, 0, :], coll_coef_list[0])
        config[0, 1, 0, 1, 0, 0, 0, 0, 0, :], config[1, 0, 1, 0, 0, 0, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 1, 0, 0, 0, 0, 0, :], config[1, 0, 1, 0, 0, 0, 0, 0, 0, :], coll_coef_list[1])
        config[0, 0, 0, 1, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 0, 1, 1, 0, :] = self.q_HPP(config[0, 0, 0, 1, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 0, 1, 1, 0, :], coll_coef_list[2])
        config[0, 0, 1, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 1, 1, 0, 0, :] = self.q_HPP(config[0, 0, 1, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 0, 1, 1, 0, 0, :], coll_coef_list[2])
        config[0, 1, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 1, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 1, 0, 0, 0, :], coll_coef_list[2])
        config[1, 0, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 0, 0, 1, 0, :] = self.q_HPP(config[1, 0, 0, 0, 0, 0, 0, 0, 1, :], config[0, 0, 0, 0, 1, 0, 0, 1, 0, :], coll_coef_list[2])
        config[0, 0, 0, 1, 1, 0, 0, 0, 0, :], config[1, 0, 0, 0, 0, 0, 1, 0, 0, :] = self.q_HPP(config[0, 0, 0, 1, 1, 0, 0, 0, 0, :], config[1, 0, 0, 0, 0, 0, 1, 0, 0, :], coll_coef_list[3])
        config[0, 0, 1, 0, 0, 0, 0, 1, 0, :], config[0, 0, 0, 1, 0, 1, 0, 0, 0, :] = self.q_HPP(config[0, 0, 1, 0, 0, 0, 0, 1, 0, :], config[0, 0, 0, 1, 0, 1, 0, 0, 0, :], coll_coef_list[3])
        config[0, 1, 0, 0, 0, 0, 1, 0, 0, :], config[0, 0, 1, 0, 1, 0, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 0, 0, 0, 1, 0, 0, :], config[0, 0, 1, 0, 1, 0, 0, 0, 0, :], coll_coef_list[3])
        config[1, 0, 0, 0, 0, 1, 0, 0, 0, :], config[0, 1, 0, 0, 0, 0, 0, 1, 0, :] = self.q_HPP(config[1, 0, 0, 0, 0, 1, 0, 0, 0, :], config[0, 1, 0, 0, 0, 0, 0, 1, 0, :], coll_coef_list[3])
        config = config.flatten()
        config = config.reshape(512, 2, 2, 2, 2, 2, 2, 2, 2, 2, self.nx * self.ny)
        config[:, 0, 0, 0, 0, 0, 1, 0, 1, 0, :], config[:, 0, 0, 0, 0, 1, 0, 1, 0, 0, :] = self.q_HPP(config[:, 0, 0, 0, 0, 0, 1, 0, 1, 0, :], config[:, 0, 0, 0, 0, 1, 0, 1, 0, 0, :], coll_coef_list[0])
        config[:, 0, 1, 0, 1, 0, 0, 0, 0, 0, :], config[:, 1, 0, 1, 0, 0, 0, 0, 0, 0, :] = self.q_HPP(config[:, 0, 1, 0, 1, 0, 0, 0, 0, 0, :], config[:, 1, 0, 1, 0, 0, 0, 0, 0, 0, :], coll_coef_list[1])
        config[:, 0, 0, 0, 1, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 0, 0, 1, 1, 0, :] = self.q_HPP(config[:, 0, 0, 0, 1, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 0, 0, 1, 1, 0, :], coll_coef_list[2])
        config[:, 0, 0, 1, 0, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 0, 1, 1, 0, 0, :] = self.q_HPP(config[:, 0, 0, 1, 0, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 0, 1, 1, 0, 0, :], coll_coef_list[2])
        config[:, 0, 1, 0, 0, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 1, 1, 0, 0, 0, :] = self.q_HPP(config[:, 0, 1, 0, 0, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 1, 1, 0, 0, 0, :], coll_coef_list[2])
        config[:, 1, 0, 0, 0, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 1, 0, 0, 1, 0, :] = self.q_HPP(config[:, 1, 0, 0, 0, 0, 0, 0, 0, 1, :], config[:, 0, 0, 0, 0, 1, 0, 0, 1, 0, :], coll_coef_list[2])
        config[:, 0, 0, 0, 1, 1, 0, 0, 0, 0, :], config[:, 1, 0, 0, 0, 0, 0, 1, 0, 0, :] = self.q_HPP(config[:, 0, 0, 0, 1, 1, 0, 0, 0, 0, :], config[:, 1, 0, 0, 0, 0, 0, 1, 0, 0, :], coll_coef_list[3])
        config[:, 0, 0, 1, 0, 0, 0, 0, 1, 0, :], config[:, 0, 0, 0, 1, 0, 1, 0, 0, 0, :] = self.q_HPP(config[:, 0, 0, 1, 0, 0, 0, 0, 1, 0, :], config[:, 0, 0, 0, 1, 0, 1, 0, 0, 0, :], coll_coef_list[3])
        config[:, 0, 1, 0, 0, 0, 0, 1, 0, 0, :], config[:, 0, 0, 1, 0, 1, 0, 0, 0, 0, :] = self.q_HPP(config[:, 0, 1, 0, 0, 0, 0, 1, 0, 0, :], config[:, 0, 0, 1, 0, 1, 0, 0, 0, 0, :], coll_coef_list[3])
        config[:, 1, 0, 0, 0, 0, 1, 0, 0, 0, :], config[:, 0, 1, 0, 0, 0, 0, 0, 1, 0, :] = self.q_HPP(config[:, 1, 0, 0, 0, 0, 1, 0, 0, 0, :], config[:, 0, 1, 0, 0, 0, 0, 0, 1, 0, :], coll_coef_list[3])
        config = config.flatten()
        return config

    def q_collide_double_2(self, config, coll_coef_list):
        config = config.reshape(2, 2, 2, 2, 2, 2, 8, 64, 2, 2, 2, self.nx * self.ny)
        config[0, 0, 0, 0, 0, 1, :, :, 0, 1, 0, :], config[0, 0, 0, 0, 1, 0, :, :, 1, 0, 0, :] = self.q_HPP(config[0, 0, 0, 0, 0, 1, :, :, 0, 1, 0, :], config[0, 0, 0, 0, 1, 0, :, :, 1, 0, 0, :], coll_coef_list[0])
        config[0, 1, 0, 1, 0, 0, :, :, 0, 0, 0, :], config[1, 0, 1, 0, 0, 0, :, :, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 1, 0, 0, :, :, 0, 0, 0, :], config[1, 0, 1, 0, 0, 0, :, :, 0, 0, 0, :], coll_coef_list[1])
        config[0, 0, 0, 1, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 0, 0, :, :, 1, 1, 0, :] = self.q_HPP(config[0, 0, 0, 1, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 0, 0, :, :, 1, 1, 0, :], coll_coef_list[2])
        config[0, 0, 1, 0, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 0, 1, :, :, 1, 0, 0, :] = self.q_HPP(config[0, 0, 1, 0, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 0, 1, :, :, 1, 0, 0, :], coll_coef_list[2])
        config[0, 1, 0, 0, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 1, 1, :, :, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 0, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 1, 1, :, :, 0, 0, 0, :], coll_coef_list[2])
        config[1, 0, 0, 0, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 1, 0, :, :, 0, 1, 0, :] = self.q_HPP(config[1, 0, 0, 0, 0, 0, :, :, 0, 0, 1, :], config[0, 0, 0, 0, 1, 0, :, :, 0, 1, 0, :], coll_coef_list[2])
        config[0, 0, 0, 1, 1, 0, :, :, 0, 0, 0, :], config[1, 0, 0, 0, 0, 0, :, :, 1, 0, 0, :] = self.q_HPP(config[0, 0, 0, 1, 1, 0, :, :, 0, 0, 0, :], config[1, 0, 0, 0, 0, 0, :, :, 1, 0, 0, :], coll_coef_list[3])
        config[0, 0, 1, 0, 0, 0, :, :, 0, 1, 0, :], config[0, 0, 0, 1, 0, 1, :, :, 0, 0, 0, :] = self.q_HPP(config[0, 0, 1, 0, 0, 0, :, :, 0, 1, 0, :], config[0, 0, 0, 1, 0, 1, :, :, 0, 0, 0, :], coll_coef_list[3])
        config[0, 1, 0, 0, 0, 0, :, :, 1, 0, 0, :], config[0, 0, 1, 0, 1, 0, :, :, 0, 0, 0, :] = self.q_HPP(config[0, 1, 0, 0, 0, 0, :, :, 1, 0, 0, :], config[0, 0, 1, 0, 1, 0, :, :, 0, 0, 0, :], coll_coef_list[3])
        config[1, 0, 0, 0, 0, 1, :, :, 0, 0, 0, :], config[0, 1, 0, 0, 0, 0, :, :, 0, 1, 0, :] = self.q_HPP(config[1, 0, 0, 0, 0, 1, :, :, 0, 0, 0, :], config[0, 1, 0, 0, 0, 0, :, :, 0, 1, 0, :], coll_coef_list[3])
        config = config.reshape(64, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, self.nx * self.ny)
        config[:, 0, 1, 0, 0, 0, 0, 0, 0, 1, :, :], config[:, 1, 0, 0, 0, 0, 0, 0, 1, 0, :, :] = self.q_HPP(config[:, 0, 1, 0, 0, 0, 0, 0, 0, 1, :, :], config[:, 1, 0, 0, 0, 0, 0, 0, 1, 0, :, :], coll_coef_list[0])
        config[:, 0, 0, 0, 1, 0, 1, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 1, 0, 1, 0, 0, :, :] = self.q_HPP(config[:, 0, 0, 0, 1, 0, 1, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 1, 0, 1, 0, 0, :, :], coll_coef_list[1])
        config[:, 0, 0, 1, 0, 0, 0, 1, 0, 0, :, :], config[:, 1, 1, 0, 0, 0, 0, 0, 0, 0, :, :] = self.q_HPP(config[:, 0, 0, 1, 0, 0, 0, 1, 0, 0, :, :], config[:, 1, 1, 0, 0, 0, 0, 0, 0, 0, :, :], coll_coef_list[2])
        config[:, 0, 0, 1, 0, 0, 1, 0, 0, 0, :, :], config[:, 1, 0, 0, 0, 0, 0, 0, 0, 1, :, :] = self.q_HPP(config[:, 0, 0, 1, 0, 0, 1, 0, 0, 0, :, :], config[:, 1, 0, 0, 0, 0, 0, 0, 0, 1, :, :], coll_coef_list[2])
        config[:, 0, 0, 1, 0, 1, 0, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 0, 0, 0, 1, 1, :, :] = self.q_HPP(config[:, 0, 0, 1, 0, 1, 0, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 0, 0, 0, 1, 1, :, :], coll_coef_list[2])
        config[:, 0, 0, 1, 1, 0, 0, 0, 0, 0, :, :], config[:, 0, 1, 0, 0, 0, 0, 0, 1, 0, :, :] = self.q_HPP(config[:, 0, 0, 1, 1, 0, 0, 0, 0, 0, :, :], config[:, 0, 1, 0, 0, 0, 0, 0, 1, 0, :, :], coll_coef_list[2])
        config[:, 0, 0, 0, 0, 0, 0, 1, 1, 0, :, :], config[:, 1, 0, 0, 1, 0, 0, 0, 0, 0, :, :] = self.q_HPP(config[:, 0, 0, 0, 0, 0, 0, 1, 1, 0, :, :], config[:, 1, 0, 0, 1, 0, 0, 0, 0, 0, :, :], coll_coef_list[3])
        config[:, 0, 1, 0, 0, 0, 1, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 0, 0, 1, 0, 1, :, :] = self.q_HPP(config[:, 0, 1, 0, 0, 0, 1, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 0, 0, 1, 0, 1, :, :], coll_coef_list[3])
        config[:, 1, 0, 0, 0, 1, 0, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 0, 1, 0, 1, 0, :, :] = self.q_HPP(config[:, 1, 0, 0, 0, 1, 0, 0, 0, 0, :, :], config[:, 0, 0, 0, 0, 0, 1, 0, 1, 0, :, :], coll_coef_list[3])
        config[:, 0, 0, 0, 1, 0, 0, 0, 0, 1, :, :], config[:, 0, 1, 0, 0, 1, 0, 0, 0, 0, :, :] = self.q_HPP(config[:, 0, 0, 0, 1, 0, 0, 0, 0, 1, :, :], config[:, 0, 1, 0, 0, 1, 0, 0, 0, 0, :, :], coll_coef_list[3])
        config = config.flatten()
        return config

    def q_collision(self, A, coll_coef_list):
        config = np.reshape(A, (2, 2, 2, 2, 2, 2, 2, 2, 2, self.nx * self.ny))
        config[0, 1, 0, 1, :, :, :, :, :, :], config[1, 0, 1, 0, :, :, :, :, :, :] = self.q_HPP(config[0, 1, 0, 1, :, :, :, :, :, :], config[1, 0, 1, 0, :, :, :, :, :, :], coll_coef_list[1])

        def code1():
            config[:, :, :, 1, :, :, 0, 0, 1, :], config[:, :, :, 0, :, :, 1, 1, 0, :] = self.q_HPP(config[:, :, :, 1, :, :, 0, 0, 1, :], config[:, :, :, 0, :, :, 1, 1, 0, :], coll_coef_list[2])

        def code2():
            config[:, :, 1, :, :, 0, 0, :, 1, :], config[:, :, 0, :, :, 1, 1, :, 0, :] = self.q_HPP(config[:, :, 1, :, :, 0, 0, :, 1, :], config[:, :, 0, :, :, 1, 1, :, 0, :], coll_coef_list[2])

        def code3():
            config[:, 1, :, :, 0, 0, :, :, 1, :], config[:, 0, :, :, 1, 1, :, :, 0, :] = self.q_HPP(config[:, 1, :, :, 0, 0, :, :, 1, :], config[:, 0, :, :, 1, 1, :, :, 0, :], coll_coef_list[2])

        def code4():
            config[1, :, :, :, 0, :, :, 0, 1, :], config[0, :, :, :, 1, :, :, 1, 0, :] = self.q_HPP(config[1, :, :, :, 0, :, :, 0, 1, :], config[0, :, :, :, 1, :, :, 1, 0, :], coll_coef_list[2])
        code_list = [code1, code2, code3, code4]
        random.shuffle(code_list)
        for code in code_list:
            code()

        def code5():
            config[0, :, :, 1, 1, :, 0, :, :, :], config[1, :, :, 0, 0, :, 1, :, :, :] = self.q_HPP(config[0, :, :, 1, 1, :, 0, :, :, :], config[1, :, :, 0, 0, :, 1, :, :, :], coll_coef_list[3])

        def code6():
            config[:, :, 1, 0, :, 0, :, 1, :, :], config[:, :, 0, 1, :, 1, :, 0, :, :] = self.q_HPP(config[:, :, 1, 0, :, 0, :, 1, :, :], config[:, :, 0, 1, :, 1, :, 0, :, :], coll_coef_list[3])

        def code7():
            config[:, 1, 0, :, 0, :, 1, :, :, :], config[:, 0, 1, :, 1, :, 0, :, :, :] = self.q_HPP(config[:, 1, 0, :, 0, :, 1, :, :, :], config[:, 0, 1, :, 1, :, 0, :, :, :], coll_coef_list[3])

        def code8():
            config[1, 0, :, :, :, 1, :, 0, :, :], config[0, 1, :, :, :, 0, :, 1, :, :] = self.q_HPP(config[1, 0, :, :, :, 1, :, 0, :, :], config[0, 1, :, :, :, 0, :, 1, :, :], coll_coef_list[3])
        code_list = [code5, code6, code7, code8]
        random.shuffle(code_list)
        for code in code_list:
            code()
        config[:, :, :, :, 0, 1, 0, 1, :, :], config[:, :, :, :, 1, 0, 1, 0, :, :] = self.q_HPP(config[:, :, :, :, 0, 1, 0, 1, :, :], config[:, :, :, :, 1, 0, 1, 0, :, :], coll_coef_list[0])
        config = config.flatten()
        return config

    def q_collision_variedensity(self):
        n = self.nx * self.ny
        self.gamma2_1 = 0.5
        self.gamma2_4 = 0.2
        self.gamma1_2 = 0.2
        self.gamma1_4 = 0.1
        self.gamma4_1 = 0.3
        self.gamma4_2 = 0.2
        t_list0 = [10, 20, 1, 480]
        g2 = np.zeros((len(t_list0), n))
        g2_temp = np.zeros((len(t_list0), n))
        for i in range(len(t_list0)):
            g2[i, :] = self.config[t_list0[i] * n:(t_list0[i] + 1) * n]
        g2_temp[0, :] = g2[0, :] * (1 - self.gamma2_1 - self.gamma2_4) + g2[2, :] * self.gamma1_2 / 2 + g2[3, :] * self.gamma4_2 / 2
        g2_temp[1, :] = g2[1, :] * (1 - self.gamma2_1 - self.gamma2_4) + g2[2, :] * self.gamma1_2 / 2 + g2[3, :] * self.gamma4_2 / 2
        g2_temp[2, :] = g2[2, :] * (1 - self.gamma1_2 - self.gamma1_4) + g2[0, :] * self.gamma2_1 + g2[1, :] * self.gamma2_1 + g2[3, :] * self.gamma4_1
        g2_temp[3, :] = g2[3, :] * (1 - self.gamma4_1 - self.gamma4_2) + g2[0, :] * self.gamma2_4 + g2[1, :] * self.gamma2_4 + g2[2, :] * self.gamma1_4
        for i in range(len(t_list0)):
            self.config[t_list0[i] * n:(t_list0[i] + 1) * n] = g2_temp[i, :]
        self.beta1_2 = 1
        self.beta2_1 = 1
        t_list2_2 = [384, 192, 96, 288]
        b2 = np.zeros((len(t_list2_2), n))
        for i in range(len(t_list2_2)):
            b2[i, :] = self.config[t_list2_2[i] * n:(t_list2_2[i] + 1) * n]
        t_list2_1 = [16, 8, 4, 2]
        b1 = np.zeros((len(t_list2_1), n))
        for i in range(len(t_list2_1)):
            b1[i, :] = self.config[t_list2_1[i] * n:(t_list2_1[i] + 1) * n]
        b2_temp = b2 * (1 - self.beta2_1) + b1 * self.beta1_2
        b1_temp = b1 * (1 - self.beta1_2) + b2 * self.beta2_1
        for i in range(len(t_list2_2)):
            self.config[t_list2_2[i] * n:(t_list2_2[i] + 1) * n] = b2_temp[i, :]
        for i in range(len(t_list2_1)):
            self.config[t_list2_1[i] * n:(t_list2_1[i] + 1) * n] = b1_temp[i, :]

def densitymatrix_to_array(state):
    threshold = 1e-15
    diagonal_elements = np.diag(state.data)
    real_diagonal_elements = np.real(diagonal_elements)
    arr_copy = real_diagonal_elements.copy()
    arr_copy[np.abs(arr_copy) < threshold] = 0
    return arr_copy

def diag_to_config(diag, rho):
    return diag * np.sum(rho)

def check_array(A, lower=0, upper=1):
    import sys
    if np.any(A > upper):
        print('Values beyond max.')
        sys.exit(1)
    if np.any(A < lower):
        positions = np.where(A < 0)
        values = A[positions]
        print('Positions of elements < 0:', positions)
        print('Values of elements < 0:', values)
        print('Values lower minimal.')
        sys.exit(2)

def get_portion(A, B):
    print(f'the portion is {np.sum(A) / np.sum(B)}')

def average_deviation(A, B):
    difference = np.abs(A - B)
    aa = np.mean(np.abs(A))
    bb = np.mean(np.abs(B))
    cc = np.mean(difference)
    print(f'相对误差是{cc / aa}')
    return cc

def get_enstrophy(w):
    temp = 0.5 * w * w
    ave = np.mean(temp)
    return ave

def energy_spectrum(u, v):
    ny, nx = np.shape(u)
    kx = fftshift(np.fft.fftfreq(nx)) * nx
    ky = fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_max = min(nx, ny) // 2
    u_fft = fftshift(fft2(u)) / (ny * nx)
    v_fft = fftshift(fft2(v)) / (ny * nx)
    energy_density = (np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2) / 2
    spectrum = np.zeros(k_max)
    for k in range(k_max):
        mask = (k <= K) & (K < k + 1)
        spectrum[k] = np.sum(energy_density[mask])
    k_range = np.arange(k_max)
    return (k_range[1:], spectrum[1:])

def get_spectrum(vor):
    ny, nx = np.shape(vor)
    kx = fftshift(np.fft.fftfreq(nx)) * nx
    ky = fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_max = min(nx, ny) // 2
    u_fft = fftshift(fft2(vor)) / (ny * nx)
    energy_density = np.abs(u_fft) ** 2
    spectrum = np.zeros(k_max)
    for k in range(k_max):
        mask = (k <= K) & (K < k + 1)
        spectrum[k] = np.sum(energy_density[mask])
    k_range = np.arange(k_max)
    return (k_range[1:], spectrum[1:])

def spectrum_2D(vor):
    k, spe = get_spectrum(vor)
    plt.figure()
    plt.loglog(k, spe, 'b-', label='Energy Spectrum')
    plt.xlabel('Wave number (k)')
    plt.ylabel('Energy')
    plt.title('2D Turbulence Energy Spectrum')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def f_HPP(A, i1, i2, i3, i4):
    total_indices = A.shape[0]
    all_indices = set(range(total_indices))
    other_indices = list(all_indices - {i1, i2, i3, i4})
    product1 = A[i1, :] * A[i2, :]
    for idx in other_indices:
        product1 *= 1 - A[idx, :]
    product2 = A[i3, :] * A[i4, :]
    for idx in other_indices:
        product2 *= 1 - A[idx, :]
    return (product1, product2)

def f_to_state(A):
    length, ny, nx = A.shape
    f_lattice = A.reshape(length * ny * nx)
    import itertools
    config = []
    combinations = list(itertools.product([0, 1], repeat=length))
    for combination in combinations:
        product = np.ones(nx * ny)
        for i, bit in enumerate(combination):
            if bit == 0:
                product = product * (1 - f_lattice[(length - 1 - i) * nx * ny:(length - i) * nx * ny])
            else:
                product = product * f_lattice[(length - 1 - i) * nx * ny:(length - i) * nx * ny]
        config.append(product)
    config = np.concatenate(config)
    return config

def state_to_f(config, nx, ny, length):
    n = nx * ny
    assert len(config) == 2 ** length * n, 'Config length must be 2^length * n'
    f_list = []
    for bit in range(length):
        f = np.sum((config[i * n:(i + 1) * n] for i in range(2 ** length) if i & 1 << bit))
        f_list.append(f)
    f_lattice = np.reshape(np.concatenate(f_list), (length, ny, nx))
    return f_lattice

def state_H(config1, nx, ny, length):
    f = state_to_f(config1, nx, ny, length)
    f_lattice = f.reshape(length * ny * nx)
    import itertools
    config = []
    combinations = list(itertools.product([0, 1], repeat=length))
    for combination in combinations:
        product = np.zeros(nx * ny)
        for i, bit in enumerate(combination):
            if bit == 0:
                product = product + (1 - f_lattice[(length - 1 - i) * nx * ny:(length - i) * nx * ny])
            else:
                product = product + f_lattice[(length - 1 - i) * nx * ny:(length - i) * nx * ny]
        config.append(product)
    config = np.concatenate(config)
    config = config / length / 2 ** (length - 1)
    return config

def state_noise_uniform(config, p):
    temp = np.ones_like(config) / np.size(config)
    return config * (1 - p) + temp * p

def state_noise_binomial(config, p, depth, size):
    samples = np.random.binomial(depth, p, size=size)
    samples = np.clip(samples, 0, size / 2)
    clipped_samples = samples / size * p
    return config * (1 - p) + clipped_samples * p

def state_noise_Gauss(config, n, p, depth):
    size = np.size(config)
    samples = np.random.normal(loc=p * depth, scale=p * np.sqrt(depth), size=size)
    clipped_samples = samples / size
    clipped_samples = np.clip(clipped_samples, 0, 1)
    clipped_samples = clipped_samples / np.sum(clipped_samples) * p * depth * n * n
    return config * (1 - p * depth) + clipped_samples

def state_noise_Gauss_partial(config, n, p, depth):
    pass

def state_to_double(state, length, nx, ny):
    if state.size != 2 ** length * nx * ny:
        print(f'config size error!')
        sys.exit(1)
    config = state.reshape(2 ** length, nx * ny)
    shape = (2 ** length, 2 ** length, nx * ny)
    double_config = np.zeros(shape, dtype=np.float32)
    for i in range(nx * ny):
        double_config[:, :, i] = np.outer(config[:, i], config[:, i])
    print(double_config.dtype)
    return double_config

def print_memory_usage():
    process = os.getpid()
    info = resource.getrusage(resource.RUSAGE_SELF)
    mem = info.ru_maxrss / 1024
    print(f'Process {process} consumes {mem:.2f} MB of memory.')