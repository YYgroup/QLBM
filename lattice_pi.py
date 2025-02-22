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
from scipy.optimize import root,fsolve

class triangle():
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.xmax = 1
        self.xmin = -1
        self.ymax = self.xmax/2*np.sqrt(3)*ny/nx
        self.ymin = self.xmin/2*np.sqrt(3)*ny/nx
        self.dx = (self.xmax-self.xmin)/nx
        self.dy = (self.ymax-self.ymin)/ny

    def plot(self, vor,i=None, vmin=None, vmax=None, str='RdBu_r'):
        nx = self.nx; ny = self.ny 
        lx=self.xmax-self.xmin; ly=self.ymax-self.ymin
        if i is None:
            i=0
        if vmin is None:
            vmin=np.min(vor)
        if vmax is None:
            vmax=np.max(vor)
        X = np.linspace(-1+lx/nx, 1, nx) 
        Y = np.linspace(-1+ly/ny, 1, ny)
        x, y = np.meshgrid(X, Y)
        plt.figure(i)
        # plt.contourf(x, y, vor, cmap=str, vmin=vmin, vmax=vmax,levels=6) #bwr 'Reds'
        plt.pcolormesh(x, y,vor,cmap=str, vmin=vmin, vmax=vmax)
        # plt.gca().set_aspect(np.sqrt(3)/2)
        plt.gca().set_aspect(1)
        plt.colorbar(label='Vorticity',shrink=0.9)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('rho=1')
        plt.show()

    def FHP_Gauss(self, vn1, vn2, Gamma, sigma):
        """
        vn1, vn2是涡中心左下的网格坐标
        """
        n1 = self.nx
        n2 = self.ny
        pi=np.pi
        x = np.zeros ((n2,n1)) 
        y = np.zeros ((n2,n1)) 
        u = np.zeros ((n2,n1)) 
        v = np.zeros ((n2,n1)) 
        rt=3*sigma
        delta1=self.dx; delta2=delta1
        x0=(vn1-n1/2+1)*delta1+(vn2+0.5-n1/2+1)*delta2/2
        y0=(vn2-n1/2+1+2/3)*delta2*np.sqrt(3)/2
        for i in range (n1):
            for j in range (n2): 
                x[j][i]=(i-n1/2+1)*delta1+(j-n2+1)*delta2/2
                y[j][i]=(j-n1/2+1)*delta2*np.sqrt(3)/2
        X = np.where(x <= -1, x + 2, x)
        X = np.where(X > 1, X - 2, X)
        xx=X-x0; yy=y-y0
        r = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        # utheta = np.where(r < rt, Gamma/2/pi/r *(1- np.exp(-r**2/2/sigma**2)),Gamma/2/pi/r *(1- np.exp(-9/2)))
        utheta = Gamma/2/pi/r *(1- np.exp(-r**2/2/sigma**2))-r^2/(self.xmax-self.xmin)/(self.ymax-self.ymin)*Gamma/2/pi/r*(1- np.exp(-9/2))
        v = utheta*np.cos(theta)
        u = -utheta*np.sin(theta)
        return x,y,u,v

class square():

    def __init__(self, nx, ny, xmin=-1, xmax=1, ymin=-1, ymax=1):
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Lx = xmax-xmin
        self.Ly = ymax-ymin
        self.dx = (self.xmax-self.xmin)/nx
        self.dy = (self.ymax-self.ymin)/ny
        self.name = ""
        self.step = 1 #每个时间步对流几个网格
        self.ex = []
        self.ey = []

    def __str__(self):
        return f"Square lattice with space grid: {self.nx}*{self.ny}"
    
    def print_vel(self,u,v,cs2):
        cs = np.sqrt(cs2)
        magnitude = np.sqrt(u**2 + v**2)
        max_magnitude = np.max(magnitude)
        # print("maximum SPEED is:", max_magnitude)
        # print("grid SPEED is:",max_magnitude/np.sqrt(self.dx**2+self.dy**2))
        print("Mach number is:",max_magnitude/np.sqrt(self.dx**2+self.dy**2)/cs)

    def init_Gauss(self, vnx, vny, Gamma, sigma):
        """
        vnx, vny是涡中心左下的网格坐标
        sigma是物理空间的
        """
        nx=self.nx; ny=self.ny
        lx=self.xmax-self.xmin; ly=self.ymax-self.ymin
        X = np.linspace(-1+lx/nx, 1, nx)
        Y = np.linspace(-1+ly/ny, 1, ny)
        x, y = np.meshgrid(X, Y)
        pi=np.pi
        x0=vnx*lx/nx-1+1.5*lx/nx
        y0=vny*ly/ny-1+1.5*ly/ny
        xx=x-x0; yy=y-y0
        r = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        utheta = Gamma/2/pi/r *(1- np.exp(-r**2/2/sigma**2))
        # print(np.exp(-r**2/2/sigma**2))
        v =  utheta*np.cos(theta)
        u = -utheta*np.sin(theta)
        return X,Y,u,v
    
    def init_TG(self, gamma):
        """
        根据坐标值生成TG流
        gamma是一个涡眼的涡通量
        对于[-pi,pi]的网格，一个眼睛的通量是gamma，那么，
        表达式是u = gamma/8 sincos; v = -gamma/8 cossin; omega = gamma/4 sinsin
        对于一般的网格[-a,a]暂时不考虑
        """
        coef = gamma/8
        ny=self.ny; nx=self.nx
        u = np.zeros ((self.ny,self.nx)) 
        v = np.zeros ((self.ny,self.nx)) 
        for i in range (nx):
            for j in range (ny): 
                x=self.xmin+(self.Lx/self.nx)*(i+1)
                y=self.ymin+(self.Ly/self.ny)*(j+1)
                u[j][i]= coef*math.sin(x)*math.cos(y)
                v[j][i]=-coef*math.sin(y)*math.cos(x)
        return u,v
    
    def Gauss_vortex(self, vnx, vny, Gamma, sigma):
        """
        vnx, vny是涡中心左下的网格坐标
        f=1/2 pi sigma^2 exp(-x^2/2 sigma^2)
        """
        ny=self.ny; nx=self.nx
        pi = np.pi
        lx=self.xmax-self.xmin; ly=self.ymax-self.ymin
        X = np.linspace(self.xmin+lx/nx, self.xmax, nx) 
        Y = np.linspace(self.ymin+ly/ny, self.ymax, ny)
        x, y = np.meshgrid(X, Y)
        x0=vnx*lx/nx+self.xmin+1.5*lx/nx
        y0=vny*ly/ny+self.ymin+1.5*ly/ny
        xx=x-x0; yy=y-y0
        r = np.sqrt(xx**2 + yy**2)
        # theta = np.arctan2(yy, xx)
        return Gamma/2/pi/sigma**2 * np.exp(-r**2/2/sigma**2)
    
    def cross_vortex(self, vnx, vny, amp, width):
        ny=self.ny; nx=self.nx
        nnx = width / (self.xmax-self.xmin) * self.nx
        nny = width / (self.ymax-self.ymin) * self.ny
        vor = np.zeros ((ny,nx)) 
        vor[: , round(vnx-1-nnx/2):round(vnx-1+nnx/2)] = amp
        vor[round(vny-1-nny/2):round(vny-1+nny/2) , :] = amp
        return vor

    def vor_periodic(self, vor):
        fft_data = np.fft.fft2(vor)
        fft_data[0, 0] = 0  # ensure average is 0
        return np.real(ifft2(fft_data))
    
    def vor_to_psi(self, vor):
        '''
        这一步得到了流函数
        '''
        nx=self.nx; ny=self.ny
        if vor.shape != (ny, nx):
            raise ValueError(f"Error: The size must be ({ny}, {nx}).")
        vor_filter = self.vor_periodic(vor)
        ny, nx = self.ny, self.nx
        kx = np.fft.fftfreq(nx, (self.xmax-self.xmin) / nx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, (self.ymax-self.ymin) / ny) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)
        k_squared = kx**2 + ky**2
        k_squared[0, 0] = 1  # Avoid division by zero
        psi_hat = fft2(vor_filter) / (k_squared)        
        psi = np.real(ifft2(psi_hat))
        return psi
    
    def psi_to_vel(self, psi):
        ''' u = ppx; v= -ppy'''
        nx=self.nx; ny=self.ny
        dx=self.dx; dy=self.dy
        if psi.shape != (ny, nx):
            raise ValueError(f"Error: The size must be ({ny}, {nx}).")
        u_ext = np.pad(psi, ((1, 1), (1, 1)), mode='wrap')
        v =-(u_ext[1:-1, 2:] - u_ext[1:-1, :-2]) / (2 * dx)
        u = (u_ext[2:, 1:-1] - u_ext[:-2, 1:-1]) / (2 * dy)
        return u,v
    
    def plot(self, vor,i=None, vmin=None, vmax=None, str='RdBu_r'):
        nx = self.nx; ny = self.ny 
        lx=self.xmax-self.xmin; ly=self.ymax-self.ymin
        if i is None:
            i=0
        if vmin is None:
            vmin=np.min(vor)
        if vmax is None:
            vmax=np.max(vor)
        X = np.linspace(self.xmin+lx/nx, self.xmax, nx) 
        Y = np.linspace(self.ymin+ly/ny, self.ymax, ny)
        x, y = np.meshgrid(X, Y)
        plt.figure(i)
        # plt.contourf(x, y, vor, cmap=str, vmin=vmin, vmax=vmax,levels=6) #bwr 'Reds'
        plt.pcolormesh(x, y,vor,cmap=str, vmin=vmin, vmax=vmax)
        # plt.gca().set_aspect(np.sqrt(3)/2)
        plt.gca().set_aspect(1)
        plt.colorbar(label='Vorticity',shrink=0.9)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('rho=1')
        plt.show()

    def streamline_plot(self, u, v):
        nx = self.nx; ny = self.ny 
        speed = np.sqrt(u**2 + v**2)
        lx=self.xmax-self.xmin; ly=self.ymax-self.ymin
        X = np.linspace(-1+lx/nx, 1, nx) 
        Y = np.linspace(-1+ly/ny, 1, ny)
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
        '''
        输入一个2维向量，然后，然后进行fft
        频率宽度1是-0.5到0.5，cutoff根据这个选
        这个对频谱一定是削弱，没有前面的归一化系数
        '''
        self.plot(data)
        fft_data = np.fft.fftshift(np.fft.fft2(data))
        # self.plot(np.abs(fft_data))
        x = np.linspace(-0.5, 0.5, self.ny)
        y = np.linspace(-0.5, 0.5, self.nx)
        X, Y = np.meshgrid(x, y)
        d = np.sqrt(X**2 + Y**2)
        filter_ = np.exp(-(d**2) / (2 * (cutoff**2)))
        filtered_fft_data = fft_data * filter_
        # self.plot(np.abs(filtered_fft_data))
        return np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_data)))
    
    def poisson_solver_fft(self, w):
        '''
        就是求解流函数方程nabla^2 f = -w
        直接用fourier解会怎么样呢？是物理解吗，反解是w吗
        '''
        ny, nx = self.ny, self.nx
        kx = np.fft.fftfreq(nx).reshape(1, nx) #真的方便啊，直接就是标准fft
        ky = np.fft.fftfreq(ny).reshape(ny, 1)
        kx, ky = np.meshgrid(kx, ky)
        k_squared = kx**2 + ky**2
        k_squared[0, 0] = 1  # Avoid division by zero
        
        psi_hat = fft2(w) / (-k_squared)
        psi_hat[0, 0] = 0  # Set the mean of psi to zero
        
        psi = np.real(ifft2(psi_hat))
        return psi

    def vorcitity(self, u, v):
        # Calculate the partial derivatives
        nx=self.nx; ny=self.ny
        dx=self.dx; dy=self.dy
        if u.shape != (ny, nx) or v.shape != (ny, nx):
            raise ValueError(f"Error: The size of u and v must be ({ny}, {nx}). "
                         f"Got u: {u.shape}, v: {v.shape}")
        u_ext = np.pad(u, ((1, 1), (1, 1)), mode='wrap')
        v_ext = np.pad(v, ((1, 1), (1, 1)), mode='wrap')
        dvdx = (v_ext[:, 2:] - v_ext[:, :-2]) / (2 * dx)
        dudy = (u_ext[2:, :] - u_ext[:-2, :]) / (2 * dy)
        
        # Calculate the vorticity
        vorticity = dvdx[1:-1, :] - dudy[:, 1:-1]
        return vorticity
    
    def write_scalar(self, vor , time , basename):
        name=self.name
        np.save( basename+str(time)+name+'.npy', vor)

    def write_file(self,vor,rho,time, name1=None, name2=None):
        name=self.name
        if name1 is None:
            name1="vor"
        if name2 is None:
            name2="rho"
        
        if name1+name+".npz" in os.listdir('.'):
            data = np.load(name1+name+".npz")
            data = dict(data)
        else:
            data = {}
        data[f'vor_{time}'] = vor
        np.savez(name1+name+".npz", **data)
        
        if name2+name+".npz" in os.listdir('.'):
            data = np.load(name2+name+".npz")
            data = dict(data)
        else:
            data = {}
        data[f'rho_{time}'] = rho
        np.savez(name2+name+".npz", **data)

########################  D2Q9  #####################################
class D2Q9(square):
    def __init__(self, nx, ny, xmin=None, xmax=None, ymin=None, ymax=None):
        super().__init__(
            nx, ny,
            xmin=xmin if xmin is not None else -1,
            xmax=xmax if xmax is not None else 1,
            ymin=ymin if ymin is not None else -1,
            ymax=ymax if ymax is not None else 1
        )

        self.step = 1
        self.cs2 = 1/3
        self.Re = 10
        self.ex = [0, 1, 0, -1, 0,   1, -1, -1, 1]  # Override ex
        self.ey = [0, 0, 1, 0, -1,   1, 1, -1, -1]  # Override ey
        self.ea2 = [0,  1,1,1,1, 2,2,2,2]
        t0=4/9; t1=1/9; t2=1/36 #下标是速度的模平方t0
        self.w = [t0,t1,t1,t1,t1,t2,t2,t2,t2]
        self.set_viscosity() #通过Re设定各种弛豫时间

        # self.f =   np.zeros((9,self.ny,self.nx))
        # self.feq = np.zeros((9,self.ny,self.nx))
        # self.rho = np.ones((self.ny,self.nx))
        # self.u = np.zeros((self.ny,self.nx))
        # self.v = np.zeros((self.ny,self.nx))
        # self.vor = np.zeros((self.ny,self.nx))

    def set_viscosity(self, Re=None):
        if Re is None:
            Re = self.Re
        self.nu = 1/Re # self.u_avg*self.L/self.Re L=ny
        # print(f"Re={Re}, grid viscosity={self.nu}")
        self.tau = 0.5 + self.nu/(self.cs2)
        self.omega     = 1.0/self.tau
        self.tau_p  = self.tau #双时间尺度第一个时间尺度
        self.lambda_trt = 1.0/4.0 # Best for stability
        self.tau_m  = self.lambda_trt/(self.tau_p - 0.5) + 0.5 #双时间尺度的第二个尺度
        self.omega_p   = 1.0/self.tau_p
        self.omega_m   = 1.0/self.tau_m

    def get_equil(self, rho, u, v):
        '''
        在二阶近似下，平衡态分布是：N=rho t [1+eu/cs^2+(eu)^2/2cs^4-(u^2+v^2)/2cs^2]
        cs^2=1/3
        '''
        feq = np.zeros((9,self.ny,self.nx))
        cdot= np.zeros((9,self.ny,self.nx))
        w=self.w
        ex = self.ex; ey = self.ey
        cs2 = self.cs2
        # print(f"u max is {np.max(u)}, min is {np.min(u)}")
        # print(f"v max is {np.max(v)}, min is {np.min(v)}")
        for i in range(9):
            cdot[i,:,:]=ex[i]*u+ey[i]*v
            # print(f"max is {np.max(cdot[i,:,:])}, min is {np.min(cdot[i,:,:])}")
            feq[i,:,:]=rho*w[i]*(1+cdot[i,:,:]/cs2+cdot[i,:,:]*cdot[i,:,:]/2/cs2**2-(u*u+v*v)/2/cs2)
            check_array(feq[i,:,:],0,1)
        return feq

    def macro_to_f_LGA(self, rho, u, v, epsilon):
        '''
        f ea**2/2 - uea +u**2/2 = sum fa ea**2 - u**2/2
        '''
        ex=self.ex
        ey=self.ey
        ea2 = self.ea2
        
        alpha1 = 1/(1-epsilon)
        gamma1 = (1-3*epsilon)/4/epsilon**2/(1-epsilon)
        beta0 = - 1/epsilon
        beta1 = (3*epsilon-1)/4/epsilon**3
        d0 = rho*(1-epsilon)**2
        d1 = rho/2*(1-epsilon)*epsilon
        d2 = rho/4*epsilon**2

        f = np.zeros((9,self.ny,self.nx))
        cdot= np.zeros((9,self.ny,self.nx))
        for i in range(9):
            cdot[i,:,:]=ex[i]*u+ey[i]*v
        d = d0
        for i in range(0,1):
            f[i,:,:] = d-d*(1-d)*beta0*cdot[i,:,:] - d*(1-d)*(alpha1+gamma1*ea2[i])*(u**2+v**2) + d/2*(1-d)*(1-2*d)*beta0**2*cdot[i,:,:]**2 
            # f[i,:,:] = d-d*(1-d)*beta0*cdot[i,:,:]*(u**2+v**2) + d*(1-d)*(1-2*d)*beta0*(alpha1+gamma1*ea2[i])*(u**2+v**2) - d/6*(1-d)*(1-6*d+6*d**2)*beta0**3*cdot[i,:,:]**3
        d = d1
        for i in range(1,5):
            f[i,:,:] = d-d*(1-d)*beta0*cdot[i,:,:] - d*(1-d)*(alpha1+gamma1*ea2[i])*(u**2+v**2) + d/2*(1-d)*(1-2*d)*beta0**2*cdot[i,:,:]**2
        d = d2
        for i in range(5,9):
            f[i,:,:] = d-d*(1-d)*beta0*cdot[i,:,:] - d*(1-d)*(alpha1+gamma1*ea2[i])*(u**2+v**2) + d/2*(1-d)*(1-2*d)*beta0**2*cdot[i,:,:]**2
        check_array(f,0,1)
        return f
    
    def macro_to_f_exact(self, rhoall, uall, vall, epsilonall):
        ex=self.ex
        ey=self.ey
        ea2 = self.ea2 
        nx = self.nx
        ny = self.ny
        feq = np.zeros([9,ny,nx])
        for jy in range(ny):
            for ix in range(nx):
                rho = rhoall[jy,ix]
                u = uall[jy,ix]
                v = vall[jy,ix]
                epsilon = epsilonall[jy,ix]
                # print(rho,u,v)
                
                def equations(vars):
                    # print("vars:", vars)  # 打印 vars 的内容
                    alpha, beta, gamma = vars
                    f = np.zeros(9)  # 初始化 f 数组
                    for i in range(9):
                        cdot = ex[i] * u + ey[i] * v
                        f[i] = 1 / (1 + np.exp(alpha + beta * cdot + gamma * ea2[i]))
                    
                    eq1 = np.sum(f) - rho
                    eq2 = np.dot(f, ex) - rho * u
                    eq3 = np.dot(f, ey) - rho * v                    
                    return [eq1, eq2, eq3]
                initial_guess = [1/(1-epsilon),-1/epsilon,(1-3*epsilon)/4/epsilon**2/(1-epsilon)]
                # initial_guess = [-1,-1,-1]
                [alpha, beta, gamma] = fsolve(equations, initial_guess)
                for i in range(9):
                    cdot = ex[i] * u + ey[i] * v
                    feq[i,jy,ix] = 1 / (1 + np.exp(alpha + beta * cdot + gamma * ea2[i]))
        return feq

    def f_to_macro_thermal(self, f):
        rho = np.sum(f, axis=0)
        u=np.zeros((self.ny,self.nx))
        v=np.zeros((self.ny,self.nx))
        epsilon = np.zeros((self.ny,self.nx))
        for i in range(9):
            u += self.ex[i]*f[i,:,:]
            v += self.ey[i]*f[i,:,:]
        u = u/rho
        v = v/rho
        for i in range(9):
            epsilon += ((self.ex[i]-u)**2+(self.ey[i]-v)**2)*f[i,:,:]/2
        return rho,u,v,epsilon

    def f_to_macro(self, f):
        '''
        内部过程，从self.f到self.u/v/rho
        '''
        rho = np.sum(f, axis=0)
        # ex=[0,  1, 0, -1, 0,   1, -1, -1, 1]
        # ey=[0,  0, 1, 0, -1,   1, 1, -1, -1]
        u=np.zeros((self.ny,self.nx))
        v=np.zeros((self.ny,self.nx))
        for i in range(9):
            u += self.ex[i]*f[i,:,:]
            v += self.ey[i]*f[i,:,:]
        u = u/rho
        v = v/rho
        return rho,u,v
        
    def convect(self,f):
        # A = copy.deepcopy(self.f[1,:,:])
        for i in range(1,9):
            f[i,:,:] = copy.deepcopy(np.roll(f[i,:,:], shift=self.step*self.ex[i], axis=1))
            f[i,:,:] = copy.deepcopy(np.roll(f[i,:,:], shift=self.step*self.ey[i], axis=0))
        return f

    def collide_ORT(self, f, feq ):
        '''
        tau是特征时间，omega是松弛因子
        '''
        ff = np.zeros((9,self.ny,self.nx))
        om_p=self.omega
        for i in range(9):
            ff[i,:,:] = (1.0-om_p)*f[i,:,:] + om_p*feq[i,:,:]
        return ff

    def collide_TRT(self, f, feq):
        '''
        二阶时间松弛因子:omega;
        特征时间tau
        '''
        om_p = self.omega_p
        om_m = self.omega_m
        ns=[0,2,1,4,3,6,5,8,7]
        # Take care of q=0 first
        g_up=np.zeros((9,self.ny,self.nx))
        # Collide other indices
        for q in range(1,9):
            qb = ns[q]
            g_up[q,:,:] = ((1.0-0.5*(om_p+om_m))*f[q,:,:]   -
                                0.5*(om_p-om_m)*f[qb,:,:]   +
                                0.5*(om_p+om_m)*feq[q,:,:] +
                                0.5*(om_p-om_m)*feq[qb,:,:])
        g_up[0,:,:] = (1.0-om_p)*f[0,:,:] + om_p*feq[0,:,:]
        return g_up

    def collide_LGA(self, f, coll_coef_list):
        '''
        在f表象下的碰撞
        碰撞规则：在各个方向都是相同密度的情况下，
        1. 两种对心碰撞，分别是e2-e5和e6-e9之间的对心碰撞
        2. e6-e9和e1之间的碰撞，斜的+静止的=两个正的
        3. 一个斜的和一个正的，碰成对称的形状
        输入的f要写成9*n的二维形式
        ''' 
        ff = f

        i1 = 1; i2 = 3; i3 = 2; i4 = 4
        t1,t2 = (x*coll_coef_list[0] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 5; i2 = 7; i3 = 6; i4 = 8
        t1,t2 = (x*coll_coef_list[1] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2

        i1 = 0; i2 = 5; i3 = 1; i4 = 2
        t1,t2 = (x*coll_coef_list[2] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0; i2 = 6; i3 = 2; i4 = 3
        t1,t2 = (x*coll_coef_list[2] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0; i2 = 7; i3 = 3; i4 = 4
        t1,t2 = (x*coll_coef_list[2] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 0; i2 = 8; i3 = 4; i4 = 1
        t1,t2 = (x*coll_coef_list[2] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2

        i1 = 4; i2 = 5; i3 = 8; i4 = 2
        t1,t2 = (x*coll_coef_list[3] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 1; i2 = 6; i3 = 5; i4 = 3
        t1,t2 = (x*coll_coef_list[3] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 2; i2 = 7; i3 = 6; i4 = 4
        t1,t2 = (x*coll_coef_list[3] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2
        i1 = 3; i2 = 8; i3 = 7; i4 = 1
        t1,t2= (x*coll_coef_list[3] for x in f_HPP(f,i1,i2,i3,i4))
        ff[i1, :] = ff[i1, :] - t1 + t2; ff[i2, :] = ff[i2, :] - t1 + t2; ff[i3, :] = ff[i3, :] + t1 - t2; ff[i4, :] = ff[i4, :] + t1 - t2

        return ff

    def collide_LGA_commute(self):
        '''
        分别测试几个step的可交换性质，测试不同顺序的碰撞是否会极大影响结果。
        这就体现出了这种编码的好处，只作用于一部分的Qbit能体现出一大类碰撞，所以碰撞项可以并不复杂。
        这也提供了一种Hstep的思路，即让尽可能多的状态参与碰撞，尤其是rho_l比较小的态，要参加碰撞
        可惜的是。rho=0,1的态在这种情况下永远不会参与碰撞，他们一个是保证密度小，一个是管对流，高阶项才是管碰撞的，所以对流至少一阶精度，废话。
        这就带来了很操蛋的问题，只能说，能算。
        Q: 按照真实idea物理分布，是不是一个线性的东西呢？首先它肯定是保迹的
        A：其实就是最简单的酉矩阵性质，对于重合的态有一个到很多态的演化
        Q：这种独立的算符写成矩阵变换的形式会是什么样子呢？怎么理解这个过程中的可分离性？
        '''
        # A = self.f[1,:,:]

        # case 1 正碰
        self.f[1,:,:],self.f[3,:,:],self.f[2,:,:],self.f[4,:,:] = copy.deepcopy(self.HPP_coll(self.f[1,:,:],self.f[3,:,:],self.f[2,:,:],self.f[4,:,:]))
        # B,_,_,_ = self.HPP_coll(self.f[1,:,:],self.f[3,:,:],self.f[2,:,:],self.f[4,:,:])
        self.f[5,:,:],self.f[7,:,:],self.f[6,:,:],self.f[8,:,:] = copy.deepcopy(self.HPP_coll(self.f[5,:,:],self.f[7,:,:],self.f[6,:,:],self.f[8,:,:]))
        # case 2 静止碰撞：设置了一种随机方式，使得每个时间步的侧重方向都不一样
        quadrant = [1,2,3,4] # 代表第几象限内的作用
        random.shuffle(quadrant)
        for i in quadrant:
            self.f[0,:,:],self.f[i+4,:,:],self.f[i,:,:],self.f[i%4+1,:,:] = copy.deepcopy(self.HPP_coll(self.f[0,:,:],self.f[i+4,:,:],self.f[i,:,:],self.f[i%4+1,:,:])) # %余数的优先级很高的
        # case 3 斜碰，同样设置一种随机方式
        quadrant = [1,2,3,4] # 代表动量在哪个方向 i是给定方向，两个正方向分别是i-2%4+1，i%4+1 (核心就是要设计进位点)；两个斜方向是i和i-2%4+1
        random.shuffle(quadrant)
        for i in quadrant:
            self.f[(i-2)%4+1,:,:],self.f[i+4,:,:],self.f[i,:,:],self.f[(i-2)%4+5,:,:] = copy.deepcopy(self.HPP_coll(self.f[(i-2)%4+1,:,:],self.f[i+4,:,:],self.f[i,:,:],self.f[(i-2)%4+5,:,:])) # %余数的优先级很高的
        # average_deviation(A,B)

    # @staticmethod
    def HPP_coll(self,a,b,c,d):
        '''
        ab是碰撞前的方向，cd是碰撞后的方向；当且仅当ab有且cd无或者ab无且cd有的情况下，发生碰撞
        在分布函数的表象下，保持可逆性是一件困难的事情， 不可逆的事情发生的非常自然。
        对角线上一个元素随时融合到另一个元素里。但是确实符合量子通道的原理。
        首先计算每个时间发生的概率，随着每个事件发生，都会对原有事件产生影响。这个是不是非线性的我需要想一想
        '''
        if a.shape != b.shape or b.shape != c.shape or c.shape != d.shape:
            raise ValueError(f"HPP dimension Error")
        check_array(a); check_array(b); check_array(c); check_array(d)
        fp = a * b * (1-c) * (1-d)
        fr = (1-a) * (1-b) * c * d
        # print(fp)
        # print(fr)
        a = a - fp + fr; b = b - fp + fr
        c = c + fp - fr; d = d + fp - fr
        return a,b,c,d
    
    def f_H(self, A):
        f = A.reshape(9,self.ny,self.nx)
        H = np.zeros([self.ny,self.nx])
        w=self.w
        for i in range(9):
            # H = H-f[i,:,:]*np.log(f[i,:,:]/w[i])-(1-f[i,:,:])*np.log((1-f[i,:,:])/w[i])
            H = H-f[i,:,:]*np.log(f[i,:,:]/w[i])
        return H
    
    def f_H1(self, A):
        f = A.reshape(9,self.ny,self.nx)
        H = np.zeros([self.ny,self.nx])
        w=self.w
        for i in range(9):
            H = H-f[i,:,:]*np.log(f[i,:,:])-(1-f[i,:,:])*np.log((1-f[i,:,:]))
        return H
    
    def config_H(self, A):
        config = A.reshape(512,self.ny*self.nx)
        H = np.zeros([self.ny*self.nx])
        for i in range(512):
            H = H-config[i,:]*np.log(config[i,:])
        return H

class qD2Q9(D2Q9):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister 
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import SwapGate,HGate, MCXGate
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    def __init__(self, nx, ny, xmin=None, xmax=None, ymin=None, ymax=None):
        super().__init__(nx, ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        self.rho_weight = [4,2,2,2,2,1,1,1,1]

    def macro_to_config():
        '''
        从3个宏观量衍生出另外的，并非平衡态的config
        rho,u,v三个量，变化到config的
        001010000 010100000 000000101 000001010
        1开头8个量
        有两种思路：1. macro直接到config。之前就是这么做的，很扯淡。
        2. 先求出f，然后在得config。
        现在还有一个问题，就是平衡态的H为什么不是最大，唉
        '''
        pass

    def frho_to_config(self): 
        nx = self.nx; ny = self.ny
        f_lattice = self.frho.reshape(9*self.nx*self.ny)
        import itertools
        config = []
        combinations = list(itertools.product([0, 1], repeat=9))
        for combination in combinations:
            product = np.ones (nx*ny) 
            for i, bit in enumerate(combination):
                if bit == 0:
                    product = product* (1 - f_lattice[(8-i)*nx*ny:(9-i)*nx*ny])
                else:
                    product = product* f_lattice[(8-i)*nx*ny:(9-i)*nx*ny]
            config.append(product)
            # print(f"size of product is {len(lattice)}")
        config = np.concatenate(config)
        # print(f"length of config is {config.shape}")
        self.config = config
        # print(f"length of config is {self.config.shape}")

    def config_to_frho(self):
        '''
        从config计算frho
        '''
        n=self.nx*self.ny
        # A=lattice
        f9 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b100000000)
        f8 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b010000000)
        f7 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b001000000)
        f6 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b000100000)
        f5 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b000010000)
        f4 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b000001000)
        f3 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b000000100)
        f2 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b000000010)
        f1 = np.sum(self.config[i*n:(i+1)*n] for i in range(512) if i & 0b000000001)
        self.frho = np.reshape(np.concatenate([f1,f2,f3,f4,f5,f6,f7,f8,f9]),(9,self.ny,self.nx))

    def f_to_config(self, A): 
        nx = self.nx; ny = self.ny
        f_lattice = A.reshape(9*self.nx*self.ny)
        import itertools
        config = []
        combinations = list(itertools.product([0, 1], repeat=9))
        for combination in combinations:
            product = np.ones (nx*ny) 
            for i, bit in enumerate(combination):
                if bit == 0:
                    product = product* (1 - f_lattice[(8-i)*nx*ny:(9-i)*nx*ny])
                else:
                    product = product* f_lattice[(8-i)*nx*ny:(9-i)*nx*ny]
            config.append(product)
        config = np.concatenate(config)
        return config

    def config_to_f(self, config):
        '''
        从config计算frho
        '''
        n=self.nx*self.ny
        config = config.ravel()
        f9 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b100000000)
        f8 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b010000000)
        f7 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b001000000)
        f6 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b000100000)
        f5 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b000010000)
        f4 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b000001000)
        f3 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b000000100)
        f2 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b000000010)
        f1 = np.sum(config[i*n:(i+1)*n] for i in range(512) if i & 0b000000001)
        f_lattice = np.reshape(np.concatenate([f1,f2,f3,f4,f5,f6,f7,f8,f9]),(9,self.ny,self.nx))
        return f_lattice
    
    def config_to_quantum(self, rho, config):
        '''
        从node内概率，变成系综概率，然后变成量子态概率
        '''
        # if config.shape 
        config = config/np.sum(rho)
        return np.sqrt(config)
    
    def quantum_to_config(self, rho, quantum, an_num):
        length = self.nx*self.ny*512
        statevector = np.asarray(quantum.get_statevector())
        full_Hilbert = statevector.real.astype(np.float64)
        all_lattice = np.zeros(length)
        for i in range(0,2**an_num):
            all_lattice = all_lattice + full_Hilbert[length*i:length*(1+i)]**2
        all_lattice = all_lattice*np.sum(rho) # 要乘以所有的粒子数，因为初始的密度是1，而量子系统的总概率为1
        return all_lattice
    
    def q_convect(self,config,m_th):
        # print(type(config))
        n = self.nx*self.ny
        B = config
        counter = 0
        for iii in range(0,512):
            spin_index = format(iii, '09b')
            bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i])  for i in range(9))
            bit = [bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8,bit9]
            m = np.sum(np.array(bit))
            # print(m)
            if m>m_th or m==0:
                continue
            iiif = config[iii*n:(iii+1)*n]
            f_temp = np.reshape(iiif,(self.ny,self.nx))
            A = self.convect_diffuse(f_temp,iii)  
            B[iii*n:(iii+1)*n] = A.flatten()
            counter = counter+1
        return B
    
    def config_convect_noise(self, config, m_th, p):
        
        depth = int(np.log2(self.nx*self.ny))^2*40
        if p*depth > 1:
            print("概率小于1！！！")

        noise = np.ones_like(config)/config.size*self.nx*self.ny

        A = self.q_convect(config,m_th)
        return A*(1-p*depth)+noise*p*depth
    
    def q_convect_double(self,config,m_th):
        '''
        输入的是512,512,nx*ny大小的数组
        使用扩展的那个维数作为控制
        '''
        config = config.reshape(512,512,self.ny,self.nx)
        counter = 0
        for iii in range(0,512):
            spin_index = format(iii, '09b')
            bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i])  for i in range(9))
            bit = [bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8,bit9]
            m = np.sum(np.array(bit))
            # print(m)
            if m>m_th or m==0:
                continue
            iiif = config[iii,:,:,:]
            config[iii,:,:,:] = self.convect_diffuse_double(iiif,iii)  
            counter = counter+1
        # print(f"there are {counter} transportation")
        return config
    
    def convect_diffuse_double(self,data,iii):
        '''输入的是一个3维的东西'''
        lattice_temp = np.zeros((512,self.ny,self.nx))
        spin_index = format(iii, '09b')
        bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i]) for i in range(9))
        bit = [bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8,bit9]
        m = np.sum(np.array(bit))
        if(m==0):
            print("m=0!!")
        for ii in range(9):
            if bit[ii]==1:
                temp = data/m
                temp = np.roll(temp, shift=self.step*self.ex[ii], axis=2)
                temp = np.roll(temp, shift=self.step*self.ey[ii], axis=1)
                lattice_temp = lattice_temp+temp
        return lattice_temp
    
    def convect_diffuse(self,data,iii):
        lattice_temp = np.zeros((self.ny,self.nx))
        if data.shape!= lattice_temp.shape:
            print("convect size error!")
            sys.exit(6)
        spin_index = format(iii, '09b')
        bit9, bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1 = (int(spin_index[i]) for i in range(9))
        bit = [bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8,bit9]
        m = np.sum(np.array(bit))
        if(m==0):
            print("m=0!!")
        for ii in range(9):
            if bit[ii]==1:
                temp = data/m
                temp = np.roll(temp, shift=self.step*self.ex[ii], axis=1)
                temp = np.roll(temp, shift=self.step*self.ey[ii], axis=0)
                lattice_temp = lattice_temp+temp
        return lattice_temp
    
    @staticmethod
    def q_HPP(A,B,ratio):
        # print(ratio)
        if A.shape != B.shape:
            print("q_HPP shape error")
            sys.exit(4)
        return A*(1-ratio)+B*ratio, A*ratio+B*(1-ratio)

    def q_collide_Chen(self, A, coll_coef_list):
        config = np.reshape(A,(2,2,2,2,2,2,2,2,2,self.nx*self.ny))
        config[0,0,0,0,0,1,0,1,0,:],config[0,0,0,0,1,0,1,0,0,:] = self.q_HPP(config[0,0,0,0,0,1,0,1,0,:],config[0,0,0,0,1,0,1,0,0,:],coll_coef_list[0])

        config[0,1,0,1,0,0,0,0,0,:],config[1,0,1,0,0,0,0,0,0,:] = self.q_HPP(config[0,1,0,1,0,0,0,0,0,:],config[1,0,1,0,0,0,0,0,0,:],coll_coef_list[1])

        config[0,0,0,1,0,0,0,0,1,:],config[0,0,0,0,0,0,1,1,0,:] = self.q_HPP(config[0,0,0,1,0,0,0,0,1,:],config[0,0,0,0,0,0,1,1,0,:],coll_coef_list[2])
        config[0,0,1,0,0,0,0,0,1,:],config[0,0,0,0,0,1,1,0,0,:] = self.q_HPP(config[0,0,1,0,0,0,0,0,1,:],config[0,0,0,0,0,1,1,0,0,:],coll_coef_list[2])
        config[0,1,0,0,0,0,0,0,1,:],config[0,0,0,0,1,1,0,0,0,:] = self.q_HPP(config[0,1,0,0,0,0,0,0,1,:],config[0,0,0,0,1,1,0,0,0,:],coll_coef_list[2])
        config[1,0,0,0,0,0,0,0,1,:],config[0,0,0,0,1,0,0,1,0,:] = self.q_HPP(config[1,0,0,0,0,0,0,0,1,:],config[0,0,0,0,1,0,0,1,0,:],coll_coef_list[2])
        
        config[0,0,0,1,1,0,0,0,0,:],config[1,0,0,0,0,0,1,0,0,:] = self.q_HPP(config[0,0,0,1,1,0,0,0,0,:],config[1,0,0,0,0,0,1,0,0,:],coll_coef_list[3])
        config[0,0,1,0,0,0,0,1,0,:],config[0,0,0,1,0,1,0,0,0,:] = self.q_HPP(config[0,0,1,0,0,0,0,1,0,:],config[0,0,0,1,0,1,0,0,0,:],coll_coef_list[3])
        config[0,1,0,0,0,0,1,0,0,:],config[0,0,1,0,1,0,0,0,0,:] = self.q_HPP(config[0,1,0,0,0,0,1,0,0,:],config[0,0,1,0,1,0,0,0,0,:],coll_coef_list[3])
        config[1,0,0,0,0,1,0,0,0,:],config[0,1,0,0,0,0,0,1,0,:] = self.q_HPP(config[1,0,0,0,0,1,0,0,0,:],config[0,1,0,0,0,0,0,1,0,:],coll_coef_list[3])
        config = config.flatten()
        return config
    
    def config_collide_noise(self, A, coll_coef_list, p):
        
        depth = 80
        if p*depth > 1:
            print("概率小于1！！！")

        noise_o = np.reshape(A,(512,self.nx*self.ny))
        partial_trace = np.sum(noise_o, axis=0)
        arrays = [partial_trace for _ in range(512)]
        noise = np.stack(arrays, axis=0)/512
        noise = noise.flatten()

        config = self.q_collide_Chen(A, coll_coef_list)
        return config*(1-p*depth)+noise*p*depth
    
    def q_collide_double_1(self, config, coll_coef_list):
        config = config.reshape(2, 2,2,2,2, 2,2,2,2, 512*self.nx*self.ny)
        config[0,0,0,0,0,1,0,1,0,:],config[0,0,0,0,1,0,1,0,0,:] = self.q_HPP(config[0,0,0,0,0,1,0,1,0,:],config[0,0,0,0,1,0,1,0,0,:],coll_coef_list[0])
        config[0,1,0,1,0,0,0,0,0,:],config[1,0,1,0,0,0,0,0,0,:] = self.q_HPP(config[0,1,0,1,0,0,0,0,0,:],config[1,0,1,0,0,0,0,0,0,:],coll_coef_list[1])

        config[0,0,0,1,0,0,0,0,1,:],config[0,0,0,0,0,0,1,1,0,:] = self.q_HPP(config[0,0,0,1,0,0,0,0,1,:],config[0,0,0,0,0,0,1,1,0,:],coll_coef_list[2])
        config[0,0,1,0,0,0,0,0,1,:],config[0,0,0,0,0,1,1,0,0,:] = self.q_HPP(config[0,0,1,0,0,0,0,0,1,:],config[0,0,0,0,0,1,1,0,0,:],coll_coef_list[2])
        config[0,1,0,0,0,0,0,0,1,:],config[0,0,0,0,1,1,0,0,0,:] = self.q_HPP(config[0,1,0,0,0,0,0,0,1,:],config[0,0,0,0,1,1,0,0,0,:],coll_coef_list[2])
        config[1,0,0,0,0,0,0,0,1,:],config[0,0,0,0,1,0,0,1,0,:] = self.q_HPP(config[1,0,0,0,0,0,0,0,1,:],config[0,0,0,0,1,0,0,1,0,:],coll_coef_list[2])
        
        config[0,0,0,1,1,0,0,0,0,:],config[1,0,0,0,0,0,1,0,0,:] = self.q_HPP(config[0,0,0,1,1,0,0,0,0,:],config[1,0,0,0,0,0,1,0,0,:],coll_coef_list[3])
        config[0,0,1,0,0,0,0,1,0,:],config[0,0,0,1,0,1,0,0,0,:] = self.q_HPP(config[0,0,1,0,0,0,0,1,0,:],config[0,0,0,1,0,1,0,0,0,:],coll_coef_list[3])
        config[0,1,0,0,0,0,1,0,0,:],config[0,0,1,0,1,0,0,0,0,:] = self.q_HPP(config[0,1,0,0,0,0,1,0,0,:],config[0,0,1,0,1,0,0,0,0,:],coll_coef_list[3])
        config[1,0,0,0,0,1,0,0,0,:],config[0,1,0,0,0,0,0,1,0,:] = self.q_HPP(config[1,0,0,0,0,1,0,0,0,:],config[0,1,0,0,0,0,0,1,0,:],coll_coef_list[3])
        config = config.flatten()

        config = config.reshape(512, 2, 2,2,2,2, 2,2,2,2, self.nx*self.ny)
        config[:,0,0,0,0,0,1,0,1,0,:],config[:,0,0,0,0,1,0,1,0,0,:] = self.q_HPP(config[:,0,0,0,0,0,1,0,1,0,:],config[:,0,0,0,0,1,0,1,0,0,:],coll_coef_list[0])
        config[:,0,1,0,1,0,0,0,0,0,:],config[:,1,0,1,0,0,0,0,0,0,:] = self.q_HPP(config[:,0,1,0,1,0,0,0,0,0,:],config[:,1,0,1,0,0,0,0,0,0,:],coll_coef_list[1])

        config[:,0,0,0,1,0,0,0,0,1,:],config[:,0,0,0,0,0,0,1,1,0,:] = self.q_HPP(config[:,0,0,0,1,0,0,0,0,1,:],config[:,0,0,0,0,0,0,1,1,0,:],coll_coef_list[2])
        config[:,0,0,1,0,0,0,0,0,1,:],config[:,0,0,0,0,0,1,1,0,0,:] = self.q_HPP(config[:,0,0,1,0,0,0,0,0,1,:],config[:,0,0,0,0,0,1,1,0,0,:],coll_coef_list[2])
        config[:,0,1,0,0,0,0,0,0,1,:],config[:,0,0,0,0,1,1,0,0,0,:] = self.q_HPP(config[:,0,1,0,0,0,0,0,0,1,:],config[:,0,0,0,0,1,1,0,0,0,:],coll_coef_list[2])
        config[:,1,0,0,0,0,0,0,0,1,:],config[:,0,0,0,0,1,0,0,1,0,:] = self.q_HPP(config[:,1,0,0,0,0,0,0,0,1,:],config[:,0,0,0,0,1,0,0,1,0,:],coll_coef_list[2])
        
        config[:,0,0,0,1,1,0,0,0,0,:],config[:,1,0,0,0,0,0,1,0,0,:] = self.q_HPP(config[:,0,0,0,1,1,0,0,0,0,:],config[:,1,0,0,0,0,0,1,0,0,:],coll_coef_list[3])
        config[:,0,0,1,0,0,0,0,1,0,:],config[:,0,0,0,1,0,1,0,0,0,:] = self.q_HPP(config[:,0,0,1,0,0,0,0,1,0,:],config[:,0,0,0,1,0,1,0,0,0,:],coll_coef_list[3])
        config[:,0,1,0,0,0,0,1,0,0,:],config[:,0,0,1,0,1,0,0,0,0,:] = self.q_HPP(config[:,0,1,0,0,0,0,1,0,0,:],config[:,0,0,1,0,1,0,0,0,0,:],coll_coef_list[3])
        config[:,1,0,0,0,0,1,0,0,0,:],config[:,0,1,0,0,0,0,0,1,0,:] = self.q_HPP(config[:,1,0,0,0,0,1,0,0,0,:],config[:,0,1,0,0,0,0,0,1,0,:],coll_coef_list[3])
        config = config.flatten()
        return config
    
    def q_collide_double_2(self, config, coll_coef_list):
        '''
        第二类碰撞类型：
        编组1: 0,1,2
        编组2: 3,4,5,6,7,8
        config[0,0,0,0, 0,0,:(8), :(64),0,0,0,:]
        '''
        # config[0,0,0,0, 0,0,:, :,0,0,0,:]
        config = config.reshape(2,2,2,2, 2,2,8,  64,2,2, 2, self.nx*self.ny)
        config[0,0,0,0, 0,1,:, :,0,1,0,:],config[0,0,0,0, 1,0,:, :,1,0,0,:] = self.q_HPP(config[0,0,0,0, 0,1,:, :,0,1,0,:],config[0,0,0,0, 1,0,:, :,1,0,0,:],coll_coef_list[0])
        config[0,1,0,1, 0,0,:, :,0,0,0,:],config[1,0,1,0, 0,0,:, :,0,0,0,:] = self.q_HPP(config[0,1,0,1, 0,0,:, :,0,0,0,:],config[1,0,1,0, 0,0,:, :,0,0,0,:],coll_coef_list[1])

        config[0,0,0,1, 0,0,:, :,0,0,1,:],config[0,0,0,0, 0,0,:, :,1,1,0,:] = self.q_HPP(config[0,0,0,1, 0,0,:, :,0,0,1,:],config[0,0,0,0, 0,0,:, :,1,1,0,:],coll_coef_list[2])
        config[0,0,1,0, 0,0,:, :,0,0,1,:],config[0,0,0,0, 0,1,:, :,1,0,0,:] = self.q_HPP(config[0,0,1,0, 0,0,:, :,0,0,1,:],config[0,0,0,0, 0,1,:, :,1,0,0,:],coll_coef_list[2])
        config[0,1,0,0, 0,0,:, :,0,0,1,:],config[0,0,0,0, 1,1,:, :,0,0,0,:] = self.q_HPP(config[0,1,0,0, 0,0,:, :,0,0,1,:],config[0,0,0,0, 1,1,:, :,0,0,0,:],coll_coef_list[2])
        config[1,0,0,0, 0,0,:, :,0,0,1,:],config[0,0,0,0, 1,0,:, :,0,1,0,:] = self.q_HPP(config[1,0,0,0, 0,0,:, :,0,0,1,:],config[0,0,0,0, 1,0,:, :,0,1,0,:],coll_coef_list[2])
        
        config[0,0,0,1, 1,0,:, :,0,0,0,:],config[1,0,0,0, 0,0,:, :,1,0,0,:] = self.q_HPP(config[0,0,0,1, 1,0,:, :,0,0,0,:],config[1,0,0,0, 0,0,:, :,1,0,0,:],coll_coef_list[3])
        config[0,0,1,0, 0,0,:, :,0,1,0,:],config[0,0,0,1, 0,1,:, :,0,0,0,:] = self.q_HPP(config[0,0,1,0, 0,0,:, :,0,1,0,:],config[0,0,0,1, 0,1,:, :,0,0,0,:],coll_coef_list[3])
        config[0,1,0,0, 0,0,:, :,1,0,0,:],config[0,0,1,0, 1,0,:, :,0,0,0,:] = self.q_HPP(config[0,1,0,0, 0,0,:, :,1,0,0,:],config[0,0,1,0, 1,0,:, :,0,0,0,:],coll_coef_list[3])
        config[1,0,0,0, 0,1,:, :,0,0,0,:],config[0,1,0,0, 0,0,:, :,0,1,0,:] = self.q_HPP(config[1,0,0,0, 0,1,:, :,0,0,0,:],config[0,1,0,0, 0,0,:, :,0,1,0,:],coll_coef_list[3])

        config = config.reshape(64,2,2,2, 2,2,2,2, 2,2,8, self.nx*self.ny)
        config[:,0,1,0, 0,0,0,0, 0,1,:,:],config[:,1,0,0, 0,0,0,0, 1,0,:,:] = self.q_HPP(config[:,0,1,0, 0,0,0,0, 0,1,:,:],config[:,1,0,0, 0,0,0,0, 1,0,:,:],coll_coef_list[0])
        config[:,0,0,0, 1,0,1,0, 0,0,:,:],config[:,0,0,0, 0,1,0,1, 0,0,:,:] = self.q_HPP(config[:,0,0,0, 1,0,1,0, 0,0,:,:],config[:,0,0,0, 0,1,0,1, 0,0,:,:],coll_coef_list[1])

        config[:,0,0,1, 0,0,0,1, 0,0,:,:],config[:,1,1,0, 0,0,0,0, 0,0,:,:] = self.q_HPP(config[:,0,0,1, 0,0,0,1, 0,0,:,:],config[:,1,1,0, 0,0,0,0, 0,0,:,:],coll_coef_list[2])
        config[:,0,0,1, 0,0,1,0, 0,0,:,:],config[:,1,0,0, 0,0,0,0, 0,1,:,:] = self.q_HPP(config[:,0,0,1, 0,0,1,0, 0,0,:,:],config[:,1,0,0, 0,0,0,0, 0,1,:,:],coll_coef_list[2])
        config[:,0,0,1, 0,1,0,0, 0,0,:,:],config[:,0,0,0, 0,0,0,0, 1,1,:,:] = self.q_HPP(config[:,0,0,1, 0,1,0,0, 0,0,:,:],config[:,0,0,0, 0,0,0,0, 1,1,:,:],coll_coef_list[2])
        config[:,0,0,1, 1,0,0,0, 0,0,:,:],config[:,0,1,0, 0,0,0,0, 1,0,:,:] = self.q_HPP(config[:,0,0,1, 1,0,0,0, 0,0,:,:],config[:,0,1,0, 0,0,0,0, 1,0,:,:],coll_coef_list[2])
        
        config[:,0,0,0, 0,0,0,1, 1,0,:,:],config[:,1,0,0, 1,0,0,0, 0,0,:,:] = self.q_HPP(config[:,0,0,0, 0,0,0,1, 1,0,:,:],config[:,1,0,0, 1,0,0,0, 0,0,:,:],coll_coef_list[3])
        config[:,0,1,0, 0,0,1,0, 0,0,:,:],config[:,0,0,0, 0,0,0,1, 0,1,:,:] = self.q_HPP(config[:,0,1,0, 0,0,1,0, 0,0,:,:],config[:,0,0,0, 0,0,0,1, 0,1,:,:],coll_coef_list[3])
        config[:,1,0,0, 0,1,0,0, 0,0,:,:],config[:,0,0,0, 0,0,1,0, 1,0,:,:] = self.q_HPP(config[:,1,0,0, 0,1,0,0, 0,0,:,:],config[:,0,0,0, 0,0,1,0, 1,0,:,:],coll_coef_list[3])
        config[:,0,0,0, 1,0,0,0, 0,1,:,:],config[:,0,1,0, 0,1,0,0, 0,0,:,:] = self.q_HPP(config[:,0,0,0, 1,0,0,0, 0,1,:,:],config[:,0,1,0, 0,1,0,0, 0,0,:,:],coll_coef_list[3])
        config = config.flatten()
        return config

    def q_collision(self, A, coll_coef_list):
        config = np.reshape(A,(2,2,2,2,2,2,2,2,2,self.nx*self.ny))
        # print(f"collision coef is {coll_coef_list}")
        config[0,1,0,1,:,:,:,:,:,:],config[1,0,1,0,:,:,:,:,:,:] = self.q_HPP(config[0,1,0,1,:,:,:,:,:,:],config[1,0,1,0,:,:,:,:,:,:],coll_coef_list[1])
        def code1():
            config[:,:,:,1,:,:,0,0,1,:],config[:,:,:,0,:,:,1,1,0,:] = self.q_HPP(config[:,:,:,1,:,:,0,0,1,:],config[:,:,:,0,:,:,1,1,0,:],coll_coef_list[2])
        def code2():
            config[:,:,1,:,:,0,0,:,1,:],config[:,:,0,:,:,1,1,:,0,:] = self.q_HPP(config[:,:,1,:,:,0,0,:,1,:],config[:,:,0,:,:,1,1,:,0,:],coll_coef_list[2])
        def code3():
            config[:,1,:,:,0,0,:,:,1,:],config[:,0,:,:,1,1,:,:,0,:] = self.q_HPP(config[:,1,:,:,0,0,:,:,1,:],config[:,0,:,:,1,1,:,:,0,:],coll_coef_list[2])
        def code4():
            config[1,:,:,:,0,:,:,0,1,:],config[0,:,:,:,1,:,:,1,0,:] = self.q_HPP(config[1,:,:,:,0,:,:,0,1,:],config[0,:,:,:,1,:,:,1,0,:],coll_coef_list[2])
        code_list = [code1, code2, code3, code4]
        random.shuffle(code_list)
        for code in code_list:
            code()
        def code5():
            config[0,:,:,1,1,:,0,:,:,:],config[1,:,:,0,0,:,1,:,:,:] = self.q_HPP(config[0,:,:,1,1,:,0,:,:,:],config[1,:,:,0,0,:,1,:,:,:],coll_coef_list[3])
        def code6():
            config[:,:,1,0,:,0,:,1,:,:],config[:,:,0,1,:,1,:,0,:,:] = self.q_HPP(config[:,:,1,0,:,0,:,1,:,:],config[:,:,0,1,:,1,:,0,:,:],coll_coef_list[3])
        def code7():
            config[:,1,0,:,0,:,1,:,:,:],config[:,0,1,:,1,:,0,:,:,:] = self.q_HPP(config[:,1,0,:,0,:,1,:,:,:],config[:,0,1,:,1,:,0,:,:,:],coll_coef_list[3])
        def code8():
            config[1,0,:,:,:,1,:,0,:,:],config[0,1,:,:,:,0,:,1,:,:] = self.q_HPP(config[1,0,:,:,:,1,:,0,:,:],config[0,1,:,:,:,0,:,1,:,:],coll_coef_list[3])
        code_list = [code5, code6, code7, code8]
        random.shuffle(code_list)
        for code in code_list:
            code()
        
        config[:,:,:,:,0,1,0,1,:,:],config[:,:,:,:,1,0,1,0,:,:] = self.q_HPP(config[:,:,:,:,0,1,0,1,:,:],config[:,:,:,:,1,0,1,0,:,:],coll_coef_list[0])
        # config[:,:,:,:,1,0,0,1,:,:],config[:,:,:,:,0,1,1,0,:,:] = self.q_HPP(config[:,:,:,:,1,0,0,1,:,:],config[:,:,:,:,0,1,1,0,:,:] ,coll_coef_list[0])
        
        config = config.flatten()
        return config
    
    def q_collision_variedensity(self):
        '''
        设置了以下碰撞规则，
        n=2, p=0: 010100000(两种) -- 100000000 -- 000001111 gamma
        n=2, p=2: 010000000  --  000001001 beta
        保证是TP-CP
        '''
        n = self.nx*self.ny

        self.gamma2_1 = 0.5
        self.gamma2_4 = 0.2
        self.gamma1_2 = 0.2
        self.gamma1_4 = 0.1
        self.gamma4_1 = 0.3
        self.gamma4_2 = 0.2
        # 提取 n=2, p=0 的状态
        t_list0 = [0b000001010,0b000010100,
                   0b000000001,0b111100000]
        g2 = np.zeros((len(t_list0),n))
        g2_temp = np.zeros((len(t_list0),n))
        for i in range(len(t_list0)):
            g2[i,:] = self.config[t_list0[i]*n:(t_list0[i]+1)*n]
        # get_portion(g2,self.config)
        # 把 n=2, p=2 的状态进行碰撞 
        g2_temp[0,:] = g2[0,:]*(1-self.gamma2_1-self.gamma2_4)+g2[2,:]*self.gamma1_2/2+g2[3,:]*self.gamma4_2/2
        g2_temp[1,:] = g2[1,:]*(1-self.gamma2_1-self.gamma2_4)+g2[2,:]*self.gamma1_2/2+g2[3,:]*self.gamma4_2/2
        g2_temp[2,:] = g2[2,:]*(1-self.gamma1_2-self.gamma1_4)+g2[0,:]*self.gamma2_1+g2[1,:]*self.gamma2_1+g2[3,:]*self.gamma4_1
        g2_temp[3,:] = g2[3,:]*(1-self.gamma4_1-self.gamma4_2)+g2[0,:]*self.gamma2_4+g2[1,:]*self.gamma2_4+g2[2,:]*self.gamma1_4
        for i in range(len(t_list0)):
            self.config[t_list0[i]*n:(t_list0[i]+1)*n] = g2_temp[i,:]

        self.beta1_2 = 1 #1变成2
        self.beta2_1 = 1 #2变成1
        # 提取 n=2, p=2 的状态
        t_list2_2 = [0b110000000,0b011000000,0b001100000,0b100100000]
        b2 = np.zeros((len(t_list2_2),n))
        for i in range(len(t_list2_2)):
            b2[i,:] = self.config[t_list2_2[i]*n:(t_list2_2[i]+1)*n]
        # get_portion(b2,self.config)
        # 提取 n=2, p=2 的状态
        t_list2_1 = [0b000010000,0b000001000,0b000000100,0b000000010]
        b1 = np.zeros((len(t_list2_1),n))
        for i in range(len(t_list2_1)):
            b1[i,:] = self.config[t_list2_1[i]*n:(t_list2_1[i]+1)*n]
        # get_portion(b1,self.config)
        # 把 n=2, p=2 的状态进行碰撞 
        b2_temp = b2*(1-self.beta2_1)+b1*self.beta1_2
        b1_temp = b1*(1-self.beta1_2)+b2*self.beta2_1
        for i in range(len(t_list2_2)):
            self.config[t_list2_2[i]*n:(t_list2_2[i]+1)*n] = b2_temp[i,:]
        for i in range(len(t_list2_1)):
            self.config[t_list2_1[i]*n:(t_list2_1[i]+1)*n] = b1_temp[i,:]

    @staticmethod
    def controlled_addminus(qc,lowerindex,higherindex,ctrl_str=None,ctrl_qubit=None,type=1):
        '''
        将1设置为加法；0设置为减法
        '''
        # lowerindex=0; higherindex=5; ctrl_str='00011'
        from qiskit.circuit.library import MCXGate
        ctrl_qubit.reverse()
        if lowerindex>higherindex:
            print("Qubit-transportation error.")
            sys.exit(1)
        for iiii in range(higherindex,lowerindex-1,-1):
            qubitlist = (iiii-lowerindex)*[type] # list 
            c_state=ctrl_str+''.join(map(str, qubitlist))
            ctrl_list = [int(char) for char in c_state]
            mcxgate = MCXGate(len(ctrl_list),ctrl_state=c_state)
            qc.append(mcxgate,list(range(lowerindex,iiii))+ctrl_qubit+[iiii])
        
    def quantum_transport(self, qc, step, in_anc, m_th, index=0):
        # 参数环节
        qx = int(np.log2(self.nx)); qy = int(np.log2(self.ny))
        in_anc = qx+qy+9
        if ((step&(step-1)==0) and step!=0):
            print(f"Step={step}")
        else:
            print("Step error")
            sys.exit(1)
        step = int(np.log2(step))
        if index==0:
            t_list=range(512)
        # 循环部分
        for ii in t_list:
            spin_index = format(ii, '09b') #自动成为字符串格式
            m = spin_index.count('1')
            if m>m_th or m==0:
                continue
            # if m != 2:
            #     continue
            bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1, bit0 = (int(spin_index[i]) for i in range(9)) # 分别是有速度的8个方向和没速度的一个方向
            bitlist = [bit8, bit7, bit6, bit5, bit4, bit3, bit2, bit1, bit0]; bitlist.reverse()
            # print(bitlist)

            qc.reset(in_anc); qc.reset(in_anc+1); qc.reset(in_anc+2); 
            E = [1/np.sqrt(m) if i < m else 0 for i in range(8)]
            qc.initialize(E,[in_anc,in_anc+1,in_anc+2])

            # e1 +x
            iiii = 1
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step,qx-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=1)
            iiii = 2
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step+qx,qx+qy-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=1)
            iiii = 3
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step,qx-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=0)
            iiii = 4
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step+qx,qx+qy-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=0)
            iiii = 5
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step,qx-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=1)
                self.controlled_addminus(qc,step+qx,qx+qy-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=1)
            iiii = 6
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step,qx-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=0)
                self.controlled_addminus(qc,step+qx,qx+qy-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=1)
            iiii = 7
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step,qx-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=0)
                self.controlled_addminus(qc,step+qx,qx+qy-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=0)
            iiii = 8
            if bitlist[iiii] != 0:
                E_index = sum(bitlist[:iiii])
                mix_state = format(E_index, '03b')
                c_state=mix_state+spin_index
                self.controlled_addminus(qc,step,qx-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=1)
                self.controlled_addminus(qc,step+qx,qx+qy-1,c_state,list(range(in_anc+3-1,in_anc-1,-1))+list(range(qx+qy+9-1,qx+qy-1,-1)),type=0)

    @staticmethod
    def quantum_collide_HPP(qc,ratio, colllist, anclist):
        '''
        来一个ratio的概率分配，即只有mix|1>的时候，进行这个电路
        colllisst 分别是
        '''
        if len(colllist) != 4 or len(anclist) != 2:
            print("input collide number error!")
        f1 = colllist[0]; f2 = colllist[1]; f3 = colllist[2]; f4 = colllist[3]
        ancilla = anclist[0]; mix = anclist[1] 
        from qiskit.circuit.library import MCXGate
        qc.initialize([1,0],ancilla)
        qc.initialize(np.sqrt([1-ratio,ratio]),mix)
        qc.ccx(mix,f1,f2)
        qc.ccx(mix,f3,f4)
        mcxgate = MCXGate(3,ctrl_state='111')
        qc.append(mcxgate,[mix,f2,f4,ancilla])
        qc.ccx(mix,ancilla,f1)
        qc.ccx(mix,ancilla,f3)
        qc.append(mcxgate,[mix,ancilla,f1,f3])
        qc.append(mcxgate,[mix,ancilla,f3,f1])
        qc.append(mcxgate,[mix,ancilla,f1,f3])
        qc.append(mcxgate,[mix,f2,f4,ancilla])
        qc.ccx(mix,f1,f2)
        qc.ccx(mix,f3,f4)

    def quantum_collide(self, qc, coll_coef_list, colllist, anclist):
        p1 = coll_coef_list[0] #正HPP，静止散射，斜碰撞的发生概率
        p2 = coll_coef_list[1] #斜HPP
        p3 = coll_coef_list[2] #静止散射
        p4 = coll_coef_list[3] #斜碰撞
        # HPP 
        self.quantum_collide_HPP(qc, p1, [colllist[i] for i in [1,2,3,4]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p2, [colllist[i] for i in [5,6,7,8]], [anclist[0],anclist[1]] )
        # 静止碰撞
        self.quantum_collide_HPP(qc, p3, [colllist[i] for i in [0,1,5,2]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p3, [colllist[i] for i in [0,2,6,3]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p3, [colllist[i] for i in [0,3,7,4]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p3, [colllist[i] for i in [0,4,8,1]], [anclist[0],anclist[1]] )
        # 斜碰撞
        self.quantum_collide_HPP(qc, p4, [colllist[i] for i in [2,4,5,8]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p4, [colllist[i] for i in [1,3,5,6]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p4, [colllist[i] for i in [2,4,6,7]], [anclist[0],anclist[1]] )
        self.quantum_collide_HPP(qc, p4, [colllist[i] for i in [1,3,7,8]], [anclist[0],anclist[1]] )
   
def densitymatrix_to_array(state):
    threshold = 1e-15  # You can adjust this threshold as needed
    diagonal_elements = np.diag(state.data)
    real_diagonal_elements = np.real(diagonal_elements)
    arr_copy = real_diagonal_elements.copy()
    arr_copy[np.abs(arr_copy) < threshold] = 0
    return(arr_copy)

def diag_to_config(diag,rho):
    return diag*np.sum(rho)

def check_array(A,lower=0,upper=1):
    '''
    lower and upper     f大于1是退出1，小于0是退出2
    '''
    import sys
    # print(f"max is {np.max(A)}, min is {np.min(A)}")
    if np.any(A > upper):
        print("Values beyond max.")
        sys.exit(1)
    if np.any(A < lower):
        positions = np.where(A < 0)
        values = A[positions]
        print("Positions of elements < 0:", positions)
        print("Values of elements < 0:", values)
        print("Values lower minimal.")
        sys.exit(2)

def get_portion(A,B):
    print(f"the portion is {np.sum(A)/np.sum(B)}")
        
def average_deviation(A,B):
    '''在没有深幅值的情况下，体现不出来区别'''
    difference = np.abs(A - B)
    aa = np.mean(np.abs(A))
    bb = np.mean(np.abs(B))
    cc = np.mean(difference)
    # print(f"A的幅值是{aa:.{10}f}")
    # print(f"B的幅值是{bb:.{10}f}")
    # print(f"差别是{cc:.{10}f}")
    print(f"相对误差是{cc/aa}")
    return(cc)

def get_enstrophy(w):
    '''return the average enstrophy'''
    temp = 0.5*w*w
    ave = np.mean(temp)
    return ave

def energy_spectrum(u,v):
    ny,nx = np.shape(u)
    kx = fftshift(np.fft.fftfreq(nx)) * nx
    ky = fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    k_max = min(nx, ny) // 2

    u_fft = fftshift(fft2(u))/(ny*nx)
    v_fft = fftshift(fft2(v))/(ny*nx)
    energy_density = (np.abs(u_fft)**2+np.abs(v_fft)**2)/2
    spectrum = np.zeros(k_max)
    for k in range(k_max):
        mask = (k <= K) & (K < k+1)
        spectrum[k] = np.sum(energy_density[mask])
    k_range = np.arange(k_max)
    return k_range[1:], spectrum[1:]

def get_spectrum(vor):
    ny,nx = np.shape(vor)
    kx = fftshift(np.fft.fftfreq(nx)) * nx
    ky = fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    k_max = min(nx, ny) // 2
    u_fft = fftshift(fft2(vor))/(ny*nx)
    energy_density = np.abs(u_fft)**2
    spectrum = np.zeros(k_max)
    for k in range(k_max):
        mask = (k <= K) & (K < k+1)
        spectrum[k] = np.sum(energy_density[mask])
    k_range = np.arange(k_max)
    return k_range[1:], spectrum[1:]

def spectrum_2D(vor):
    k,spe = get_spectrum(vor)
    plt.figure()
    plt.loglog(k,spe, 'b-', label='Energy Spectrum')
    plt.xlabel('Wave number (k)')
    plt.ylabel('Energy')
    plt.title('2D Turbulence Energy Spectrum')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    

def f_HPP(A, i1, i2, i3, i4):
    total_indices = A.shape[0]
    all_indices = set(range(total_indices))
    other_indices = list(all_indices - {i1, i2, i3, i4})
    product1 = A[i1, :] * A[i2, :]
    for idx in other_indices:
        product1 *= (1 - A[idx, :])
    product2 = A[i3, :] * A[i4, :]
    for idx in other_indices:
        product2 *= (1 - A[idx, :])
    return product1, product2

def f_to_state(A): 
    '''
    输入三维的正常的f，然后成为(512,ny,nx)的长度。
    '''
    length,ny,nx = A.shape
    f_lattice = A.reshape(length*ny*nx)
    import itertools
    config = []
    combinations = list(itertools.product([0, 1], repeat=length))
    for combination in combinations:
        product = np.ones (nx*ny) 
        for i, bit in enumerate(combination):
            if bit == 0:
                product = product* (1 - f_lattice[(length-1-i)*nx*ny:(length-i)*nx*ny])
            else:
                product = product* f_lattice[(length-1-i)*nx*ny:(length-i)*nx*ny]
        config.append(product)
    config = np.concatenate(config)
    return config

def state_to_f(config,nx,ny,length):
    n = nx*ny
    assert len(config) == (2 ** length) * n, "Config length must be 2^length * n"
    f_list = []
    for bit in range(length):
        f = np.sum(config[i*n:(i+1)*n] for i in range(2 ** length) if i & (1 << bit))
        f_list.append(f)
    f_lattice = np.reshape(np.concatenate(f_list), (length, ny, nx))
    return f_lattice

def state_H(config1,nx,ny,length):
    '''这玩意是回归1/2'''
    f = state_to_f(config1,nx,ny,length)
    f_lattice = f.reshape(length*ny*nx)
    import itertools
    config = []
    combinations = list(itertools.product([0, 1], repeat=length))
    for combination in combinations:
        product = np.zeros (nx*ny) 
        for i, bit in enumerate(combination):
            if bit == 0:
                product = product + (1 - f_lattice[(length-1-i)*nx*ny:(length-i)*nx*ny])
            else:
                product = product + f_lattice[(length-1-i)*nx*ny:(length-i)*nx*ny]
        config.append(product)
    config = np.concatenate(config)
    config = config/length/2**(length-1)
    return config

def state_noise_uniform(config,p):
    '''p是去极化概率'''
    temp = np.ones_like(config)/np.size(config)
    # print(np.sum(temp))
    return config*(1-p)+temp*p

def state_noise_binomial(config,p,depth,size):
    '''p是去极化概率'''
    samples = np.random.binomial(depth, p,size=size)
    samples = np.clip(samples, 0, size/2)
    
    clipped_samples = samples/size*p
    return config*(1-p)+clipped_samples*p

def state_noise_Gauss(config,n,p,depth):
    '''p是去极化概率'''
    size = np.size(config)
    samples = np.random.normal(loc=p*depth, scale=p*np.sqrt(depth), size=size)
    clipped_samples = samples/size
    clipped_samples = np.clip(clipped_samples, 0, 1)
    clipped_samples = clipped_samples/np.sum(clipped_samples)*p*depth*n*n
    return config*(1-p*depth)+clipped_samples

def state_noise_Gauss_partial(config,n,p,depth):
    # size = np.size(config)
    # samples = np.random.normal(loc=p*depth, scale=p*np.sqrt(depth), size=size)
    # clipped_samples = samples/size
    # clipped_samples = np.clip(clipped_samples, 0, 1)
    # clipped_samples = clipped_samples/np.sum(clipped_samples)*p*depth*n*n
    # return config*(1-p*depth)+clipped_samples
    pass

def state_to_double(state,length,nx,ny):
    '''
    需要输入一个正常(2**length,ny,nx)长度的config，然后转化为(2**length,2**length,ny,nx)
    '''
    if state.size!=(2**length*nx*ny):
        print(f"config size error!")
        sys.exit(1)
    config = state.reshape(2**length,nx*ny)
    shape = (2**length, 2**length, nx*ny)
    double_config = np.zeros(shape, dtype=np.float32)
    for i in range(nx*ny):
        double_config[:,:,i] = np.outer(config[:,i],config[:,i])
    print(double_config.dtype)  # 应该输出：float32
    return double_config

def print_memory_usage():
    process = os.getpid()
    info = resource.getrusage(resource.RUSAGE_SELF)
    mem = info.ru_maxrss / 1024  # Convert from kilobytes to megabytes
    print(f"Process {process} consumes {mem:.2f} MB of memory.")