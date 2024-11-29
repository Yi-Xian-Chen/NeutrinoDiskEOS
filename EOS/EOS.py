import numpy as np
from scipy.integrate import quad

# Constants
me = 9.11e-28
c = 3.0e10
kb = 1.38e-16
hbar = 1.054e-27
mp = 1.67e-24
Q = 2.531
Kweak = 6.5e-4
eV = 1.602e-12
Msun = 2.0e33
G = 6.67e-8
arad = 7.56e-15

class NeutrinoDiskEOS:
    def __init__(self, tempprec, rhoprec, logTmin, logTmax, logrhomin, logrhomax, Ye0):

        """Initialize temperature, density and result arrays"""
        self.Ye0 = Ye0 #given Ye
        # establish temperature grid
        self.temprun = np.logspace(logTmin, logTmax, tempprec)
        self.thetarun = kb * self.temprun / (me * c**2)                 #dimensionless temperature
        #establish density grid
        self.rhorun = np.logspace(logrhomin, logrhomax, rhoprec)

        # Initialize result arrays
        self.temp2D, self.rho2D = np.meshgrid(self.temprun, self.rhorun, indexing="ij") #2D arrays for temperature and density

        # energy and pressure independent of Ye
        self.erad = np.zeros_like(self.temp2D) 
        self.Prad = np.zeros_like(self.temp2D)
        self.Pion = np.zeros_like(self.temp2D)
        self.eion = np.zeros_like(self.temp2D)

        # quantities for fixed/given Ye
        self.mu = np.zeros_like(self.temp2D)
        self.mukT = np.zeros_like(self.temp2D)
        self.nplus = np.zeros_like(self.temp2D)
        self.nminus = np.zeros_like(self.temp2D)
        self.ndotplus = np.zeros_like(self.temp2D)
        self.ndotminus = np.zeros_like(self.temp2D)
        self.edotplus = np.zeros_like(self.temp2D)
        self.edotminus = np.zeros_like(self.temp2D)
        self.Pplus = np.zeros_like(self.temp2D)
        self.Pminus = np.zeros_like(self.temp2D)
        self.Prad = np.zeros_like(self.temp2D)
        self.eplus = np.zeros_like(self.temp2D)
        self.eminus = np.zeros_like(self.temp2D)
        self.erad = np.zeros_like(self.temp2D)
        

        self.edotcool = np.zeros_like(self.temp2D)
        self.Ptot = np.zeros_like(self.temp2D)
        self.etot = np.zeros_like(self.temp2D)

        # quantities for equilibrium Ye(rho, T)
        self.YeEQ = np.zeros_like(self.temp2D)
        self.muEQ = np.zeros_like(self.temp2D)
        self.mukTEQ = np.zeros_like(self.temp2D)
        self.nplusEQ = np.zeros_like(self.temp2D)
        self.nminusEQ = np.zeros_like(self.temp2D)
        self.ndotplusEQ = np.zeros_like(self.temp2D)
        self.ndotminusEQ = np.zeros_like(self.temp2D)
        self.edotplusEQ = np.zeros_like(self.temp2D)
        self.edotminusEQ = np.zeros_like(self.temp2D)
        self.PplusEQ = np.zeros_like(self.temp2D)
        self.PminusEQ = np.zeros_like(self.temp2D)
        self.Pion = np.zeros_like(self.temp2D)
        self.Eion = np.zeros_like(self.temp2D)
        self.eplusEQ = np.zeros_like(self.temp2D)
        self.eminusEQ = np.zeros_like(self.temp2D)
        self.Xnuc = np.zeros_like(self.temp2D)

        self.edotcoolEQ = np.zeros_like(self.temp2D)
        self.PtotEQ = np.zeros_like(self.temp2D)
        self.etotEQ = np.zeros_like(self.temp2D)


    def integral(self, func, pmin=0):
        """Integration of Fermi-Dirac distribution. upper limit using 100 to approximate infinity"""
        return  quad(lambda x: func(x), pmin, 100)[0]


    def closest(self, array, value):
        """Find index of the closest value in array."""
        return np.abs(array - value).argmin()
    
    def run(self):

        #hard-coded establish Ye, chemical potential, and nucleon fraction arrays, used for simple root-finding
        
        # Ye array
        yemin = np.log(0.01)
        yemax = np.log(1.0)
        yeprec = 1000
        Yerun = np.exp(np.linspace(yemin, yemax, yeprec))

        # Chemical potential array
        muprec = 1000
        mumin = np.log(0.001)
        mumax = np.log(100.0)
        murun = np.exp(np.linspace(mumin, mumax, muprec))

        # Free nucleon fraction array
        Xfmin = np.log(1.0e-10)
        Xfmax = np.log(1.0)
        Xfprec = 1000
        Xnucrun = np.exp(np.linspace(Xfmin, Xfmax, Xfprec))

        for k, theta in enumerate(self.temprun):
            nplusrun = np.zeros(muprec)
            nminusrun = np.zeros(muprec)
            leptonrun = np.zeros(muprec)

            # Inner loop over chemical potentials for each temperature
            for i, mur in enumerate(murun):
                x = lambda p: np.sqrt(p**2 + 1)
                fplus = lambda p: 1.0 / (np.exp((x(p) + mur) /  self.thetarun[k]) + 1.0)
                fminus = lambda p: 1.0 / (np.exp((x(p) - mur) /  self.thetarun[k]) + 1.0)
                
                # Integrals
                nplusrun[i] = 1.78e30 * self.integral(lambda p: fplus(p) * p**2)
                nminusrun[i] = 1.78e30 * self.integral(lambda p: fminus(p) * p**2)
                leptonrun[i] = nminusrun[i] - nplusrun[i]

            # Loop over densities
            for j, rho in enumerate(self.rhorun):
                self.erad[k, j] = arad * (theta**4)
                self.Prad[k, j] = (arad / 3.0) * (theta**4)
                self.Pion[k, j] = rho * kb * theta / mp
                self.eion[k, j] = 1.5 * rho * kb * theta / mp

                # Pair capture rates for Ye grid
                pig = np.zeros(yeprec, dtype=int)
                Yeeqpig = np.zeros(yeprec)

                for f, Ye in enumerate(Yerun):
                    pig[f] = np.argmin(np.abs(leptonrun - rho * Ye / mp))
                    mupig = murun[pig[f]]

                    # Compute integrals for pair capture
                    fminus = lambda p: 1.0 / (np.exp((p + Q - mupig) /  self.thetarun[k]) + 1.0)
                    fplus = lambda p: 1.0 / (np.exp((p - Q + mupig) / self.thetarun[k]) + 1.0)

                    ndotminuspig = Kweak * self.Ye0 * self.integral(lambda p: fminus(p) * (p + Q)**2 *(1.0-(p+Q)**(-2.0))**(0.5)*p**(2.0))
                    ndotpluspig = Kweak * (1 - self.Ye0) * self.integral(lambda p: fplus(p) * (p - Q)**2*(1.0-(p-Q)**(-2.0))**(0.5)*p**(2.0), Q+1)

                    Yeeqpig[f] = ndotpluspig / (ndotpluspig + ndotminuspig)

                # Equilibrium and closest Ye calculations
                hogEQ = self.closest(Yeeqpig / Yerun, 1.0)
                hog0 = self.closest(self.Ye0 / Yerun, 1.0)

                self.YeEQ[k, j] = Yeeqpig[hogEQ]#Yerun[hogEQ]
                self.muEQ[k, j] = murun[pig[hogEQ]]
                self.nplusEQ[k, j] = nplusrun[pig[hogEQ]]
                self.nminusEQ[k, j] = nminusrun[pig[hogEQ]]
                self.nplus[k, j] = nplusrun[pig[hog0]]
                self.nminus[k, j] = nminusrun[pig[hog0]]
                # Calculate pair capture rates
                x = lambda p: p + Q
                fminus = lambda p: 1.0 / (np.exp((x(p) - self.muEQ[k, j]) / self.thetarun[k]) + 1.0)
                integrand_minus = lambda p: fminus(p) * (p + Q) ** 2 * (1.0 - (p + Q) ** (-2.0)) ** 0.5 * p ** 2
                self.ndotminusEQ[k, j] = Kweak * self.YeEQ[k, j] * self.integral(lambda p: integrand_minus(p))
                self.edotminusEQ[k, j] = (Kweak / mp) * self.YeEQ[k, j] * (me * c ** 2) * self.integral(lambda p: integrand_minus(p) * p)

                fminus = lambda p: 1.0 / (np.exp((x(p) - self.mu[k, j]) / self.thetarun[k]) + 1.0)
                integrand_minus = lambda p: fminus(p) * (p + Q) ** 2 * (1.0 - (p + Q) ** (-2.0)) ** 0.5 * p ** 2
                self.ndotminus[k, j] = Kweak * self.Ye0 * self.integral(lambda p:integrand_minus(p))
                self.edotminus[k, j] = (Kweak / mp) * self.Ye0 * (me * c ** 2) * self.integral(lambda p:integrand_minus(p) * p)

                # Calculate pair capture cooling rates
                x = lambda p: p - Q
                fplus = lambda p: 1.0 / (np.exp((x(p) + self.mu[k, j]) / self.thetarun[k]) + 1.0)
                integrand_plus = lambda p: fplus(p) * (p - Q) ** 2 * (1.0 - (p - Q) ** (-2.0)) ** 0.5 * p ** 2
                self.ndotplus[k, j] = Kweak * (1.0 - self.Ye0) * self.integral(lambda p:integrand_plus(p), Q+1)
                self.edotplus[k, j] = (Kweak / mp) * (1.0 - self.Ye0) * (me * c ** 2) * self.integral(lambda p:integrand_plus(p) * p, Q+1)

                fplus = lambda p: 1.0 / (np.exp((x(p) + self.muEQ[k, j]) / self.thetarun[k]) + 1.0)
                integrand_plus = lambda p: fplus(p) * (p - Q) ** 2 * (1.0 - (p - Q) ** (-2.0)) ** 0.5 * p ** 2
                self.ndotplusEQ[k, j] = Kweak * (1.0 - self.YeEQ[k, j]) * self.integral(lambda p:integrand_plus(p), Q+1)
                self.edotplusEQ[k, j] = (Kweak / mp) * (1.0 - self.YeEQ[k, j]) * (me * c ** 2) * self.integral(lambda p:integrand_plus(p) * p, Q+1)
                #in units of luminosity per density

                # Calculate electron/positron pressures
                x = lambda p:np.sqrt( p ** 2 + 1.0)
                fplus = lambda p: 1.0 / (np.exp((x(p) + self.mu[k, j]) / self.thetarun[k]) + 1.0)
                fminus = lambda p: 1.0 / (np.exp((x(p) - self.mu[k, j]) / self.thetarun[k]) + 1.0)
                self.Pplus[k, j] = 4.86e23 * self.integral(lambda p:fplus(p) * p ** 4 / x(p))
                self.Pminus[k, j] = 4.86e23 * self.integral(lambda p:fminus(p) * p ** 4 / x(p))

                fplus = lambda p: 1.0 / (np.exp((x(p) + self.muEQ[k, j]) / self.thetarun[k]) + 1.0)
                fminus = lambda p: 1.0 / (np.exp((x(p) - self.muEQ[k, j]) / self.thetarun[k]) + 1.0)
                self.PplusEQ[k, j] = 4.86e23 * self.integral(lambda p:fplus(p) * p ** 4 / x(p))
                self.PminusEQ[k, j] = 4.86e23 * self.integral(lambda p:fminus(p) * p ** 4 / x(p))

                # Calculate electron/positron energy densities
                fplus = lambda p: 1.0 / (np.exp((x(p) + self.mu[k, j]) / self.thetarun[k]) + 1.0)
                fminus = lambda p: 1.0 / (np.exp((x(p) - self.mu[k, j]) / self.thetarun[k]) + 1.0)
                self.eplus[k, j] = 4.86e23 * self.integral(lambda p:fplus(p) * p ** 2 / x(p))
                self.eminus[k, j] = 4.86e23 * self.integral(lambda p:fminus(p) * p ** 2 / x(p))

                fplus = lambda p: 1.0 / (np.exp((x(p) + self.muEQ[k, j]) / self.thetarun[k]) + 1.0)
                fminus = lambda p: 1.0 / (np.exp((x(p) - self.muEQ[k, j]) / self.thetarun[k]) + 1.0)
                self.eplusEQ[k, j] = 4.86e23 * self.integral(lambda p:fplus(p) * p ** 2 / x(p))
                self.eminusEQ[k, j] = 4.86e23 * self.integral(lambda p:fminus(p) * p ** 2 / x(p))

                # Additional calculations
                t10 = self.temprun[k] / 1.0e10
                rho10 = self.rhorun[j] / 1.0e10
                Left = ((1.57e4 * (1.0 - Xnucrun)) * (rho10 ** -3.0) * (t10 ** 4.5) * np.exp(-32.81 / t10)) ** 0.25
                frog = self.closest(Left / Xnucrun, 1.0)
                self.Xnuc[k, j] = Xnucrun[frog]

        self.edotcool = self.edotplus + self.edotminus
        self.Ptot = self.Pminus + self.Pplus + self.Prad + self.Pion*((1.0+3.*self.Xnuc)/4.)
        self.etot = self.eplus + self.eminus + self.erad + self.eion*((1.0+3.*self.Xnuc)/4.)

        self.edotcoolEQ = self.edotplusEQ + self.edotminusEQ
        self.PtotEQ = self.PminusEQ + self.PplusEQ + self.Prad + self.Pion*((1.0+3.*self.Xnuc)/4.)
        self.etotEQ = self.eplusEQ + self.eminusEQ + self.erad + self.eion*((1.0+3.*self.Xnuc)/4.)
