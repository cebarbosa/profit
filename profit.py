#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 29/04/15 15:30

@author: Carlos Eduardo Barbosa

The PROfile FITing program in PYthon performs a structural decomposition of
surface brightness profiles into multiple components, such as bulge, disk,
bar and point sources. The program includes some commonly adopted function
for parametrization of such components, such as Sersic functions and
exponential profiles, but other functions can be easily adapted into the
program.

Version 1.0 (31/05/2016): Adaptation to standalone version.

"""
import os
import re
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import InterpolatedUnivariateSpline as ius

import cap_mpfit as mpfit

class Profit():
    """ Parser for the input file"""
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self._load_models()
        self._read_input_file()
        self._set_arrays()
        return

    def _read_input_file(self):
        """ Read input file, remove comments and blank lines and split
            into header and body. """
        with open(self.infile) as f:
            lines = [x.strip() for x in f.readlines()]
        #######################################################################
        # Dealing with blank lines and comments
        lines = [x for x in lines if x]
        lines = [x for x in lines if not x.startswith("#")]
        lines = [x.split("#")[0].strip() for x in lines]
        #######################################################################
        # Separating header and body
        header = [x for x in lines if re.match("[a-e]", x[0])]
        body = [x.lower() for x in lines[len(header):]]
        ######################################################################
        # Parsing header 
        self.header = dict([(x.split(")")[0], x.split(")")[1].strip())
                           for x in header])
        ######################################################################
        # Setting input table and reading data
        self.table = self.header["a"]
        self.sma, self.intens, self.err = np.loadtxt(self.table,
                                                     usecols=(0,1,2)).T
        ######################################################################
        self.psffunct = self.header["b"]
        self.psf = np.array([float(x) for x in self.header["c"].split()])
        if "c1" not in self.header.keys():
            self.psferr = 0.1 * self.psf
        else:
            self.psferr = np.array([float(x) for x in
                                    self.header["c1"].split()])
        # Reading weigths from the file.
        if "e" in self.header:
            if self.header["e"].lower() == "none":
                self.weights = np.ones_like(self.sma)
            elif self.header["e"].lower() == "error":
                self.weights = 1. / self.err
        # Default value for error
        else:
            self.header["e"] = "error"
            self.weights = 1. / self.err
        self.conv_box = float(self.header["d"])
        #####################################################################
        # Parsing the body of nput
        # Split body into components
        self.complist = []
        self.complist_comments = []
        ######################################################################
        # Finding all the different components in the input model
        ######################################################################
        idxs = [i for i,x in enumerate(body) if x.startswith("1)")]
        for i in idxs:
            line = body[i]
            c = line[2:].split("@")
            c = [x.strip() for x in c]
            if len(c) == 1:
                c.append("default")
            self.complist_comments.append(c[1])
            self.complist.append(c[0])
        test_functions = [x in self.models.keys() for x in self.complist]
        if not all(test_functions):
            raise "Function not found in {0}".format(self.complist)
        self.p0err = []
        self.parinfo = []
        self.idx = []
        self.pert = []
        for i, (idx, comp) in enumerate(zip(idxs, self.complist)):
            nparams = self.models[comp].npar
            self.pert += self.models[comp].pert
            compdict = dict([(int(x.split(")")[0]), x.split(")")[1].strip())
                           for x in body[idx:idx+nparams+1]])
            self.idx.append(range(idx - i, idx + nparams - i))
            parinfo = {}
            for k,j in enumerate(range(2, nparams + 2)):
                val = compdict[j].split("+/-")[0]
                valerr = compdict[j].split("+/-")[1]
                self.p0err.append(valerr)
                parinfo["limited"]=[1,1]
                lims = self.models[comp].limits[k][:]
                parinfo["limits"] = lims
                if len(val.split()) == 1:
                    parinfo["fixed"]= 0.
                    parinfo["value"]= np.minimum(np.maximum(float(val),
                                                            lims[0]), lims[1])
                else:
                    val, info = val.split()
                    parinfo["fixed"]=float(info)
                    parinfo["value"]= np.minimum(np.maximum(float(val),
                                                            lims[0]), lims[1])
                self.parinfo.append(copy.deepcopy(parinfo))
        self.p0 = np.zeros(len(self.parinfo)).astype(float)
        self.pfix = np.zeros_like(self.p0)
        for i in range(len(self.parinfo)):
            self.p0[i] = self.parinfo[i]["value"]
            self.pfix[i] = self.parinfo[i]["fixed"]
        return

    def _load_models(self):
        """ Load models"""
        self.models = dict([("exponential", exponential()),
                            ("brokenexp", brokenexp()),
                            ("sersic", sersic()),
                            ("moffat", moffat()),
                            ("ps", point_source())])
        return

    def _set_arrays(self):
        # Increase the model size in 10% to avoid border issues
        self.rr, self.dr, self.rsize = self._make_radial_grid(1.1 *
                                                              self.conv_box)
        self._make_psfs()
        return

    def _make_psfs(self):
        """ Produces PSF arrays"""
        self.psf2D = self.models[self.psffunct](self.rr, *self.psf) * \
                     self.dr * self.dr
        self.psf2D /= np.sum(self.psf2D)
        self.psf1D = ius(self.rr.diagonal()[self.rsize:],
                         self.psf2D.diagonal()[self.rsize:])
        self.fft_psf2D = np.fft.fft2(self.psf2D)
        return

    def _make_radial_grid(self, l, k=4):
        """ Make a 2D grid of data points with radial coordinates.

        This programs make a array for which each point gives the radial
        distance to the center. The output array has the size as a power of 2,
        and always assure that the step between points in the x/y direction is
        less than unity.

        ----------------
        Input Parameters
        ----------------
        l: int
            Size of the box to be used.

        k: int
            Defines the minimum order of points in the grid.

        -----------------
        Output Parameters
        -----------------
        array :
            The output array.
        float :
            radial step between sucessive itens of array
        """
        while 2**k < 2*l:
            k += 1
        r = int(2**float(k))
        dr = l / 2**float(k)
        x = np.arange(-2**k, 2**k) * dr
        xx, yy = np.meshgrid(x, x)
        rr = np.sqrt(xx**2 + yy**2)
        return rr, dr, r

    def _print_results(self):
        chi2 = np.sum(self.residue(self.pfit)[1]**2)
        aic = chi2 + 2 * len(self.pfit)
        print "Fitting process complete."
        print "Reduced chi-square: ", round(chi2, 2)
        print "AIC: ", round(aic/len(self.sma), 2)
        return

    def fit(self, silence=False):
        """ Wrapper to fit model.

        This program perform the minimization program using the
        Levenberg-Markquadt algorithm.

        ----------------
        Input Parameters
        ----------------
        self.model : function
            The model to be minimized.
        self.p0 : array/list
            The first guess required for the LM algorithm.
        self.sma : array
            Array indicating the radial vector.
        self.intens : array
            Array indicating the intensity vector
        self.err : array
            Array indicating the intensity vector error
        self.psf : list (a1, sig1, sig2)
            PSF parameter list for the PSF. Currently, it supports only a
            double Gaussian indicating with: a1: weight of the first gaussian;
            sig1: standard deviation of the first gaussian; sig2: standard
            deviation of the second Gaussian. The weigth of the second
            gaussian is set to 1 - a1.
        self.conv_box: int
            Size of the convolution box to be used.

        -----------------
        Output Parameters
        -----------------
        self.rr : array
            2D version of the radial array
        self.dr : float
            Radial shift between pixels.
        self.psf2D : array
            2D array of the PSF
        self.fft_psf2D : array
            Fourier transform of the PSF
        self.pfit : array
            Parameters of the best fit model.
        self.pcov : array
            Errors in the parameters of the best fit model.
        """
        print "Calculating best model..."
        m = mpfit.mpfit(self.residue, parinfo=self.parinfo, quiet=1, ftol=1e-4)
        self.perr = m.perror
        self.pfit = m.params
        self._print_results()
        self.write_output()
        return

    def residue(self, p, fjac=None):
        """ Returns the function to be minimized. """
        status = 0
        return (status, (self.profile(p) - self.intens) * self.weights / \
               (len(self.intens) - len(p)))

    def profile(self, p, sma=None):
        """Returns the profile of the input model for parameters p.  """
        k = 0
        if sma == None:
            sma = self.sma
        model2D_unc = np.zeros_like(self.rr)
        model1D_unc = np.zeros_like(sma)
        for idx, comp in zip(self.idx, self.complist):
            model2D_unc += self.models[comp](self.rr, *p[idx])
            model1D_unc += self.models[comp](sma, *p[idx])
        return self.psfconvolve(model1D_unc, model2D_unc)

    def psfconvolve(self, model1D_unc, model2D_unc):
        """ Returns a PSF convolved profile. """
        model2D = np.fft.ifft2(np.fft.fft2(model2D_unc) * self.fft_psf2D).real
        model2D = np.fft.fftshift(model2D)
        model1D = np.column_stack((self.rr.diagonal(), model2D.diagonal()))
        index = np.where(model1D[:,0] == np.min(model1D[:,0]))[0][0]
        x, y = model1D[index:].T
        s = ius(x, y)
        rbox = x[-1] / np.sqrt(2)
        profile = model1D_unc
        profile[np.where(self.sma<rbox)] = s(self.sma[np.where(self.sma<rbox)])
        return profile

    def write_output(self):
        """ Write output file. """
        text = ["# Input file for profit.py"]
        text.append("a) {0} # Input table".format(self.table))
        text.append("b) {0} # PSF type".format(self.psffunct))
        text.append("c) {0} # PSF parameters".format(
                                np.array2string(self.psf, precision=3)[1:-1]))
        text.append("c1) {0} # PSF parameters err".format(
                            np.array2string(self.psferr, precision=3)[1:-1]))
        text.append("d) {0} # Convolution box".format(self.conv_box))
        text.append("e) {0} # Weights for fitting".format(
                                                          self.header["e"]))
        self.pfit = self.pfit.astype(np.float64)
        self.perr = self.perr.astype(np.float64)
        for idx, comp, comment in zip(self.idx, self.complist, \
                                      self.complist_comments):
            text.append("1) {0} @ {1} # Component type".format(comp, comment))
            for j, i in enumerate(idx):
                text.append("{0}) {1:.7f} {2} +/- {3:.5f} # {4}".format(
                            j+2, self.pfit[i], self.pfix[i], self.perr[i],
                            self.models[comp].comments[j]))
            text.append("\n")
        with open(self.outfile, "w") as f:
            f.write("\n".join(text))
        return

    def plot(self, ax1=None, ax2=None):
        """ Plot. """
        # self.fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(4, 1)
        gs.update(left=0.15, right=0.95, bottom = 0.10, top=0.94, wspace=0.05)
        if ax1 == None:
            ax1 = plt.subplot(gs[:3, :])
        ax1.minorticks_on()
        r = np.linspace(self.sma.min(), self.sma.max(), 1000)
        self.subvalues = np.zeros((len(self.sma), len(self.complist)))
        dashes = dict([("disk", [9,3]), ("bulge", [8, 4, 2, 4]),
                  ("bar", [8, 4, 2, 4, 2, 4]),
                  ("nucleus", [8, 4, 2, 4, 2, 4, 2, 4]),
                  ("ring", [8, 4, 2, 4, 2, 4, 2, 4, 2, 4]),
                  ("lens", [2,2]), ("arm", [8,4,8,4,2,4,2,4])])
        colors = dict([("disk", "b"), ("bulge", "r"), ("bar", "g"),
                       ("nucleus", "y"), ("ring", "c"), ("lens", "m"),
                       ("nucleii", "y"), ("arm", "lightblue"),
                       ("arms", "lightblue"), ("psf", "y")])
        for i, (idx, comp, comm) in enumerate(zip(self.idx, self.complist, \
                                   self.complist_comments)):
            model2D_unc = self.models[comp](self.rr, *self.pfit[idx])
            model1D_unc = self.models[comp](self.sma, *self.pfit[idx])
            model1D = self.psfconvolve(model1D_unc, model2D_unc)
            label = self.complist_comments[i]
            c = colors[comm] if comm in colors.keys() else "k"
            dash = dashes[comm] if comm in dashes.keys() else [10,1]
            ax1.plot(self.sma, mag(model1D), c, label=label, lw=2.5,
                     dashes=dash)
            self.subvalues[:,i] = model1D
        self.total = self.subvalues.sum(axis=1)
        # plt.errorbar(self.sma, self.mu, yerr = self.muerr, c="k", fmt="o",
        #              ecolor="0.5", capsize=0, lw=1.5, markersize=3)
        ax1.plot(self.sma, mag(self.intens), "kx", ms=8, mew=2 )
        ax1.plot(self.sma, mag(self.total), ls="-", c="0.5", lw=3.5, alpha=0.5)
        y1, y2 = ax1.get_ylim()
        y2 = np.ceil(np.max(mag(self.intens)))
        y1 = np.floor(np.min(mag(self.intens))) - 1
        ax1.set_ylim(y2, y1)
        ax1.set_ylabel(r"$\mu$ (mag arcsec$^{-2}$)")
        ax1.legend(prop={'size':13}, handlelength=4.)
        if ax2 == None:
            ax2 = plt.subplot(gs[3, :])
        ax2.minorticks_on()
        # plt.errorbar(self.sma, - self.mu + mag(total), yerr = -self.muerr,
        #               c="k", fmt="x", ecolor="0.5", capsize=0)
        ax2.plot(self.sma, - mag(self.intens) + mag(self.total), "xk", ms=8, mew=2)
        ax2.axhline(y=0, ls="--", c="k")
        ax2.set_ylabel(r"$\Delta \mu$ ")
        plt.ylim(-0.5, 0.5)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax2.set_xlabel("Radius (arcsec)")
        ax1.tick_params('both', length=8, width=1, which='major')
        ax1.tick_params('both', length=4, width=1, which='minor')
        ax1.tick_params('both', length=8, width=1, which='major')
        ax1.tick_params('both', length=4, width=1, which='minor')
        # for ax in [ax1, ax1]:
        #     ax.set_xscale("log")
        ax1.xaxis.set_ticklabels([])
        return

###############################################################################
# Fitting models

class exponential():
    def __init__(self):
        self.npar = 2
        self.limits = [[10., 30.], [0.,500]]
        self.pert = [0.05, 0.1]
        self.comments = ["Disc central SB", "Disc scalelenght"]

    def __call__(self, r, mu0, h):
        return intens(mu0) * np.exp(-(r/h))

    def __str__(self, pars, perr):
        p_new = []
        offset = [0.02, 1, 1, 1, 1.]
        # offset = [0., 0, 0, 0, 0]
        for i, (p, pe) in enumerate(zip(pars, perr)):
            if np.isnan(p):
                p_new.append("...")
            else:
                p_new.append(mkstr(p, np.sqrt(pe**2+offset[i]**2)))
        p_new = []
        offset = [0.02, 1, 1, 1, 1.]
        # offset = [0., 0, 0, 0, 0]
        for i, (p, pe) in enumerate(zip(pars, perr)):
            if np.isnan(p):
                p_new.append("...")
            else:
                p_new.append(mkstr(p, np.sqrt(pe**2+offset[i]**2)))


class point_source():
    def __init__(self):
        self.npar = 1
        self.limits = [[5., 30.]]
        self.pert = [0.05]
        self.comments = ["Magnitude of the source"]

    def __call__(self, r, m):
        delta = np.zeros_like(r)
        delta[r==0] = intens(m)
        return delta


class brokenexp():
    def __init__(self):
        self.npar = 5
        self.limits = [[10., 30.], [0.,500], [0., 500],
                     [0., 500], [0., 1000.]]
        self.pert = [0.05, 0.1, 0.1, 0.1, 0.1]
        self.comments = ["Disc central SB", "Disc inner scalelenght",
                "Disc outer scalelenght", " Break radius", "Alpha parameter"]

    def __call__(self, r, mu0, h1, h2, rb, alpha):
        """" Returns broken exponential profile. """
        I0 = intens(mu0)
        expoent = (1./h1 - 1./h2) / alpha
        S = np.power(1. + np.exp(- alpha * rb), - expoent)
        base = 1. + np.exp(alpha * (r - rb))
        term = np.power(base, expoent)
        return (S * I0 * np.exp(- r / h1)) * term

class sersic():
    def __init__(self):
        self.npar = 3
        self.limits = [[10., 50.], [0, 500], [0.00, 20.]]
        self.pert = [0.05, 0.1, 0.02]
        self.comments = ["Surface brightness at effective radius",
                         "Effective radius", "Sersic index"]

    def __call__(self, r, mue, re, n):
        return intens(mue) * np.exp(- self.bn(n) * (np.power(r / re, 1./n)-1))

    def bn(self, n):
        """ Calculate term of Sersic model to use effective radius as reference.

            This function give the solution for the equation

            Gamma(2n) = gamma(2n, bn),

            where 'Gamma' is the complete Gamma function and 'gamma' is the
            incomplete gamma function, as a function of the superior integral
            limit bn. For n > 0.36, the results is given by the approximation
            of Ciotti and Berlin (1999), and for n < 0.36 the approximation
            is given by the polynomial of MacArthur, Courteau & Holtzman(2003).

        ----------------
        Input Parameters
        ----------------
        n : float
            Sersic index

        -----------------
        Output Parameters
        -----------------
        float :
            b(n)
        """
        if n > 0.36:
            bn_val = ( 2.* n - 1./3 + 4./(405 * n) + 46./ (25515 * n * n) +
                      131. / (1148175*n*n*n) + 2194697./ (30690717750.*
                                                          n * n * n * n))
        else:
            bn_val = 0.01945 - 0.8902 * n + 10.95 * n * n - 19.67 * n *n * n \
                    + 13.43 * n * n * n * n
        return bn_val

class moffat():
    def __init__(self):
        self.npar = 3
        self.limits = [[0, 40], [0., 500.], [0., 1000]]
        self.comments = ["Magnitude", "Alpha parameter", "Beta parameter"]
        self.pert = [0.01, 0.01, 0.01]

    def __call__(self, r, mu0, alpha, beta):
        """ Return Moffat function for radius r and parameters alpha and beta

        ----------------
        Input Parameters
        ----------------
        r : array or float
            Array with the radial values.
        mu0: float
            Multiplicative scaling for Moffat function. mu=1 returns a
            normalized PSF.
        alpha, beta : float
            Parameters of the Moffat function.

        -----------------
        Output Parameters
        -----------------
        array or float :
            The Moffat function values.
        """
        return intens(mu0) * (beta - 1) / (np.pi * alpha * alpha) * np.power(
               1. + (r * r / alpha / alpha), -beta)


###############################################################################
# Math functions
###############################################################################
def mag(intens):
    """Returns the magnitude of a given intensity or array of intensities. """
    return -2.5 * np.log10(np.abs(intens))

def intens(mag):
    """Returns the intensity of a given magnitude"""
    return np.power(10, -0.4 * mag)
###############################################################################

def bootstrap(infile, nsim=30, logfile="boot.txt"):
    """ Estimate errors in the fitting using bootstrap. """
    print "Calculating errors using bootstrap..."
    M = Profit(infile, "tmp")
    M.pfit = np.array(M.p0, dtype=float)
    sky_rms = M.err.min()
    yhat = M.profile(M.pfit)
    resid = mag(M.intens) - mag(yhat)
    pboot = np.zeros((nsim, len(M.pfit)))
    psf = M.psf
    psfpert = np.column_stack((np.zeros(nsim),
                        np.random.normal(0, M.psferr[1], nsim),
                        np.random.normal(0, M.psferr[2], nsim)))
    nsigs =  np.random.randn(nsim)
    parinfo = copy.copy(M.parinfo)
    for i in np.arange(nsim):
        print "Simulation {0}/{1}".format(i+1, nsim)
        for j in range(len(parinfo)):
            M.parinfo[j]["value"] = np.random.normal(parinfo[j]["value"],
                                    M.pert[j])
        residBoot = np.random.choice(resid)
        yboot = intens(mag(yhat) + residBoot) + \
                sky_rms * np.ones_like(yhat) * nsigs[i]
        M.psf = psf + psfpert[i]
        M._set_arrays()
        for j in range(len(M.p0)):
            M.parinfo[j]["fixed"] = 0
        def residue(p, fjac=None):
            status = 0
            return (status, (M.profile(p) - yboot) * M.weights / \
                   (len(M.intens) - len(p)))
        m = mpfit.mpfit(residue, parinfo=M.parinfo, quiet=1, ftol=0.1)
        pboot[i] = m.params
    print "Saving logfile..."
    with open(logfile, "w") as f:
        np.savetxt(f, pboot)
    print "Done!"
    return

def update_errors(infile, logfile):
    """ Update model with errors calculated with bootstrap. """
    # Updating output
    print "Updating model errors..."
    M = Profit(infile, infile)
    M.pfit = np.array(M.p0, dtype=float)
    data = np.loadtxt(logfile).T
    errs = np.zeros(len(data))
    for i, d in enumerate(data):
        errs[i] = np.nanstd(d)
    M.perr = errs
    M.write_output()
    print "Done!"
    return

if __name__ == "__main__":
    pass
