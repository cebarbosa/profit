#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 29/04/15 15:30

@author: Carlos Eduardo Barbosa

Remanufactoring of fitmodel.py

"""
import os
import re
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.stats import nanmedian


import mpfit
from totmag import mkstr
from fitmodel import sort_sample

class Profit():
    """ Parser for the input file"""
    def __init__(self, infile, outfile, intable=None):
        self.intable = intable
        self.input_file = infile
        self.models = dict([("exponential", exponential()),
                            ("brokenexp", brokenexp()),
                            ("sersic", sersic()),
                            ("moffat", moffat()),
                            ("ps", point_source())])
        self.outfile = outfile
        self._read_input_file()
        self._parse_input_header()
        self._parse_input_body()
        self._set_arrays()

    def _read_input_file(self):
        """ Read input file, remove comments and blank lines and split
            into header and body. """
        with open(self.input_file) as f:
            lines = [x.strip() for x in f.readlines()]
        # Remove blank lines
        lines = [x for x in lines if x]
        # Removing commented lines
        lines = [x for x in lines if not x.startswith("#")]
        # Removing comments
        lines = [x.split("#")[0].strip() for x in lines]
        self.header = [x for x in lines if re.match("[a-f]", x[0])]
        lb = len(self.header)
        self.body = [x.lower() for x in lines[lb:]]

    def _parse_input_header(self):
        self.headdict = dict([(x.split(")")[0], x.split(")")[1].strip())
                           for x in self.header])
        if self.intable == None:
            self.table = self.headdict["a"]
        else:
            self.table = self.intable
        self.psffunct = self.headdict["b"]
        if "f" not in self.headdict:
            self.headdict["f"] = "0 1 2 3 4"
        self.stop_codes = np.array([float(x) for x in
                                    self.headdict["f"].split()])
        try:
            data = np.loadtxt(self.table, usecols=np.arange(14))
        except:
            data = np.loadtxt(self.table, usecols=np.arange(13))
        ######################################################################
        # I do not keep the stop codes in all tables, so I use this part to fix
        if data.shape[1] == 14:
            self.stops = data[:,13]
            idx = np.in1d(self.stops, self.stop_codes)
            data = data[idx]
        ######################################################################
        data = data.T
        self.sma = data[0]
        self.intens = data[1]
        self.err = data[2]
        self.psf = np.array([float(x) for x in self.headdict["c"].split()])
        if "c1" not in self.headdict.keys():
            self.psferr = 0.1 * self.psf
        else:
            self.psferr = np.array([float(x) for x in
                                    self.headdict["c1"].split()])
        if "e" in self.headdict:
            if self.headdict["e"].lower() == "none":
                self.weights = np.ones_like(self.sma)
            elif self.headdict["e"].lower() == "error":
                self.weights = 1. / self.err
        # Default value for error
        else:
            self.headdict["e"] = "error"
            self.weights = 1. / self.err
        self.conv_box = float(self.headdict["d"])
        self.mu = mag(self.intens)
        return

    def _parse_input_body(self):
        """ Parser for the body of the input file"""
        # Split body into components
        self.complist = []
        self.complist_comments = []
        components = []
        ######################################################################
        # Finding all the different components in the input model
        ######################################################################
        idxs = [i for i,x in enumerate(self.body) if x.startswith("1)")]
        for i in idxs:
            line = self.body[i]
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
                           for x in self.body[idx:idx+nparams+1]])
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

    def _beep(self):
        os.system("paplay /usr/share/sounds/ubuntu/stereo/system-ready.ogg")

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
        if not silence:
            self._beep()
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
        text = ["# Input file for fitmodel.py"]
        text.append("a) {0} # Input table".format(self.table))
        text.append("b) {0} # PSF type".format(self.psffunct))
        text.append("c) {0} # PSF parameters".format(
                                np.array2string(self.psf, precision=3)[1:-1]))
        text.append("c1) {0} # PSF parameters err".format(
                            np.array2string(self.psferr, precision=3)[1:-1]))
        text.append("d) {0} # Convolution box".format(self.conv_box))
        text.append("e) {0} # Weights for fitting".format(
                                                          self.headdict["e"]))
        text.append("f) {0} # Ellipse stop codes\n".format(self.headdict["f"]))
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
                       ("arms", "lightblue")])
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
        ax1.plot(self.sma, self.mu, "kx", ms=8, mew=2 )
        ax1.plot(self.sma, mag(self.total), ls="-", c="0.5", lw=3.5, alpha=0.5)
        y1, y2 = ax1.get_ylim()
        y2 = np.ceil(np.max(self.mu))
        y1 = np.floor(np.min(self.mu)) - 1
        ax1.set_ylim(y2, y1)
        ax1.set_ylabel(r"$\mu$ (mag arcsec$^{-2}$)")
        ax1.legend(prop={'size':13}, handlelength=4.)
        if ax2 == None:
            ax2 = plt.subplot(gs[3, :])
        ax2.minorticks_on()
        # plt.errorbar(self.sma, - self.mu + mag(total), yerr = -self.muerr,
        #               c="k", fmt="x", ecolor="0.5", capsize=0)
        ax2.plot(self.sma, - self.mu + mag(self.total), "xk", ms=8, mew=2)
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
        alpha, beta : float
            Parameters of the Moffat function.
        r : array or float
            Array with the radial values.

        -----------------
        Output Parameters
        -----------------
        array or float :
            The Moffat function values.
        """
        # print 2 * alpha * np.sqrt(np.power(2., 1/beta)-1.)
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

def plot_all(sample = None):
    plt.ioff()
    home = os.path.join("/home/kadu/Dropbox/GHASP/decomp/")
    os.chdir(home)
    outfile = "sol_final.sbf"
    if sample == None:
        from fitmodel import sort_sample
        sample = os.listdir(".")
        sample = sort_sample(sample)
    sample = [x for x in sample if os.path.exists(os.path.join(home, x, outfile))]
    outdir = "/home/kadu/Dropbox/GHASP/decomp/figs/"
    obs = "ghasp"
    band = "Rc"
    output = os.path.join(outdir, "decomp_{0}_{1}.pdf".format(obs, band))
    pp = PdfPages(output)
    # sample = ["ugc9969"]
    tabout = "/home/kadu/Dropbox/GHASP/tables/decomp_tables"
    for gal in sample:
        print gal
        os.chdir(os.path.join(home, gal))
        M = Profit(outfile, outfile)
        M.pfit = np.array(M.p0).astype(float)
        M.perr = np.array(M.p0err).astype(float)
        # for i, (comp, comm) in enumerate(zip(M.complist, M.complist_comments)):
        #     if comp == "exponential" and comm == "default":
        #         M.complist_comments[i] = "disk"
        M.plot()
        # M.write_output()
        plt.annotate("{0}".format(gal.upper()),
                 (0.55, 0.86), size = 20, ha="center",
                 xycoords = "figure fraction")
        plt.savefig(os.path.join(outdir, "decomp_{0}_ohp.eps".format(gal)),
                    dpi=100)
        for i, (comp, comm) in enumerate(zip(M.complist, M.complist_comments)):
            if comp == "exponential" and comm == "default":
                M.complist_comments[i] = "disk"
        pp.savefig()
        tab = np.column_stack((M.sma, mag(M.intens),
                               np.abs(2.5 / np.log(10) * M.err / M.intens),
                               mag(M.total), M.subvalues))
        with open(os.path.join(tabout, "{0}.txt".format(gal)), "w") as f:
            f.write("# Decomposition table for galaxy {0}\n".format(
                gal.upper()))
            f.write("# (0) SMA (arcsec)\n")
            f.write("# (1) R-band surface brightness profile (mag / arcsec^2)\n")
            f.write("# (2) R-band surface brightness error (mag / arcsec^2)\n")
            f.write("# (3) R-band model (mag / arcsec^2)\n")
            for i, (func, comment) in enumerate(zip(M.complist, M.complist_comments)):
                f.write("# ({0}) {1} function for {2} (mag / arcsec^2)\n".format(i+4, func,
                                                                comment))
            np.savetxt(f, tab, fmt="%.5f")
        # plt.close(M.fig)
    pp.close()
#         plt.pause(0.001)
#         raw_input()
    return

def calc_errs(gals):
    home = os.path.join("/home/kadu/Dropbox/GHASP/decomp/")
    os.chdir(home)
    for ii, gal in enumerate(gals):
        print gal, ii+1, " of ", len(gals)
        nsim = 30
        home = os.path.join("/home/kadu/Dropbox/GHASP/decomp/", gal)
        os.chdir(home)
        infile = "sol_final.sbf"
        logfile = "bootstrap_final.txt"
        if not os.path.exists(infile):
            continue
        M = Profit(infile, outfile="sol_boot.sbf")
        M.pfit = np.array(M.p0, dtype=float)
        sky_rms = M.err.min()
        yhat = M.profile(M.pfit)
        resid = mag(M.intens) - mag(yhat)
        pboot = np.zeros((nsim, len(M.pfit)))
        psf = M.psf
        psfpert = np.column_stack((np.zeros(nsim),
                            np.random.normal(0, M.psferr[1], nsim),
                            np.random.normal(0, M.psferr[2], nsim)))
        parinfo = copy.copy(M.parinfo)
        nsigs =  np.random.randn(nsim)
        parinfo = copy.copy(M.parinfo)
        for i in np.arange(nsim):
            for j in range(len(parinfo)):
                M.parinfo[j]["value"] = np.random.normal(parinfo[j]["value"],
                                        M.pert[j])
            residBoot = np.random.permutation(resid)
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
        with open(logfile, "w") as f:
            np.savetxt(f, pboot)
        ######################################################################
        # In case the simulations are ready, just update
        if os.path.exists(logfile):
            M = Profit(infile, outfile=infile)
            M.pfit = np.array(M.p0, dtype=float)
            data = np.loadtxt(logfile).T
            errs = np.zeros(len(data))
            for i, d in enumerate(data):
                errs[i] = 1.48 * nanmedian(np.abs(d - nanmedian(d)))
                # errs[i] = d.std()
            M.perr = errs
            M.write_output()
        ######################################################################
        # M.perr = np.nanstd(pboot, axis=0)
        # M.write_output()
    return

def make_table():
    home = "/home/kadu/Dropbox/GHASP/decomp/"
    os.chdir(home)
    sample = os.listdir(".")
    sol = "sol_final.sbf"
    from fitmodel import sort_sample
    sample = sort_sample(sample)
    sample = [x for x in sample if os.path.exists(os.path.join(home, x,
                                                               sol))]
    tex, tsv = [], []
    for gal in sample:
        print gal
        os.chdir(os.path.join(home, gal))
        M = Profit(sol, sol)
        nlines = M.complist.count(max(set(M.complist), key=M.complist.count))
        M.p0 = np.array(M.p0, dtype=float)
        M.pfit = M.p0
        M.p0err = np.array(M.p0err, dtype=float)
        M.perr = M.p0err
        M.write_output()
        sersics, disks, pss, serr, derr, pserr = [], [], [], [], [], []
        comments, dtype = [], []
        for i, (comp, comm) in enumerate(zip(M.complist, M.complist_comments)):
            if comp == "sersic":
                sersics.append(M.p0[M.idx[i]].tolist())
                err_corr = np.array([0.05, 0.1, 0.05])
                err_corr = np.zeros_like(err_corr)
                err = np.sqrt(M.p0err[M.idx[i]]**2 +
                              err_corr**2)
                serr.append(err.tolist())
                comments.append(comm)
            elif comp == "exponential":
                disks.append(M.p0[M.idx[i]].tolist() +
                              [np.nan, np.nan, np.nan])
                err_corr = np.array([0.05, 0.1])
                err_corr = np.zeros_like(err_corr)
                err = np.sqrt(M.p0err[M.idx[i]]**2 + err_corr**2)
                derr.append(err.tolist() +
                             [np.nan, np.nan, np.nan])
                dtype.append("Type I")
            elif comp == "brokenexp":
                disks.append(M.p0[M.idx[i]].tolist())
                err_corr = np.array([0.05, 0.1, 0.1, 0.1, 0.1])
                err_corr = np.zeros_like(err_corr)
                err = np.sqrt(M.p0err[M.idx[i]]**2 + \
                              err_corr**2)
                derr.append(err.tolist())
                if M.p0[M.idx[i]][1] >=  M.p0[M.idx[i]][2]:
                    dtype.append("Type II")
                else:
                    dtype.append("Type III")
            elif comp == "ps":
                err_corr = np.array([0.1])
                err_corr = np.zeros_like(err_corr)
                pss.append(M.p0[M.idx[i]].tolist())
                err = np.sqrt(M.p0err[M.idx[i]]**2 + err_corr**2)
                pserr.append(err.tolist())
        if len(sersics) < nlines:
            ap = (nlines - len(sersics)) * [["..." for x in range(3)]]
            ap2 = (nlines - len(sersics)) * [["" for x in range(3)]]
            sersics += ap
            serr += ap
            comments += ap2
        if len(disks) < nlines:
            ap = (nlines - len(disks)) * [["..." for x in range(5)]]
            ap2 = (nlines - len(disks)) * ["" for x in range(5)]
            disks += ap
            derr += ap
            dtype += ap2
        if len(pss) < nlines:
            ap = (nlines - len(pss)) * [["..." for x in range(1)]]
            pss += ap
            pserr += ap
        galname = gal.replace("ugc", "UGC ").replace("ngc",
                            "NGC ").replace("ic", "IC ")
        gallist = [galname]
        if nlines > 1:
            gallist += (nlines - 1) * [""]
        for l in range(nlines):
            texline = []
            tsvline = []
            texline.append(gallist[l])
            tsvline.append(gallist[0])
            for i in range(3):
                texline.append(mkstr(sersics[l][i], serr[l][i]))
                tsvline.append(sersics[l][i])
                tsvline.append(serr[l][i])
            texline.append(comments[l])
            comm = comments[l] if isinstance(comments[l], str) else "..."
            tsvline.append(comm)
            for i in range(5):
                texline.append(mkstr(disks[l][i], derr[l][i]))
                tsvline.append(disks[l][i])
                tsvline.append(derr[l][i])
            texline.append(dtype[l])
            texline.append(mkstr(pss[l][0], pserr[l][0]))
            tsvline.append(dtype[l])
            tsvline.append(pss[l][0])
            tsvline.append(pserr[l][0])
            for k in range(len(tsvline)):
                if isinstance(tsvline[k], float):
                    tsvline[k] = "{0:.5f}".format(tsvline[k])
            tsv.append(" | ".join([str(x) for x in tsvline]))
            tex.append(" & ".join([str(x) for x in texline]))
    output = os.path.join("/home/kadu/Dropbox/GHASP/tables/decomp.tex")
    with open(output, "w") as f:
        f.write("\\\\\n".join(tex) + "\\\\")
    header = """##############################################################################
# Supplementary material for GHASP X paper
# Decomposition parameters for GHASP galaxies in the Rc band
# Observated at the OHP
# Table 4 in the paper
# Produced by Carlos Eduardo Barbosa
# Sao Paulo, Jan 10, 2015
##############################################################################
# (1) Galaxy name
# (2) Sersic function surface brightness at the effective radius [mag arcsec^-2]
# (3) Error for column 2
# (4) Effective radius of Sersic function [arcsec]
# (5) Error for column 4
# (6) Sersic index n
# (7) Error for column 6
# (8) Visual classification of the component
# (9) Central surface brightness of the disk [mag arcsec^-]
# (10) Error for column 9
# (11) (Inner) Disk scale lenght [arcsec]
# (12) Error for column 11
# (13) Outer disc scale lenght [arcsec]
# (14) Error for column 13
# (15) Break radius [arcsec]
# (16) Error for column 15
# (17) Smoothing parameter of the broken disk
# (18) Error for column 16
# (19) Classification of the disk according to breaks/ truncation
# (20) Nuclear Point source magnitude
# (21) Error for column 20
##############################################################################
"""
    with open(output.replace(".tex", ".tsv"), "w") as f:
        f.write(header)
        f.write("\n".join(tsv))
    return

def export_bulges_and_disks():
    decomp_dir = os.path.join(home, "decomp")
    os.chdir(decomp_dir)
    sol = "sol_final.sbf"
    sample = [x for x in os.listdir(".") if os.path.exists(
              os.path.join(home, "decomp", x, sol))]
    results = []
    for gal in sample[::-1]:
        os.chdir(os.path.join(home, "decomp", gal))
        M = Profit(sol, sol)
        print gal
        M.pfit = M.p0
        M.perr = np.array(M.p0err).astype(float)
        idx_bulge = [i for i,x in enumerate(M.complist_comments) if x=="bulge"]
        if len(idx_bulge) == 0:
            continue
        idx_disk = M.complist_comments.index("disk")
        pdisk = M.pfit[M.idx[idx_disk]]
        pdiskerr = M.perr[M.idx[idx_disk]]
        if len(pdisk) == 2:
            pdisk = np.hstack((pdisk, np.nan * np.zeros(3)))
            pdiskerr = np.hstack((pdiskerr, np.nan * np.zeros(3)))
        pd = np.empty(2 * pdisk.size, dtype=pdisk.dtype)
        pd[0::2] = np.round(pdisk, 5)
        pd[1::2] = np.round(pdiskerr, 5)
        for idx in idx_bulge:
            pbulge = M.pfit[M.idx[idx]]
            pbulgeerr = M.perr[M.idx[idx]]
            pb = np.empty(6)
            pb[0::2] = np.round(pbulge, 5)
            pb[1::2] = np.round(pbulgeerr, 5)
            results.append(np.hstack((gal, pb, pd)).astype(str))
    results = np.array(results)
    table = os.path.join(home, "tables", "bulge_disk_decomp.txt")
    with open(table, "w") as f:
        np.savetxt(f, results, fmt="%8s")

def single_work(gals, refit=False):
    plt.ion()
    for gal in gals:
        print gal
        sample = "ghasp"
        band = "Rc"
        home = os.path.join("/home/kadu/Dropbox/GHASP/decomp/", gal)
        os.chdir(home)
        infile = "sol_final.sbf"
        outfile = "sol_final_review.sbf"
        M = Profit(infile, outfile=outfile)
        M.pfit = np.array(M.p0)
        M.perr = np.array(M.p0err)
        os.system("gedit {0} &".format(infile))
        if refit:
            M.fit()
            os.system("gedit {0} &".format(outfile))
        M.plot()
        plt.pause(1)
        plt.show(block=True)
    return

def export_components(sample):
    tabout = "/home/kadu/Dropbox/GHASP/ghasp10/decomp"
    for gal in sample:
        home = os.path.join("/home/kadu/Dropbox/GHASP/decomp/", gal)
        os.chdir(home)
        infile = "sol_final.sbf"
        outfile = "sol_final_review.sbf"
        intable_dir = "/home/kadu/Dropbox/GHASP/ghasp10/sbprofiles/ohp_Rc"
        intable = os.path.join(intable_dir, "{0}_Rc_xyfix.dat".format(gal))
        M = Profit(infile, outfile=outfile, intable=intable)
        M.pfit = np.array(M.p0)
        M.perr = np.array(M.p0err)
        subvalues = np.zeros((len(M.sma), len(M.complist)))
        for i, (idx, comp, comm) in enumerate(zip(M.idx, M.complist, \
                                   M.complist_comments)):
            model2D_unc = M.models[comp](M.rr, *M.pfit[idx])
            model1D_unc = M.models[comp](M.sma, *M.pfit[idx])
            subvalues[:,i] = M.psfconvolve(model1D_unc, model2D_unc)
        total = subvalues.sum(axis=1)
        tab = np.column_stack((M.sma, mag(M.intens),
                               np.abs(2.5 / np.log(10) * M.err / M.intens),
                               mag(total), mag(subvalues)))
        with open(os.path.join(tabout, "{0}.txt".format(gal)), "w") as f:
            f.write("# Decomposition table for galaxy {0}\n".format(
                gal.upper()))
            f.write("# (0) SMA (arcsec)\n")
            f.write("# (1) R-band surface brightness profile (mag / arcsec^2)\n")
            f.write("# (2) R-band surface brightness error (mag / arcsec^2)\n")
            f.write("# (3) R-band model (mag / arcsec^2)\n")
            for i, (func, comment) in enumerate(zip(M.complist, M.complist_comments)):
                f.write("# ({0}) {1} function for {2} (mag / arcsec^2)\n".format(i+4, func,
                                                                comment))
            np.savetxt(f, tab, fmt="%.5f")

if __name__ == "__main__":
    home = "/home/kadu/Dropbox/GHASP/"
    os.chdir(home)
    sample = os.listdir(os.path.join(home, "decomp"))
    sample = sort_sample(sample)
    sample = [x for x in sample if os.path.exists(os.path.join(home,
              "decomp", x, "sol_final.sbf"))]
    export_components(sample)
    # print len(sample)
    # raw_input()
    # gal = "ugc12632"
    # single_work([gal], refit=0)
    # calc_errs(sample)
    # os.system("gedit {0} &".format(os.path.join(home, "decomp", gal,
    #                                             "sol_final.sbf")))
    # export_bulges_and_disks()
    # plot_all()
    # refit()
    # make_table()
