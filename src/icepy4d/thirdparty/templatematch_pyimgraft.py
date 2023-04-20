import numpy as np
import time

# import scipy.ndimage
from scipy import signal

from scipy.signal import medfilt2d
from scipy.interpolate import interp1d
import pyfftw
from matplotlib import pyplot as plt

pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"


class MatchResult:
    def __init__(self, pu, pv, du, dv, peakCorr, meanAbsCorr, method):
        self.pu = pu
        self.pv = pv
        self.du = du
        self.dv = dv
        self.peakCorr = peakCorr
        self.meanAbsCorr = meanAbsCorr
        self.snr = peakCorr / meanAbsCorr
        self.method = method

    def clean(self, maxstrain=0.1, minsnr=1.02):
        # assume that pu and pv are arranged in a regular grid.
        # TODO: input checking...
        resolution = np.abs(
            self.pu[1, 1] - self.pu[0, 0]
        )  # assume constant resolution in all directions...
        strain = (
            np.sqrt(
                (self.du - medfilt2d(self.du)) ** 2
                + (self.dv - medfilt2d(self.dv)) ** 2
            )
            / resolution
        )
        #        ix = np.logical_or(
        #            strain > maxstrain, self.snr < minsnr
        #        )  # skip nans to avoid warnings
        ix = np.logical_or(
            np.greater(strain, maxstrain, where=~np.isnan(strain)),
            np.less(self.snr, minsnr, where=~np.isnan(strain)),
        )
        ix = np.logical_or(ix, np.isnan(strain))
        self.du[ix] = np.nan
        self.dv[ix] = np.nan

    def plot(self, x=None, y=None, alpha=0.7):
        px = self.pu
        py = self.pv
        dx = 1.0
        dy = 1.0
        if x is not None:
            px = interp1d(np.arange(0, x.shape[0]), x, fill_value="extrapolate")(px)
            dx = float(x[1] - x[0])
        if y is not None:
            py = interp1d(np.arange(0, y.shape[0]), y, fill_value="extrapolate")(py)
            dy = float(y[1] - y[0])

        C = np.sqrt(self.du**2 + self.dv**2)

        plt.pcolormesh(get_corners(px), get_corners(py), C, alpha=alpha)
        plt.colorbar()
        plt.quiver(px, py, self.du * dx, self.dv * dy)

    # TODO add more methods to plot and clean results....


def get_corners(pu):
    # helper function to generate inputs for pcolormesh
    pu_extend = np.zeros((pu.shape[0] + 2, pu.shape[1] + 2))
    pu_extend[1:-1, 1:-1] = pu  # fill up with original values
    # fill in extra endpoints
    pu_extend[:, 0] = pu_extend[:, 1] + (pu_extend[:, 1] - pu_extend[:, 2])
    pu_extend[:, -1] = pu_extend[:, -2] + (pu_extend[:, -2] - pu_extend[:, -3])
    pu_extend[0, :] = pu_extend[1, :] + (pu_extend[1, :] - pu_extend[2, :])
    pu_extend[-1, :] = pu_extend[-2, :] + (pu_extend[-2, :] - pu_extend[-3, :])
    # calculate the corner points
    # return scipy.signal.convolve2d(pu_extend, np.ones((2, 2)/4), mode='valid')  # TODO: remove dependency
    return (
        pu_extend[0:-1, 0:-1]
        + pu_extend[0:-1, 1:]
        + pu_extend[1:, 0:-1]
        + pu_extend[1:, 1:]
    ) / 4.0


def templatematch(
    A,
    B,
    pu=None,
    pv=None,
    TemplateWidth=128,
    SearchWidth=128 + 16,
    Initialdu=0,
    Initialdv=0,
):
    """Feature tracking by template matching

    Usage : r = templatematch(A, B, TemplateWidth=64, SearchWidth=128)

    Notes : Currently only orientation correlation is implemented.

    Parameters
    ----------
    A, B : matrix_like
        Two images as 2d-matrices.
    pu, pv : array_like, optional
        Pixel coordinates in image A that you would like to find in image B (default is to drape a grid over A)
    TemplateWidth : int, optional
        pixel-size of the small templates being cut from image A.
    SearchWidth : int, optional
        pixel-size of the search region within image B.
    Initialdu, Initialdv : int, optional
        An initial guess of the displacement. The search window will be offset by this. (default = 0)

    Returns:
    ----------
        result : MatchResult

    """
    if not np.any(np.iscomplex(A)):  # always do Orientation correlation!
        A = forient(A)
        B = forient(B)

    if pu is None:
        pu = np.arange(
            SearchWidth / 2,
            A.shape[1] - SearchWidth / 2 + TemplateWidth / 2,
            TemplateWidth / 2,
        )
        pv = np.arange(
            SearchWidth / 2,
            A.shape[0] - SearchWidth / 2 + TemplateWidth / 2,
            TemplateWidth / 2,
        )
        pu, pv = np.meshgrid(pu, pv)

    du = np.full(pu.shape, np.nan)  # np.empty(pu.shape) * np.nan
    dv = np.full(pu.shape, np.nan)  #
    peakCorr = np.full(pu.shape, np.nan)  #
    meanAbsCorr = np.full(pu.shape, np.nan)  #
    if np.isscalar(Initialdu):
        Initialdu = np.zeros(pu.shape) + Initialdu
    if np.isscalar(Initialdv):
        Initialdv = np.zeros(pu.shape) + Initialdv

    if np.any(np.iscomplex(B)):
        B = np.conj(B)

    SearchHeight = (
        SearchWidth  # TODO: Clean-up. Dont support heights in python version.
    )
    TemplateHeight = TemplateWidth

    # PREPARE PYFFTW:
    # --------preallocate arrays for pyfftw ---------------
    AArot90 = pyfftw.empty_aligned(
        (TemplateWidth, TemplateHeight), dtype="complex64", order="C", n=None
    )
    BB = pyfftw.empty_aligned(
        (SearchHeight, SearchWidth), dtype="complex64", order="C", n=None
    )
    CC_sz = np.add(AArot90.shape, BB.shape) - 1
    CC = pyfftw.empty_aligned(CC_sz, dtype="complex64", order="C", n=None)
    # fT = pyfftw.empty_aligned(sz, dtype='complex64', order='C', n=None)
    # fB = pyfftw.empty_aligned(sz, dtype='complex64', order='C', n=None)
    #
    fft2AA = pyfftw.builders.fft2(
        AArot90, s=CC_sz, overwrite_input=True, auto_contiguous=True
    )
    fft2BB = pyfftw.builders.fft2(
        BB, s=CC_sz, overwrite_input=True, auto_contiguous=True
    )
    ifft2CC = pyfftw.builders.ifft2(
        CC, overwrite_input=True, auto_contiguous=True, avoid_copy=True
    )

    # precalculate how to interpret CC
    wkeep = np.subtract(BB.shape, AArot90.shape) / 2  # cut away edge effects
    C_center = (CC_sz - 1) / 2  # center
    C_rows = (
        (C_center[0] - wkeep[0]).astype("int"),
        (C_center[0] + wkeep[0]).astype("int"),
    )
    C_cols = (
        (C_center[1] - wkeep[1]).astype("int"),
        (C_center[1] + wkeep[1]).astype("int"),
    )
    C_uu = np.arange(-wkeep[1], wkeep[1] + 1)
    C_vv = np.arange(-wkeep[0], wkeep[0] + 1)
    # -----------------------------------------------------

    for ii, u in np.ndenumerate(pu):
        p = np.array([u, pv[ii]])

        initdu = Initialdu[ii]
        initdv = Initialdv[ii]
        # Actual pixel centre might differ from (pu, pv) because of rounding
        #
        Acenter = np.round(p) - (TemplateWidth / 2 % 1)
        Bcenter = np.round(p + np.array([initdu, initdv])) - (
            SearchWidth / 2 % 1
        )  # centre coordinate of search region

        # we should return coords that was actually used:
        pu[ii] = Acenter[0]
        pv[ii] = Acenter[1]
        initdu = Bcenter[0] - Acenter[0]  # actual offset
        initdv = Bcenter[1] - Acenter[1]
        if np.isnan(p[0] + p[1]):
            continue
        try:
            Brows = (Bcenter[1] + (-SearchHeight / 2, SearchHeight / 2)).astype(
                "int"
            )  # TODO: check "+1"
            Bcols = (Bcenter[0] + (-SearchWidth / 2, SearchWidth / 2)).astype("int")
            Arows = (Acenter[1] + (-TemplateHeight / 2, TemplateHeight / 2)).astype(
                "int"
            )
            Acols = (Acenter[0] + (-TemplateWidth / 2, TemplateWidth / 2)).astype("int")
            if Brows[0] < 0 or Arows[0] < 0 or Bcols[0] < 0 or Acols[0] < 0:
                continue
            if (
                Brows[1] >= B.shape[0]
                or Arows[1] >= A.shape[0]
                or Bcols[1] >= B.shape[1]
                or Acols[1] >= A.shape[1]
            ):
                continue  # handled by exception
            BB[:, :] = B[Brows[0] : Brows[1], Bcols[0] : Bcols[1]]
            AArot90[:, :] = np.rot90(A[Arows[0] : Arows[1], Acols[0] : Acols[1]], 2)
        except IndexError:
            continue

        # --------------- CCF ------------------
        fT = fft2AA(AArot90)
        fB = fft2BB(BB)
        fT[:] = np.multiply(fB, fT)
        CC = np.real(ifft2CC(fT))

        C = CC[C_rows[0] : C_rows[1], C_cols[0] : C_cols[1]]

        # --------------------------------------

        mix = np.unravel_index(np.argmax(C), C.shape)
        Cmax = C[mix[0], mix[1]]
        meanAbsCorr[ii] = np.mean(abs(C))
        edgedist = np.min([mix, np.subtract(C.shape, mix) - 1])
        if edgedist == 0:
            continue  # because we dont trust peak if at edge of domain.
        else:
            ww = np.amin((edgedist, 4))
            c = C[mix[0] - ww : mix[0] + ww + 1, mix[1] - ww : mix[1] + ww + 1]
            [uu, vv] = np.meshgrid(
                C_uu[mix[1] - ww : mix[1] + ww + 1], C_vv[mix[0] - ww : mix[0] + ww + 1]
            )

            # simple, fast, and excellent performance for landsat test images.
            c = c - np.mean(abs(c.ravel()))
            c[c < 0] = 0
            c = c / np.sum(c)
            mix = (np.sum(np.multiply(vv, c)), np.sum(np.multiply(uu, c)))
        du[ii] = mix[1] + initdu
        dv[ii] = mix[0] + initdv
        peakCorr[ii] = Cmax
    return MatchResult(pu, pv, du, dv, peakCorr, meanAbsCorr, method="OC")


def forient(img):
    f = np.array(
        [[1.0, 0.0, 0.0 + 1.0j], [0.0, 0.0, 0.0], [0.0 - 1.0j, 0.0, -1.0]], np.complex64
    )
    r = signal.convolve2d(img, f, mode="same")
    m = np.abs(r)
    m[m == 0] = 1
    r = np.divide(r, m)
    return r


def perftest(A, B, Twidths=np.arange(190, 210), Addwidths=np.arange(25, 38), N=100):
    pu = np.linspace(300 + A.shape[1], A.shape[1] - 300, N)
    pv = np.linspace(300 + A.shape[0], A.shape[0] - 300, N)
    (Twidths, Addwidths) = np.meshgrid(Twidths, Addwidths)
    Ctime = np.full(Twidths.shape, np.nan)
    A = forient(A)
    B = forient(B)
    for ii, Twidth in np.ndenumerate(Twidths):
        Addwidth = Addwidths[ii]
        time1 = time.time()
        r = templatematch(
            A, B, pu=pu, pv=pv, TemplateWidth=Twidth, SearchWidth=Twidth + Addwidth
        )
        time2 = time.time()
        Ctime[ii] = time2 - time1
        print(Ctime[ii])

    plt.pcolor(Twidths, Addwidths, np.log(Ctime))
    plt.xlabel("Templatewidth")
    plt.colorbar()
    return (Twidths, Addwidths, Ctime)


if __name__ == "__main__":
    pass

    # # test code
    # from geoimread import geoimread
    # import matplotlib.pyplot as plt  # noqa

    # # Read the data
    # fA = "https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20150708_20170407_01_T1/LC08_L1TP_023001_20150708_20170407_01_T1_B8.TIF"
    # fB = "https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/023/001/LC08_L1TP_023001_20160710_20170323_01_T1/LC08_L1TP_023001_20160710_20170323_01_T1_B8.TIF"

    # A = geoimread(
    #     fA, roi_x=[-30.19], roi_y=[81.245], roi_crs={"init": "EPSG:4326"}, buffer=10000
    # )
    # B = geoimread(
    #     fB, roi_x=[-30.19], roi_y=[81.245], roi_crs={"init": "EPSG:4326"}, buffer=10000
    # )

    # import time

    # #    from templatematch import templatematch  # noqa

    # time1 = time.time()
    # r = templatematch(A, B, TemplateWidth=128, SearchWidth=128 + 62)
    # time2 = time.time()
    # ##

    # from matplotlib import pyplot as plt  # noqa

    # ax = plt.axes()
    # A.plot.imshow(cmap="gray", add_colorbar=False)
    # ax.set_aspect("equal")
    # ax.autoscale(tight=True)

    # r.clean()
    # r.plot(x=A.x, y=A.y)

    # print("Time", (time2 - time1) * 1000.0)
    # # plt.hist(r.du.ravel())

    # print(np.nanmean(r.du.ravel()))
    # print(np.nanmean(r.dv.ravel()))
