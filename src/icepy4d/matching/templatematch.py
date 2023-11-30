"""
Inspired from pyimgraft https://github.com/grinsted/pyimgraft
"""
from typing import Tuple, List
from timeit import default_timer as timer

import numpy as np
import pyfftw
from matplotlib import pyplot as plt
from scipy import signal

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

class TemplateMatch:
    """
    TemplateMatch: Feature tracking by template matching

    Args:
        A (np.ndarray): image A as 2D numpy array
        B (np.ndarray): image B as 2D numpy array
        xy (np.ndarray): Pixel coordinates in image A that you would like to find in image B as 2D numpy array of shape n x 2 
        method (str, optional): Correlation method. Defaults to "OC".
        template_width (int, optional): Pixel-size of the small templates being cut from image A. Defaults to 128.
        search_width (int, optional): Pixel-size of the search region within image B. Defaults to 128 + 16.
        initialdu (int, optional): An initial guess of the displacement in x direction. The search window will be offset by this. Defaults to 0.
        initialdv (int, optional): An initial guess of the displacement in y direction. The search window will be offset by this. Defaults to 0.
        single_points (bool, optional): If True, apply template matching on single points in the xy array, otherwise define a regular grid based on the xy coordates (meshgrid) and track all. Defaults to False.


    TODO: if xy is not provided, define a grid of coordinates to search.
    TODO: Add the possibility to use a boolean mask to restrict the search in a portion of the images (and speed up computation).
    """

    available_methods = ["OC"]

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xy: np.ndarray = None,
        method: str = "OC",
        template_width: int = 128,
        search_width: int = 128 + 16,
        initialdu: float = 0,
        initialdv: float = 0,
        single_points: bool = False,
    ) -> None:
        """
        TemplateMatch: Feature tracking by template matching

        Args:
            A (np.ndarray): image A as 2D nunpy array
            B (np.ndarray): image B as 2D nunpy array
            xy (np.ndarray): Pixel coordinates in image A that you would like to find in image B as 2D nunpy array of shape n x 2 
            method (str, optional): Correlation method. Defaults to "OC".
            TemplateWidth (int, optional): pixel-size of the small templates being cut from image A. Defaults to 128.
            SearchWidth (int, optional): pixel-size of the search region within image B. Defaults to 128+16.
            Initialdu (float, optional):  An initial guess of the displacement in x direction. The search window will be offset by this. Defaults to 0.
            Initialdv (float, optional): An initial guess of the displacement in y direction. The search window will be offset by this. Defaults to 0.
        """
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise ValueError("Invalid input images. Provide grayscale images.")

        if xy.shape[1] != 2:
            raise ValueError("Invalid xy shape. Provide 2D array of shape n x 2.")
        
        if method not in self.available_methods:
            raise ValueError(f"Invalid method. Available methods: {self.available_methods}")

        self.A = A
        self.B = B
        self.method = method
        self.template_width = template_width
        self.search_width = search_width
        self.initialdu = initialdu
        self.initialdv = initialdv

        # Define meshgrid of coordinates to search
        pu, pv = self.define_grid(pu=xy[:,0], pv=xy[:,1])
        if single_points:
            def set_non_diagonal_nan(a):
                out = np.full_like(a, np.nan)
                di = np.diag_indices(a.shape[0])
                out[di] = a[di]
                return out

            pu = set_non_diagonal_nan(pu)
            pv = set_non_diagonal_nan(pv)

        self.pu = pu
        self.pv = pv


    def define_grid(
        self, pu: np.ndarray = None, pv: np.ndarray = None, step_x: int = None, step_y: int = None, mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define a grid of coordinates for search.

        Args:
            pu (np.ndarray, optional): User-provided x coordinates. Defaults to None.
            pv (np.ndarray, optional): User-provided y coordinates. Defaults to None.
            step_x (int, optional): Step size for x direction. Defaults to None.
            step_y (int, optional): Step size for y direction. Defaults to None.
            mask (np.ndarray, optional): Boolean mask to restrict the search area. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Meshgrid coordinates for search.
        """
        if pu is not None and pv is not None:
            return np.meshgrid(pu, pv)

        if step_x is None or step_y is None:
            raise ValueError("Provide step_x and step_y for automatic grid generation.")

        x_range = np.arange(self.search_width / 2, self.A.shape[1] - self.search_width / 2 + self.template_width / 2, step_x)
        y_range = np.arange(self.search_width / 2, self.A.shape[0] - self.search_width / 2 + self.template_width / 2, step_y)

        if mask is not None:
            x_range, y_range = np.meshgrid(x_range, y_range)
            mask = np.logical_and(mask, np.logical_and(x_range >= 0, y_range >= 0))
            x_range = x_range[mask]
            y_range = y_range[mask]

        return np.meshgrid(x_range, y_range)

    def match(self) -> MatchResult:
        """
        Perform template matching and return the result.

        Returns:
            MatchResult: An object containing the tracking results, including pu, pv, du, dv, peak correlation, mean absolute correlation, and the method used.
        """        
        if self.method == "OC":
            self.result = OC(
                self.A,
                self.B,
                self.pu,
                self.pv,
                self.template_width,
                self.search_width,
                self.initialdu,
                self.initialdv,
            )
        return self.result


def OC(
    A: np.ndarray,
    B: np.ndarray,
    pu: np.ndarray,
    pv: np.ndarray,
    TemplateWidth: int = 128,
    SearchWidth: int = 128 + 16,
    Initialdu: float = 0,
    Initialdv: float = 0,
) -> MatchResult:
    """Perform feature tracking by template matching using orientation correlation.

    Args:
        A (np.ndarray): First input image as a 2D matrix.
        B (np.ndarray): Second input image as a 2D matrix.
        pu (np.ndarray): Pixel coordinates in image A to be found in image B in a NxM array in meshgrid style.
        pv (np.ndarray): Pixel coordinates in image A to be found in image B in a NxM array in meshgrid style.
        TemplateWidth (int, optional): Pixel-size of the small templates cut from image A.
        SearchWidth (int, optional): Pixel-size of the search region within image B.
        Initialdu (float, optional): Initial guess for the horizontal displacement. The search window will be offset by this. (default = 0)
        Initialdv (float, optional): Initial guess for the vertical displacement. The search window will be offset by this. (default = 0)

    Returns:
        MatchResult: An object containing the tracking results, including pu, pv, du, dv, peak correlation, mean absolute correlation, and the method used.

    Usage:
        result = OC(A, B, TemplateWidth=64, SearchWidth=128)

    Notes:
        Currently, only orientation correlation is implemented.
    """

    if not np.any(np.iscomplex(A)):
        A = forient(A)
        B = forient(B)

    du = np.full(pu.shape, np.nan)
    dv = np.full(pu.shape, np.nan)
    peakCorr = np.full(pu.shape, np.nan)
    meanAbsCorr = np.full(pu.shape, np.nan)
    if np.isscalar(Initialdu):
        Initialdu = np.zeros(pu.shape) + Initialdu
    if np.isscalar(Initialdv):
        Initialdv = np.zeros(pu.shape) + Initialdv

    if np.any(np.iscomplex(B)):
        B = np.conj(B)


    SearchHeight = (
        SearchWidth
    )  # TODO: Clean-up. Dont support heights in python version.
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

        if np.isnan(u):
            continue

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
        if np.isnan(p[0]+p[1]):
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

def perftest(A,B,Twidths = np.arange(190,210), Addwidths = np.arange(25,38),N=100):
    pu = np.linspace(300+A.shape[1],A.shape[1]-300,N)
    pv = np.linspace(300+A.shape[0],A.shape[0]-300,N)
    (Twidths,Addwidths)=np.meshgrid(Twidths,Addwidths)
    Ctime = np.full(Twidths.shape, np.nan)       
    A=forient(A)
    B=forient(B)
    for ii, Twidth in np.ndenumerate(Twidths):
        Addwidth = Addwidths[ii]
        time1 = timer()
        r = OC(A, B, pu=pu, pv=pv, TemplateWidth=Twidth, SearchWidth=Twidth + Addwidth)
        time2 = timer()
        Ctime[ii] = time2-time1
        print(Ctime[ii])
        
    plt.pcolor(Twidths,Addwidths,np.log(Ctime))
    plt.xlabel('Templatewidth')
    plt.colorbar()
    return (Twidths,Addwidths,Ctime)
        


if __name__ == "__main__":
    pass