"""
Inspired from pyimgraft https://github.com/grinsted/pyimgraft
"""

import numpy as np
import pyfftw

from scipy import signal
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


def forient(img):
    f = np.array(
        [[1.0, 0.0, 0.0 + 1.0j], [0.0, 0.0, 0.0], [0.0 - 1.0j, 0.0, -1.0]], np.complex64
    )
    r = signal.convolve2d(img, f, mode="same")
    m = np.abs(r)
    m[m == 0] = 1
    r = np.divide(r, m)
    return r


class TemplateMatch:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xy: np.ndarray,
        method: str = "OC",
        template_width: int = 128,
        search_width: int = 128 + 16,
        initialdu: int = 0,
        initialdv: int = 0,
    ) -> None:
        """
        TemplateMatch: Feature tracking by template matching

        Args:
            A (np.ndarray): image A as 2D nunpy array
            B (np.ndarray): image B as 2D nunpy array
            xy (np.ndarray): Pixel coordinates in image A that you would like to find in image B
            method (str, optional): Correlation method. Defaults to "OC".
            TemplateWidth (int, optional): pixel-size of the small templates being cut from image A. Defaults to 128.
            SearchWidth (int, optional): pixel-size of the search region within image B. Defaults to 128+16.
            Initialdu (int, optional):  An initial guess of the displacement in x direction. The search window will be offset by this. Defaults to 0.
            Initialdv (int, optional): An initial guess of the displacement in y direction. The search window will be offset by this. Defaults to 0.
        """
        assert (
            len(A.shape) == 2 and len(B.shape) == 2
        ), "Invalid input images. Provide grayscale images."

        self.A = A
        self.B = B
        self.pu = xy[0]
        self.pv = xy[1]
        self.method = method
        self.template_width = template_width
        self.search_width = search_width
        self.initialdu = initialdu
        self.initialdv = initialdv

    def match(self) -> MatchResult:
        if self.method == "OC":
            self.result = self.OC(
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

    @staticmethod
    def OC(
        A,
        B,
        pu,
        pv,
        TemplateWidth=128,
        SearchWidth=128 + 16,
        Initialdu=0,
        Initialdv=0,
    ) -> MatchResult:
        """Feature tracking by template matching

        Usage : r = OC(A, B, TemplateWidth=64, SearchWidth=128)

        Notes : Currently only orientation correlation is implemented.

        Parameters
        ----------
        A, B : matrix_like
            Two images as 2d-matrices.
        pu, pv : array_like
            Pixel coordinates in image A that you would like to find in image B
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

        SearchHeight = SearchWidth
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

        p = np.array([pu, pv])

        initdu = Initialdu
        initdv = Initialdv
        # Actual pixel centre might differ from (pu, pv) because of rounding

        Acenter = np.round(p) - (TemplateWidth / 2 % 1)
        Bcenter = np.round(p + np.array([initdu, initdv])) - (
            SearchWidth / 2 % 1
        )  # centre coordinate of search region

        # we should return coords that was actually used:
        pu = Acenter[0]
        pv = Acenter[1]
        initdu = Bcenter[0] - Acenter[0]  # actual offset
        initdv = Bcenter[1] - Acenter[1]
        if np.isnan(p[0] + p[1]):
            return None
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
                return None
            if (
                Brows[1] >= B.shape[0]
                or Arows[1] >= A.shape[0]
                or Bcols[1] >= B.shape[1]
                or Acols[1] >= A.shape[1]
            ):
                print("Erorr: coordinates provided exceeds dimensions of the template.")
                return None  # handled by exception
            BB[:, :] = B[Brows[0] : Brows[1], Bcols[0] : Bcols[1]]
            AArot90[:, :] = np.rot90(A[Arows[0] : Arows[1], Acols[0] : Acols[1]], 2)
        except IndexError:
            return None  # because we dont trust peak if at edge of domain.

        # --------------- CCF ------------------
        fT = fft2AA(AArot90)
        fB = fft2BB(BB)
        fT[:] = np.multiply(fB, fT)
        CC = np.real(ifft2CC(fT))

        C = CC[C_rows[0] : C_rows[1], C_cols[0] : C_cols[1]]

        # --------------------------------------

        mix = np.unravel_index(np.argmax(C), C.shape)
        Cmax = C[mix[0], mix[1]]
        meanAbsCorr = np.mean(abs(C))
        edgedist = np.min([mix, np.subtract(C.shape, mix) - 1])
        if edgedist == 0:
            return None  # because we dont trust peak if at edge of domain.
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
        du = mix[1] + initdu
        dv = mix[0] + initdv
        peakCorr = Cmax

        return MatchResult(pu, pv, du, dv, peakCorr, meanAbsCorr, method="OC")


class Stats:
    def __init__(self, x: np.ndarray) -> None:
        """
        __init__ Compute statistics given an array of residuals

        Args:
            x (np.ndarray): nx2 array of residuals
        """
        self.x = x
        self.compute_stats()

    def compute_stats(self) -> None:
        """
        compute_stats compute basic statistics
        """
        # rmse_ = lambda x: np.sqrt((x[:, 0] ** 2).mean())
        # rmse = np.array(list(map(rmse_, self.x)))
        self.rmse = [
            np.sqrt((self.x[:, 0] ** 2).mean()),
            np.sqrt((self.x[:, 1] ** 2).mean()),
            np.sqrt((self.x.flatten() ** 2).mean()),
        ]
        self.mean = [self.x[:, 0].mean(), self.x[:, 1].mean(), self.x.flatten().mean()]
        self.std = [self.x[:, 0].std(), self.x[:, 1].std(), self.x.flatten().std()]
        self.min = [self.x[:, 0].min(), self.x[:, 1].min(), self.x.flatten().min()]
        self.max = [self.x[:, 0].max(), self.x[:, 1].max(), self.x.flatten().max()]

    def print_stats(self) -> None:
        """
        print_stats Print statistics on residuals
        """
        print("Stat:\tx\ty\tglobal")
        print(f"RMSE:\t{self.rmse[0]:.2f}\t{self.rmse[1]:.2f}\t{self.rmse[2]:.2f}")
        print(f"AVG:\t{self.mean[0]:.2f}\t{self.mean[1]:.2f}\t{self.mean[2]:.2f}")
        print(f"STD:\t{self.std[0]:.2f}\t{self.std[1]:.2f}\t{self.std[2]:.2f}")
        print(f"MIN:\t{self.min[0]:.2f}\t{self.min[1]:.2f}\t{self.min[2]:.2f}")
        print(f"MAX:\t{self.max[0]:.2f}\t{self.max[1]:.2f}\t{self.max[2]:.2f}")


def print_stats(x) -> None:
    """
    print_stats Print statistics on residuals

    Args:
        x (nx2 np.ndarray): 2xn array containing xy residuals
    """
    rmse = [
        np.sqrt((x[:, 0] ** 2).mean()),
        np.sqrt((x[:, 1] ** 2).mean()),
        np.sqrt((x.flatten() ** 2).mean()),
    ]

    print("Stat:\tx\ty\tglobal")
    print(f"RMSE:\t{rmse[0]:.2f}\t{rmse[1]:.2f}\t{rmse[2]:.2f}")
    print(f"AVG:\t{x[:, 0].mean():.2f}\t{x[:, 1].mean():.2f}\t{x.flatten().mean():.2f}")
    print(f"STD:\t{x[:, 0].std():.2f}\t{x[:, 1].std():.2f}\t{x.flatten().std():.2f}")
    print(f"MIN:\t{x[:, 0].min():.2f}\t{x[:, 1].min():.2f}\t{x.flatten().min():.2f}")
    print(f"MAX:\t{x[:, 0].max():.2f}\t{x[:, 1].max():.2f}\t{x.flatten().max():.2f}")


if __name__ == "__main__":

    # Test templateMatch class

    import matplotlib.pyplot as plt

    import time
    import cv2

    from copy import deepcopy
    from tqdm import tqdm

    from icepy4d.utils.initialization import parse_yaml_cfg, Inizialization

    cfg_file = "config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)
    init = Inizialization(cfg)
    init.inizialize_icepy()
    cams = init.cams
    images = init.images
    targets = init.targets

    cam_id = 0
    epoch = 0
    dt = 1
    roi_buffer = 128
    targets_to_use = ["F2"]  # , "F11"

    template_width = 16
    search_width = 64

    debug_viz = False
    debug = True

    t_est = {}
    diff = {}
    diff_noCC = {}

    for epoch in cfg.proc.epoch_to_process:  # tqdm(range(1, 3)):  #

        print(f"\tEpoch {epoch}... ", end=" ")

        t = targets[epoch].get_image_coor_by_label(targets_to_use, cam_id)[0].squeeze()

        t_int = np.round(t).astype(int)
        roi = [
            int(t_int[0]) - roi_buffer,
            int(t_int[1]) - roi_buffer,
            int(t_int[0]) + roi_buffer,
            int(t_int[1]) + roi_buffer,
        ]
        t_roi = np.array([t[0] - roi[0], t[1] - roi[1]])
        t_roi_int = np.round(t_roi).astype(int)

        A = images[cams[cam_id]].read_image(0).value[roi[1] : roi[3], roi[0] : roi[2]]
        B = (
            images[cams[cam_id]]
            .read_image(epoch)
            .value[roi[1] : roi[3], roi[0] : roi[2]]
        )

        #  Viz template on starting image
        if debug_viz:
            template_coor = [
                (t_roi_int[0] - template_width, t_roi_int[1] - template_width),
                (t_roi_int[0] + template_width, t_roi_int[1] + template_width),
            ]
            win_name = "template"
            img = cv2.cvtColor(deepcopy(A), cv2.COLOR_BGR2RGB)
            cv2.circle(img, (t_roi_int[0], t_roi_int[1]), 0, (0, 255, 0), -1)
            cv2.rectangle(img, template_coor[0], template_coor[1], (0, 255, 0), 1)
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        r = TemplateMatch.OC(
            cv2.cvtColor(A, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(B, cv2.COLOR_RGB2GRAY),
            np.array(t_roi[0]),
            np.array(t_roi[1]),
            TemplateWidth=template_width,
            SearchWidth=search_width,
        )

        t_est[epoch] = np.array([t[0] + r.du, t[1] + r.dv])
        # t_est[epoch] = np.array([r.pu + roi[0] + r.du, r.pv + roi[1] + r.dv])

        if debug:
            t_meas = targets[epoch].get_image_coor_by_label(targets_to_use, cam_id)[0][
                0
            ]
            diff[epoch] = t_meas - t_est[epoch]
            diff_noCC[epoch] = (
                t_meas
                - targets[0].get_image_coor_by_label(targets_to_use, cam_id)[0][0]
            )

            print(
                f"du: {r.du:.2f} dv: {r.dv:.2f} - diff {diff[epoch][0]:.2f} {diff[epoch][1]:.2f}- SNR {r.snr:.2f}"
            )

            # img = cv2.imread(str(images[cams[cam_id]].get_image_path(epoch)))
            # cv2.drawMarker(
            #     img,
            #     (
            #         np.round(t_est[epoch][0]).astype(int),
            #         np.round(t_est[epoch][1]).astype(int),
            #     ),
            #     (255, 0, 0),
            #     cv2.MARKER_CROSS,
            #     1,
            # )
            # cv2.imwrite("tmp/" + images[cams[cam_id]][epoch], img)
            # with Image.open(images[cams[cam_id]].get_image_path(epoch)) as im:
            #     draw = ImageDraw.Draw(im)
            #     draw.ellipse(list(np.concatenate((t_est[epoch],t_est[epoch]))), outline=(255,0,0), width=1)
            #     im.save('test.jpg', "JPEG")

        if debug_viz:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(A)
            ax[0].scatter(t_roi[0], t_roi[1], s=50, c="r", marker="+")
            ax[0].set_aspect("equal")
            ax[1].imshow(B)
            ax[1].scatter(t_roi[0] + r.du, t_roi[1] + r.dv, s=50, c="r", marker="+")
            ax[1].set_aspect("equal")

    diff = np.stack((diff.values()))
    nans = np.isnan(diff[:, 0])
    print(f"{sum(nans)} invalid features")
    stats = Stats(diff[~nans, :])
    stats.print_stats()

    # print("Without CC tracking:")
    # diff_noCC = np.stack((diff_noCC.values()))
    # stats = Stats(diff_noCC)
    # stats.print_stats()

    print("Done")
