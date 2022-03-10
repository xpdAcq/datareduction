import dataclasses as dc
import pathlib
import subprocess
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import ipywidgets.widgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pyFAI
import statsmodels.nonparametric.smoothers_lowess as smoothers_lowess
import tqdm.notebook as tqdm
import xarray as xr
from databroker import catalog
from diffpy.pdfgetx import PDFConfig, PDFGetter, Transformation
from diffpy.pdfgetx.pdfconfig import PDFConfigError
from ipywidgets import interact
from pkg_resources import resource_filename
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from tifffile import TiffWriter

from datareduction import __version__
from datareduction.vend import mask_img, generate_binner

NI_DSPACING_TXT = resource_filename("datareduction", "data/Ni_dspacing.txt")
NI_CIF_FILE = resource_filename("datareduction", "data/Ni_cif_file.cif")


class MyPDFConfig(PDFConfig):
    """PDFConfig for the lowess smoothing.

    Attributes
    ----------
    qcutoff :
        The Q > qcutoff region will be LOWESS smoothed.
    lowessf :
        The frac parameter used in LOWESS smoothing. The larger it is, the smoother it will be.
    """

    def __init__(self):
        super(MyPDFConfig, self).__init__()
        self.qcutoff = 24.0
        self.lowessf = 0.04
        self.endzero = True
        self.dataformat = "QA"
        self.qmin = 0.
        self.qmaxinst = 24.0
        self.qmax = 22.0
        self.composition = "Ni"


def smooth(xin: np.ndarray, yin: np.ndarray, xcutoff: float, lowessf: float, endzero: bool) -> typing.Tuple[
    np.ndarray, np.ndarray]:
    """Smooth the input data in region x >= xcutoff using lowessf parameter. If endzero True, terminate the data to the last zero point."""
    xout, yout = xin.copy(), yin.copy()
    cutoff = np.searchsorted(xin, xcutoff) + 1
    if cutoff < xin.shape[0]:
        xout[cutoff:], yout[cutoff:] = smoothers_lowess.lowess(yin[cutoff:], xin[cutoff:], frac=lowessf).T
    if endzero:
        # first element with a different sign
        ind = np.argmin(np.sign(yout[-1] * yout[::-1]))
        xout, yout = xout[:xout.shape[0] - ind], yout[:yout.shape[0] - ind]
    return xout, yout


class LowessTransform(Transformation):
    """The transformation doing the LOWESS smoothing on F(Q)."""

    summary = "LOWESS smoothing"
    outputtype = "lsfq"
    xinlabel = ""
    yinlabel = ""
    xoutlabel = ""
    youtlabel = ""

    xin = None
    yin = None
    xout = None
    yout = None

    def __init__(self, config: MyPDFConfig):
        super(LowessTransform, self).__init__(config)

    def checkConfig(self):
        if not isinstance(self.config, MyPDFConfig):
            raise PDFConfigError("The config for LowessTransform must be LowessPDFConfig.")

    def transform(self):
        self.xout, self.yout = smooth(self.xin, self.yin, self.config.qcutoff, self.config.lowessf,
                                      self.config.endzero)


class MyPDFGetter(PDFGetter):
    """The PDFGetter with LOWESS smoothing of F(Q) included."""

    def __init__(self, config: MyPDFConfig):
        super(MyPDFGetter, self).__init__(config)
        self.transformations.insert(-1, LowessTransform(config))


@dataclass
class MaskConfig:
    edge: int = 30
    lower_thresh: float = 0.0
    upper_thresh: float = None
    alpha: float = 2.
    auto_type: str = "median"
    tmsk: np.ndarray = None


@dataclass
class IntegrationConfig:
    npt: int = 3001
    correctSolidAngle: bool = False
    dummy: float = 0.
    unit: str = "q_A^-1"
    safe: bool = False
    polarization_factor: float = 0.99
    method: typing.Tuple[str, str, str] = ("bbox", "csr", "cython")


@dataclass
class BackgroundConfig:
    scale: float = 1.


@dataclass
class LabelConfig:
    Q: str = "Q"
    F: str = "F"
    I: str = "I"
    r: str = "r"
    G: str = "G"
    A: str = "Å"
    QU: str = "Å$^{-1}$"
    IU: str = "A. U."
    rU: str = "Å"
    GU: str = "Å$^{-2}$"
    FU: str = "Å$^{-1}$"


@dataclass
class IOConfig:
    output_dir: str = r"./"
    fname_template: str = r"{sample_name}"
    data_keys: typing.List[str] = field(default_factory=lambda: ["sample_name", "composition_string"])
    dataset_file: str = r"./dataset.nc"
    io_format: str = "NETCDF4"
    io_engine: str = "netcdf4"


@dataclass
class CalibrationConfig:
    calibrant: str = NI_DSPACING_TXT
    detector: str = "Perkin"
    wavelength: typing.Optional[float] = None
    structure: str = NI_CIF_FILE
    xmin: typing.Optional[float] = None
    xmax: typing.Optional[float] = None
    xstep: typing.Optional[float] = None
    qdamp: float = 0.04
    qbroad: float = 0.02


@dataclass
class ReductionConfig:
    geometry: AzimuthalIntegrator = AzimuthalIntegrator()
    mask: MaskConfig = MaskConfig()
    integration: IntegrationConfig = IntegrationConfig()
    background: BackgroundConfig = BackgroundConfig()
    pdf: MyPDFConfig = MyPDFConfig()
    label: LabelConfig = LabelConfig()
    io: IOConfig = IOConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    verbose: int = 1


class ReductionCalculator:

    def __init__(self, config: ReductionConfig):
        self.config: ReductionConfig = config
        self.dataset: xr.Dataset = xr.Dataset()
        self.dark_dataset: xr.Dataset = xr.Dataset()
        self.bkg_dataset: xr.Dataset = xr.Dataset()
        self.bkg_dark_dataset: xr.Dataset = xr.Dataset()
        self.executor = ThreadPoolExecutor(max_workers=24)
        self.fit_dataset: xr.Dataset = xr.Dataset()
        self.calib_result: xr.Dataset = xr.Dataset()
        self.mypdfgetter = MyPDFGetter(self.config.pdf)

    def set_dataset(self, dataset: xr.Dataset) -> None:
        self.dataset = dataset
        return

    def set_bkg_dark_dataset(self, dataset: typing.Optional[xr.Dataset]) -> None:
        self.bkg_dark_dataset = dataset
        return

    def set_dark_dataset(self, dataset: typing.Optional[xr.Dataset]) -> None:
        self.dark_dataset = dataset
        return

    def set_bkg_dataset(self, dataset: typing.Optional[xr.Dataset], squeeze: bool = True) -> None:
        self.bkg_dataset = dataset.squeeze() if squeeze else dataset
        return

    def bkg_img_subtract(self, image_name: str, image_dims: typing.Sequence[str] = ("dim_1", "dim_2")) -> None:
        corrected = xr.apply_ufunc(
            np.subtract,
            self.dataset[image_name],
            self.config.background.scale * self.bkg_dataset[image_name].values,
            input_core_dims=[image_dims, image_dims],
            output_core_dims=[image_dims],
            dask="parallelized",
            output_dtypes=[np.float]
        )
        self.dataset = self.dataset.assign({image_name: corrected})
        return

    @staticmethod
    def _average(
            ds: xr.Dataset,
            image_name: str,
            along: typing.Sequence[str]
    ):
        n = len(along)
        averaged = xr.apply_ufunc(
            np.mean,
            ds[image_name],
            input_core_dims=[along],
            exclude_dims=set(along),
            kwargs={"axis": tuple(range(-n, 0))},
            dask="parallelized",
            output_dtypes=[np.float]
        )
        ds = ds.assign({image_name: averaged})
        return ds

    def average_dark(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("time", "dim_0")
    ):
        self.dark_dataset = self._average(
            self.dark_dataset,
            image_name,
            along
        )
        return

    def average_bkg_dark(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("time", "dim_0")
    ):
        self.bkg_dark_dataset = self._average(
            self.bkg_dark_dataset,
            image_name,
            along
        )
        return

    def average_bkg(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("time", "dim_0")
    ):
        self.bkg_dataset = self._average(
            self.bkg_dataset,
            image_name,
            along
        )

    def average(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("dim_0",)
    ):
        self.dataset = self._average(
            self.dataset,
            image_name,
            along
        )
        return

    @staticmethod
    def _dark_subtract(
            ds,
            dark_ds,
            image_name: str,
            image_dims: typing.Sequence[str]
    ):
        corrected = xr.apply_ufunc(
            np.subtract,
            ds[image_name],
            dark_ds[image_name].values,
            input_core_dims=[image_dims, image_dims],
            output_core_dims=[image_dims],
            dask="parallelized",
            output_dtypes=[np.float]
        )
        ds = ds.assign({image_name: corrected})
        return ds

    def dark_subtract(
            self,
            image_name: str,
            image_dims: typing.Sequence[str] = ("dim_1", "dim_2")
    ):
        self.dataset = self._dark_subtract(
            self.dataset,
            self.dark_dataset,
            image_name,
            image_dims
        )
        return

    def dark_subtract_bkg(
            self,
            image_name: str,
            image_dims: typing.Sequence[str] = ("dim_1", "dim_2")
    ):
        self.bkg_dataset = self._dark_subtract(
            self.bkg_dataset,
            self.bkg_dark_dataset,
            image_name,
            image_dims
        )

    def mask(
            self,
            image_name: str,
            image_dims: typing.Sequence[str] = ("dim_1", "dim_2")
    ) -> None:
        ds = self.dataset
        ai = self.config.geometry
        mc = self.config.mask
        kwargs = dc.asdict(mc)
        shape = [ds.dims[d] for d in image_dims]
        kwargs["binner"] = generate_binner(ai, shape)
        mask = xr.apply_ufunc(
            mask_img,
            ds[image_name],
            kwargs=kwargs,
            input_core_dims=[image_dims],
            output_core_dims=[image_dims],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float]
        )
        ds = ds.assign({image_name: ds[image_name] * mask})
        self.dataset = ds
        return

    def _integrate(
            self,
            ds: xr.Dataset,
            image_name: str,
            image_dims: typing.Tuple[str, str],
            chi_name: str,
            q_name: str
    ) -> xr.Dataset:
        ai = self.config.geometry
        exe = self.executor
        ic = self.config.integration
        images = ds[image_name]
        other_dims = tuple(set(images.dims) - set(image_dims))
        images.transpose(*other_dims, *image_dims)
        kwargs = dc.asdict(ic)
        if len(other_dims) > 0:
            res = np.asarray(
                list(exe.map(lambda img: ai.integrate1d(img.values, **kwargs), images))
            )
            q = res[0, 0, :]
            i = res[:, 1, :]
        else:
            res = ai.integrate1d(images_data, **kwargs)
            q = res[0]
            i = res[1]
        dims = other_dims + (q_name,)
        ds = ds.assign_coords({q_name: q})
        ds = ds.assign({chi_name: (dims, i)})
        return ds

    def integrate(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        """Integrate the image to I(Q)."""
        self.dataset = self._integrate(
            self.dataset,
            image_name,
            image_dims,
            chi_name,
            q_name
        )
        return

    def integrate_bkg(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        self.bkg_dataset = self._integrate(
            self.bkg_dataset,
            image_name,
            image_dims,
            chi_name,
            q_name
        )

    def mask_and_integrate(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            chi_name: str = "I",
            q_name: str = "Q"
    ) -> None:
        ds = self.dataset
        ai = self.config.geometry
        mc = self.config.mask
        ic = self.config.integration
        mask_kwargs = dc.asdict(mc)
        shape = [ds.sizes[d] for d in image_dims]
        mask_kwargs["binner"] = generate_binner(ai, shape)
        integ_kwargs = dc.asdict(ic)
        gen = self._mask_and_integrate(image_name, image_dims, chi_name, q_name, mask_kwargs, integ_kwargs)
        chi: xr.Dataset = xr.merge(gen)
        self.dataset = self.dataset.drop_dims(image_dims).update(chi)
        label = self.config.label
        self.dataset[chi_name].attrs = {"units": label.IU, "standard_name": label.I}
        self.dataset[q_name].attrs = {"units": label.QU, "standard_name": label.Q}
        self.dataset = self.dataset.compute()
        return

    def save(self):
        f = pathlib.PurePath(self.config.io.dataset_file)
        self.dataset.to_netcdf(str(f), engine=self.config.io.io_engine, format=self.config.io.io_format)
        return

    def load(self):
        f = pathlib.PurePath(self.config.io.dataset_file)
        self.dataset = xr.load_dataset(str(f), engine=self.config.io.io_engine)
        return

    def _mask_and_integrate(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str],
            chi_name: str,
            q_name: str,
            mask_kwargs: dict,
            integ_kwargs: dict
    ) -> typing.Generator[xr.Dataset, None, None]:
        ai = self.config.geometry
        for coords, image in self._gen_image(image_name, image_dims):
            mask = mask_img(image, **mask_kwargs)
            q, chi = ai.integrate1d(image, **integ_kwargs, mask=1 - mask)
            ds = xr.Dataset(
                {chi_name: ([q_name], chi)},
                {q_name: q}
            ).assign_coords(
                coords
            ).expand_dims(
                list(coords.keys())
            )
            yield ds
        return

    def _gen_image(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str]
    ) -> typing.Generator[typing.Tuple[dict, np.ndarray], None, None]:
        arr = self.dataset[image_name]
        other_dims = set(arr.dims) - set(image_dims)
        sizes = [arr.sizes[d] for d in other_dims]
        idxs = np.stack([np.ravel(i) for i in np.indices(sizes)]).transpose()
        gen = tqdm.tqdm(idxs, disable=(self.config.verbose <= 0), desc="Images", leave=False)
        for idx in gen:
            image = arr.isel(dict(zip(other_dims, idx))).compute()
            coords = {k: image.coords[k] for k in other_dims if k in image.coords}
            yield coords, image.data
        gen.close()
        return

    def bkg_subtract(
            self,
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        """Background subtraction."""
        scale = self.config.background.scale
        ds = self.dataset
        bkg_ds = self.bkg_dataset
        subtracted = xr.apply_ufunc(
            lambda x, y: np.subtract(x, scale * y),
            ds[chi_name],
            bkg_ds[chi_name],
            input_core_dims=[[q_name], [q_name]],
            output_core_dims=[[q_name]],
            dask="parallelized",
            output_dtypes=[np.float]
        )
        self.dataset = self.dataset.assign(
            {chi_name: subtracted}
        ).compute()
        return

    def img_bkg_subtract(
            self,
            image_name: str
    ):
        """Background subtraction on image level."""
        scale = self.config.background.scale
        ds = self.dataset
        bkg_ds = self.bkg_dataset
        attrs = ds[image_name].attrs
        subtracted = ds[image_name] - bkg_ds[image_name] * xr.DataArray(scale)
        subtracted.attrs = attrs
        self.dataset = self.dark_dataset.assign(
            {image_name: subtracted}
        )
        return

    def find_bkg_and_subtract(self, bkg_condition: xr.DataArray) -> None:
        """Find the background I(Q) in the dataset and subtract it from other I(Q)."""
        dataset = self.dataset.where(~bkg_condition, drop=True)
        bkg_dataset = self.dataset.where(bkg_condition, drop=True)
        self.set_dataset(dataset)
        self.set_bkg_dataset(bkg_dataset)
        return

    def get_G(
            self,
            chi_name: str = "I",
            q_name: str = "Q",
            g_name: str = "G",
            r_name: str = "r",
            c_name: str = "composition_string"
    ):
        """Transform the I(Q) to G(r)."""
        label = self.config.label
        ds = xr.merge(self._gen_gr(g_name, r_name, chi_name, q_name, c_name))
        if r_name in self.dataset:
            self.dataset = self.dataset.drop_dims(r_name)
        self.dataset.update(ds)
        self.dataset[r_name].attrs = {"units": label.rU, "standard_name": label.r}
        self.dataset[g_name].attrs = {"units": label.GU, "standard_name": label.G}
        return

    def _gen_gr(
            self,
            g_name: str,
            r_name: str,
            chi_name: str,
            q_name: str,
            c_name: str
    ) -> typing.Generator[xr.Dataset, None, None]:
        mpg = self.mypdfgetter
        for coords, ds in self._gen_data_along_q(chi_name, q_name):
            chi = ds[chi_name].values
            q = ds[q_name].values
            if c_name in ds:
                mpg.config.composition = ds[c_name].values[0]
            r, g = mpg.__call__(q, chi)
            yield xr.Dataset(
                {g_name: ([r_name], g)},
                {r_name: r}
            ).assign_coords(
                coords
            ).expand_dims(
                list(coords.keys())
            )
        return

    def _gen_data_along_q(
            self,
            chi_name: str,
            q_name: str
    ) -> typing.Generator[typing.Tuple[dict, xr.Dataset], None, None]:
        arr = self.dataset[chi_name]
        other_dims = set(arr.dims) - {q_name}
        sizes = [arr.sizes[d] for d in other_dims]
        idxs = np.stack([np.ravel(i) for i in np.indices(sizes)]).transpose()
        gen = tqdm.tqdm(idxs, disable=(self.config.verbose <= 0), desc="PDFs")
        for idx in gen:
            ds = self.dataset.isel(dict(zip(other_dims, idx)))
            coords = {k: ds.coords[k] for k in other_dims if k in ds.coords}
            yield coords, ds
        gen.close()
        return

    def interact_fq(
            self,
            index: int = 0,
            chi_name: str = "I",
            q_name: str = "Q",
            c_name: str = "composition_string"
    ):
        """Interactive plot of F(Q)."""
        i = self.dataset[chi_name][index]
        q = i[q_name]
        mpg = self.mypdfgetter
        config: MyPDFConfig = mpg.config
        config.composition = self.dataset[c_name].data[index]
        pdf_config = self.config.pdf
        label = self.config.label

        def func(
                rpoly,
                qmin,
                qmax,
                qmaxinst,
                lowessf,
                qcutoff,
                endzero
        ):
            config.rpoly = rpoly
            config.qmin = qmin
            config.qmax = qmax
            config.qmaxinst = qmaxinst
            config.lowessf = lowessf
            config.qcutoff = qcutoff
            config.endzero = endzero
            _, g = mpg.__call__(q, i)
            q1, f1 = mpg.t[-3].xout, mpg.t[-3].yout
            q2, f2 = mpg.t[-2].xout, mpg.t[-2].yout
            ax: plt.Axes = plt.subplots()[1]
            ax.plot(q1, f1)
            ax.plot(q2, f2)
            ax.set_xlabel("{} [{}]".format(label.Q, label.QU))
            ax.set_ylabel("{} [{}]".format(label.F, label.FU))
            plt.pause(0.1)

        qlim = np.max(q.values)
        interact(
            func,
            rpoly=widgets.FloatSlider(pdf_config.rpoly, min=0., max=5., step=0.05),
            qmin=widgets.FloatSlider(pdf_config.qmin, min=0., max=5., step=0.05),
            qmax=widgets.FloatSlider(pdf_config.qmax, min=0., max=qlim, step=0.1),
            qmaxinst=widgets.FloatSlider(pdf_config.qmaxinst, min=0.0, max=qlim, step=0.1),
            lowessf=widgets.FloatSlider(pdf_config.lowessf, min=0.0, max=0.2, step=0.01),
            qcutoff=widgets.FloatSlider(pdf_config.qcutoff, min=0.0, max=qlim, step=0.1),
            endzero=widgets.Checkbox(pdf_config.endzero)
        )
        return

    def get_I(
            self,
            image_name: str,
            chi_name: str = "I",
            q_name: str = "Q",
            avg_along: typing.Sequence[str] = ("dim_0",),
            dark_avg_along: typing.Sequence[str] = ("time", "dim_0"),
            bkg_avg_along: typing.Sequence[str] = ("time", "dim_0"),
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            process_image: bool = True,
            drop_image: bool = True
    ):
        """Dark subtraction and integrate the dark subtracted image to I(Q)."""
        if process_image:
            self.average(image_name, avg_along)
            if image_name in self.dark_dataset:
                self.average_dark(image_name, dark_avg_along)
                self.dark_subtract(image_name, image_dims)
            if image_name in self.bkg_dataset:
                self.average_bkg(image_name, bkg_avg_along)
                if image_name in self.bkg_dark_dataset:
                    self.average_bkg_dark(image_name, avg_along)
                    self.dark_subtract_bkg(image_name, image_dims)
        self.mask(image_name, image_dims)
        self.integrate(image_name, image_dims, chi_name, q_name)
        if process_image and image_name in self.bkg_dataset:
            self.integrate_bkg(image_name, image_dims, chi_name, q_name)
            self.bkg_subtract(chi_name)
        ds = self.dataset
        if drop_image:
            ds = ds.drop_vars(image_name)
        label = self.config.label
        ds[chi_name].attrs.update({"units": label.IU, "standard_name": label.I})
        ds[q_name].attrs.update({"units": label.QU, "standard_name": label.Q})
        self.dataset = ds.compute()
        return

    def reset_dims(
            self,
            dim2dims: typing.Dict[str, typing.List[str]],
    ):
        """Reset the dimension of the dataset."""
        self.dataset = self._reset_dims(self.dataset, dim2dims)
        return

    def export(self) -> xr.Dataset:
        """Return a copy of the dataset attribute with json serialized configuration in the metadata."""
        ds = self.dataset.copy()
        ds.attrs["time"] = time.time()
        ds.attrs["version"] = __version__
        return ds

    def write_files(self, x_name: str = "r", y_name: str = "G", suffix: str = "gr"):
        """Write the data in the file format of pdfgetx."""
        od = pathlib.Path(self.config.io.output_dir)
        od.mkdir(parents=True, exist_ok=True)
        ft = self.config.io.fname_template
        ds = self.dataset
        data_keys = self.config.io.data_keys
        mpg = MyPDFGetter(self.config.pdf)
        trns = mpg.getTransformation(suffix)
        trns.xout = ds[x_name].values
        for i, data in enumerate(self.dataset[y_name]):
            dct = {k: ds[k][i].item() for k in data_keys}
            filename = ft.format(**dct) + ".{}".format(suffix)
            filepath = od.joinpath(filename)
            trns.yout = data.values
            mpg.writeOutput(str(filepath), suffix)
        return

    @staticmethod
    def _reset_dims(
            ds: xr.Dataset,
            dim2dims: typing.Dict[str, typing.List[str]]
    ) -> xr.Dataset:
        # set new dims
        old_dims = list(dim2dims.keys())
        ds = ds.reset_index(old_dims)
        ds = ds.set_index(dim2dims)
        ds = ds.unstack()
        # rename new dims
        replaced_dims = {}
        for old_dim, new_dims in dim2dims.items():
            if isinstance(new_dims, str):
                replaced_dims[old_dim] = new_dims
            elif len(new_dims) == 1:
                replaced_dims[old_dim] = new_dims[0]
        if len(replaced_dims) > 0:
            ds = ds.rename_dims(replaced_dims)
            ds = ds.rename_vars(replaced_dims)
        # rename old dims (coords now)
        rule = {"{}_".format(old_dim): old_dim for old_dim in old_dims}
        ds = ds.rename_vars(rule)
        return ds

    def write_one_tiff(self, image_name: str) -> str:
        _output_dir = pathlib.Path(self.config.io.output_dir)
        tiff_file = _output_dir.joinpath("image_for_calib.tiff")
        tw = TiffWriter(str(tiff_file))
        data = self.dataset[image_name].values[0]
        tw.save(data)
        return str(tiff_file)

    def set_geometry_by_poni(self, poni_file: str) -> None:
        self.config.geometry = pyFAI.load(poni_file)
        return

    def run_calib(
            self,
            tiff_file: str
    ) -> None:
        import pdffitx.io as io
        wavelength = self.config.calibration.wavelength
        calibrant = self.config.calibration.calibrant
        detector = self.config.calibration.detector
        poni_file = str(pathlib.Path(self.config.io.output_dir).joinpath("calib.poni"))
        args = ["pyFAI-calib2", "--poni", poni_file]
        if wavelength:
            args.extend(["--wavelength", str(wavelength)])
        if calibrant:
            args.extend(["--calibrant", str(calibrant)])
        if detector:
            args.extend(["--detector", str(detector)])
        args.append(tiff_file)
        cp = subprocess.run(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode != 0:
            io.server_message("Error in Calibration. See below:")
            print(r"$", " ".join(args))
            print(cp.stdout.decode())
            print(cp.stderr.decode())
        return

    def get_poni_files(self) -> typing.List[str]:
        _output_dir = pathlib.Path(self.config.io.output_dir)
        lst = []
        for f in _output_dir.glob("*.poni"):
            lst.append(str(f))
        return lst

    def run_fit(self) -> None:
        from pdffitx.model import MultiPhaseModel
        ni = io.load_crystal(self.config.calibration.structure)
        mpm = MultiPhaseModel(equation="Standard", structures={"Standard": ni})
        g = mpm.get_generators().get("Standard")
        recipe = mpm.get_recipe()
        recipe.addVar(g.qdamp)
        recipe.addVar(g.qbroad)
        recipe.qdamp.setValue(self.config.calibration.qdamp)
        recipe.qbroad.setValue(self.config.calibration.qbroad)
        mpm.set_order("scale", "lat", ["adp", "delta"], ["qdamp", "qbroad"])
        mpm.set_verbose(0)
        results = mpm.fit_many_data(
            self.dataset,
            xmin=self.config.calibration.xmin,
            xmax=self.config.calibration.xmax,
            xstep=self.config.calibration.xstep,
            progress_bar=False,
            exclude_vars=["Q", "I"]
        )
        self.fit_dataset = xr.merge(results)
        return

    def clear_dark_dataset(self) -> None:
        self.set_dark_dataset(xr.Dataset())
        return

    def clear_bkg_dark_dataset(self) -> None:
        self.set_bkg_dark_dataset(xr.Dataset())
        return

    def clear_bkg_dataset(self) -> None:
        self.set_bkg_dataset(xr.Dataset())
        return

    def run_calib_and_fit(self, image_name: str) -> None:
        self.average(image_name)
        self.average_dark(image_name)
        self.dark_subtract(image_name)
        if image_name in self.bkg_dataset:
            self.average_bkg(image_name)
            self.dark_subtract_bkg(image_name)
            self.bkg_img_subtract(image_name)
        self.clear_dark_dataset()
        self.clear_bkg_dataset()
        tiff_file = self.write_one_tiff(image_name)
        self.run_calib(tiff_file)
        dss = []
        for f in tqdm.tqdm(self.get_poni_files()):
            self.set_geometry_by_poni(f)
            self.mask(image_name)
            self.integrate(image_name)
            self.get_G()
            self.run_fit()
            ds = self.fit_dataset.copy()
            w = float(self.config.geometry.wavelength) * 1e10
            ds = ds.assign_coords({"wavelength": (["time"], [w])}).swap_dims({"time": "wavelength"}).drop_vars(
                "time")
            dss.append(ds)
        self.calib_result = xr.merge(dss)
        self.calib_result = self.calib_result.sortby("wavelength")
        self.update_calib_result_attrs()
        return

    def update_calib_result_attrs(self) -> None:
        self.calib_result["rw"].attrs.update(
            {"standard_name": "$R_w$"}
        )
        self.calib_result["qdamp"].attrs.update(
            {"standard_name": "Q$_{damp}$", "units": "$\mathrm{\AA}^{-1}$"}
        )
        self.calib_result["qbroad"].attrs.update(
            {"standard_name": "Q$_{broad}$", "units": "$\mathrm{\AA}^{-1}$"}
        )
        self.calib_result["wavelength"].attrs.update(
            {"standard_name": "wavelength", "units": "$\mathrm{\AA}$"}
        )
        return


@dataclass
class DatabaseConfig:
    name: str = ""
    image_key: str = "pe1_image"
    metadata: dict = field(default_factory=lambda: {})
    calibration_md_key: str = "calibration_md"
    sc_dk_field_uid_key: str = "sc_dk_field_uid"
    bt_wavelength_key: str = "bt_wavelength"
    sample_data_keys: typing.List[str] = field(default_factory=lambda: ["sample_name"])


@dataclass
class DataProcessConfig:
    reduction: ReductionConfig = ReductionConfig()
    database: DatabaseConfig = DatabaseConfig()
    verbose: int = 1


class DataProcessor:

    def __init__(self, config: DataProcessConfig):
        self.config = config
        self.rc = ReductionCalculator(self.config.reduction)
        self.db = catalog[self.config.database.name]
        self._cached_start = dict()

    def load_data(self, uid: str) -> None:
        """Load data from the Bluesky run and dark subtract."""
        run = self.db[uid]
        start = dict(run.metadata["start"])
        start.update(self.config.database.metadata)
        self._cached_start = start
        calibration_md = start[self.config.database.calibration_md_key]
        sc_dk_field_uid = start[self.config.database.sc_dk_field_uid_key]
        dataset = run.primary.to_dask()
        dark_run = self.db[sc_dk_field_uid]
        dark_dataset = dark_run.primary.to_dask()
        self._load_ai_from_calib_result(calibration_md)
        self.rc.set_dataset(dataset)
        self.rc.set_dark_dataset(dark_dataset)
        self.rc.average(self.config.database.image_key)
        self.rc.average_dark(self.config.database.image_key)
        self.rc.dark_subtract(self.config.database.image_key)
        self.rc.clear_dark_dataset()
        return

    def load_bkg_data(self, uid: str) -> None:
        """Load background data from the bluesky run and dark subtract."""
        run = self.db[uid]
        start = dict(run.metadata["start"])
        start.update(self.config.database.metadata)
        sc_dk_field_uid = start[self.config.database.sc_dk_field_uid_key]
        dataset = run.primary.to_dask()
        dark_run = self.db[sc_dk_field_uid]
        dark_dataset = dark_run.primary.to_dask()
        self.rc.set_bkg_dataset(dataset, squeeze=False)
        self.rc.set_bkg_dark_dataset(dark_dataset)
        self.rc.average_bkg(self.config.database.image_key)
        self.rc.average_bkg_dark(self.config.database.image_key)
        self.rc.dark_subtract_bkg(self.config.database.image_key)
        self.rc.clear_bkg_dark_dataset()
        return

    def imgsub_batch(self, uids: typing.Iterable[str], bkg_uid: str):
        self.load_bkg_data(bkg_uid)
        gen = self._gen_imgsub_data(uids)
        self.rc.set_dataset(xr.merge(gen))
        return

    def _gen_imgsub_data(self, uids: typing.Iterable[str]) -> xr.Dataset:
        gen = tqdm.tqdm(uids, disable=(self.config.verbose <= 0), desc="Experiments")
        for uid in gen:
            self.load_data(uid)
            self.imgsub_data()
            yield self.rc.dataset
        gen.close()
        return

    def imgsub_data(self):
        """Process the data using the image subtraction."""
        self.rc.bkg_img_subtract(self.config.database.image_key)
        self.rc.mask_and_integrate(self.config.database.image_key)
        self._assign_sample_data()
        return

    def process(self, uid: str) -> None:
        """Process the data in a run specified by the uid."""
        run = self.db[uid]
        start = dict(run.metadata["start"])
        start.update(self.config.database.metadata)
        self._cached_start = start
        calibration_md = start[self.config.database.calibration_md_key]
        sc_dk_field_uid = start[self.config.database.sc_dk_field_uid_key]
        dataset = run.primary.to_dask()
        dark_run = self.db[sc_dk_field_uid]
        dark_dataset = dark_run.primary.to_dask()
        self._load_ai_from_calib_result(calibration_md)
        self.rc.set_dataset(dataset)
        self.rc.set_dark_dataset(dark_dataset)
        self.rc.get_I(self.config.database.image_key)
        self._assign_sample_data()
        return

    def process_calib(self, uid: str, bkg_uid: typing.Optional[str] = None) -> None:
        run = self.db[uid]
        start = dict(run.metadata["start"])
        start.update(self.config.database.metadata)
        sc_dk_field_uid = start[self.config.database.sc_dk_field_uid_key]
        dataset = run.primary.read()
        dark_run = self.db[sc_dk_field_uid]
        dark_dataset = dark_run.primary.read()
        bwk = self.config.database.bt_wavelength_key
        if bwk in start:
            self.rc.config.calibration.wavelength = float(start[bwk])
        self.rc.set_dataset(dataset)
        self.rc.set_dark_dataset(dark_dataset)
        if bkg_uid:
            bkg_run = self.db[bkg_uid]
            bkg_dataset = bkg_run.primary.read()
            self.rc.set_bkg_dataset(bkg_dataset)
        self.rc.run_calib_and_fit(self.config.database.image_key)
        return

    def process_batch(self, uids: typing.Iterable[str], bkg_idx: int = -1) -> None:
        """Process and merge the data in a series of run and subtract the data from the background sample."""
        datasets = []
        for uid in tqdm.tqdm(uids):
            self.process(uid)
            datasets.append(self.rc.dataset)
        n = len(datasets)
        merged = xr.merge((datasets[i] for i in range(n) if i != bkg_idx))
        self.rc.set_dataset(merged)
        if bkg_idx > 0:
            bkg_dataset = datasets[bkg_idx]
            self.rc.set_bkg_dataset(bkg_dataset)
        return

    def _assign_sample_data(self) -> None:
        start = self._cached_start
        n = self.rc.dataset.dims["time"]
        sample_data = {k: ("time", [start[k]] * n) for k in self.config.database.sample_data_keys}
        self.rc.dataset = self.rc.dataset.assign(sample_data)
        return

    def _load_ai_from_calib_result(self, calib_result: dict) -> None:
        """Initiate the AzimuthalIntegrator using calibration information."""
        ai = self.config.reduction.geometry
        # different from poni file, set_config only accepts dictionary of lowercase keys
        _calib_result = _lower_key(calib_result)
        # the pyFAI only accept strings so the None should be parsed to a string
        _calib_result = _str_none(_calib_result)
        # the old version of poni uses dist intead of distance and the new version only recognize "distance"
        if ("dist" in _calib_result) and "distance" not in _calib_result:
            _calib_result["distance"] = _calib_result["dist"]
        ai.set_config(_calib_result)
        return


def _lower_key(dct: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """Return dictionary with all keys in lower case."""
    return {key.lower(): value for key, value in dct.items()}


def _str_none(dct: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """Make all the None value to string 'none'."""
    return {key: "none" if value is None else value for key, value in dct.items()}
