import sys

sys.path.append(r'C:\Users\laser\Documents\PHDFILES\github\PyIR\src')
import pyir_spectralcollection as pir
import pyir_image as pir_im
import pyir_mask as pir_mask
import pyir_pca as pir_pca

import numpy as np


class AbstractTransform:
    def __init__(self):
        self.name = ""
        self.desc = ""
        self.kit = pir.PyIR_SpectralCollection()

    def trans_func(self,spectra, wavenumbers, trans_mask=None):
        return spectra,wavenumbers,trans_mask

    def describe(self):
        print(self.name)
        print(self.desc)

class WhatIdidBefore(AbstractTransform):
    def __init__(self):
        super().__init__()
        self.name = "First pytorch non-vanilla transform"
        self.desc = """
        This was the first transformation set I did for pytorch experiments on the full dataset with resnet50 3channel pca.
        Its a strange amide norm, and rubber band correction
        """

    def trans_func(self,spectra, wavenumbers, trans_mask=None):
        spectra, wavenumbers = self.kit.keep_range(3500,1000, spectra, wavenumbers)
        spectra,baseline = self.kit.baseline_correct(spectra,mean=True)
        spectra = spectra / self.kit.area_between(1600,1700,spectra,wavenumbers)[:,None]
        return spectra,wavenumbers,trans_mask

class BiorelevantClip(AbstractTransform):
    def __init__(self):
        super().__init__()
        self.name = "Biorelevant Clip"
        self.desc = """
        Clip to 1820 - 1000 cm^-1 range, determined as biologically relevant by something should probably find what exactly
        """

    def trans_func(self,spectra, wavenumbers, trans_mask=None):
        spectra, wavenumbers = self.kit.keep_range(1820,1000, spectra, wavenumbers)
        return spectra,wavenumbers,trans_mask

class PreDougalTransform(AbstractTransform):
    def __init__(self):
        super().__init__()
        self.name = "Pre-transform Dougal"
        self.desc = """
        The first half of the Dougal transforms, used to generate a PCA model to use for smoothing within the actual transforms.
        """

    def trans_func(self,spectra,wavenumbers,trans_mask=[]):
        if len(trans_mask) > 0: rebuild_tis_mask = np.zeros_like(trans_mask)
        #Quality control (normally amide I threshold)
        am1 = (self.kit.area_between(
            1600,1700, spectra, wavenumbers))
        amide_mask = (am1>2)
        spectra = self.kit.apply_mask(spectra, amide_mask)

        if len(trans_mask) > 0:
            count=0
            for i in np.arange(0, len(trans_mask)):
                if trans_mask[i] == 1:
                    rebuild_tis_mask[i] = amide_mask[count]
                    count = count + 1

            trans_mask = rebuild_tis_mask

        spectra, wavenumbers = self.kit.keep_range(3500,960, spectra, wavenumbers)
        spectra, wavenumbers = self.kit.remove_wax(spectra,wavenumbers)
        spectra, wavenumbers = self.kit.remove_co2(spectra,wavenumbers)

        return spectra,wavenumbers,trans_mask

class DougalPipeline1(AbstractTransform):
    def __init__(self,mean,loadings):
        super().__init__()

        self.name = "Dougal Transform 1"
        self.desc = """
        Quality control (normally amide I threshold)
        Clip to 3500-960 region
        Delete wax regions and co2 region
        PCA smooth/denoise (~23 components)
        Min2zero all spectra (vertical baseline removal)
        vector normalise,
        convert to derivative (1st or 2nd, can change it up
        """
        self.mean = mean
        self.loadings = loadings

    def trans_func(self,spectra,wavenumbers,trans_mask=[]):

        if len(trans_mask) > 0: rebuild_tis_mask = np.zeros_like(trans_mask)
        #Quality control (normally amide I threshold)
        am1 = (self.kit.area_between(
            1600,1700, spectra, wavenumbers))
        amide_mask = (am1>2)
        spectra = self.kit.apply_mask(spectra, amide_mask)

        if len(trans_mask) > 0:
            count=0
            for i in np.arange(0, len(trans_mask)):
                if trans_mask[i] == 1:
                    rebuild_tis_mask[i] = amide_mask[count]
                    count = count + 1

            trans_mask = rebuild_tis_mask
        # Clip to 3500-960 region

        spectra, wavenumbers = self.kit.keep_range(3500,960, spectra, wavenumbers)
        # Delete wax regions and co2 region
        spectra, wavenumbers = self.kit.remove_wax(spectra,wavenumbers)
        spectra, wavenumbers = self.kit.remove_co2(spectra,wavenumbers)

        # PCA smooth/denoise (~23 components)
        temp_pca = pir_pca.PyIR_PCA()
        spectra = temp_pca.pca_smoothing(spectra,self.mean,self.loadings,n_comp=23)

        # Min2zero all spectra (vertical baseline removal)
        spectra = self.kit.all_spec_min2zero(spectra)

        # vector normalise,
        spectra = self.kit.vector_norm(spectra)

        # convert to derivative (1st or 2nd, can change it up
        spectra, wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 17, 5, 1)

        return spectra,wavenumbers,trans_mask



class DougalPipeline2(AbstractTransform):
    def __init__(self,mean,loadings):
        super().__init__()

        self.name = "Dougal Transform 2"
        self.desc = """
        Quality control (normally amide I threshold)
        Clip to 3500-960 region
        Delete wax regions and co2 region
        PCA smooth/denoise (~23 components)
        after the pca smooth clip to 1840-960,
        min2zero
        vector norm
        derivative conversion
        clip to 1800-1000
        """
        self.mean = mean
        self.loadings = loadings

    def trans_func(self,spectra,wavenumbers,trans_mask=[]):

        if len(trans_mask) > 0: rebuild_tis_mask = np.zeros_like(trans_mask)
        #Quality control (normally amide I threshold)
        am1 = (self.kit.area_between(
            1600,1700, spectra, wavenumbers))
        amide_mask = (am1>2)
        spectra = self.kit.apply_mask(spectra, amide_mask)

        if len(trans_mask) > 0:
            count=0
            for i in np.arange(0, len(trans_mask)):
                if trans_mask[i] == 1:
                    rebuild_tis_mask[i] = amide_mask[count]
                    count = count + 1

            trans_mask = rebuild_tis_mask
        # Clip to 3500-960 region

        spectra, wavenumbers = self.kit.keep_range(3500,960, spectra, wavenumbers)
        # Delete wax regions and co2 region
        spectra, wavenumbers = self.kit.remove_wax(spectra,wavenumbers)
        spectra, wavenumbers = self.kit.remove_co2(spectra,wavenumbers)

        # PCA smooth/denoise (~23 components)
        temp_pca = pir_pca.PyIR_PCA()
        spectra = temp_pca.pca_smoothing(spectra,self.mean,self.loadings,n_comp=23)
        spectra, wavenumbers = self.kit.keep_range(1840,960, spectra, wavenumbers)
        # Min2zero all spectra (vertical baseline removal)
        spectra = self.kit.all_spec_min2zero(spectra)
        # vector normalise,
        spectra = self.kit.vector_norm(spectra)
        # convert to derivative (1st or 2nd, can change it up
        spectra, wavenumbers = self.kit.data_deriv(spectra, wavenumbers, 17, 5, 1)
        spectra, wavenumbers = self.kit.keep_range(1800,1000, spectra, wavenumbers)
        return spectra,wavenumbers,trans_mask
