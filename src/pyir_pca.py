"""
    .. module:: pyir_pca
       :platform: Windows
       :synopsis: PCA module for all things principal component analysis with FT-IR Spectral data

    .. moduleauthor:: Dougal Ferguson <dougal.ferguson@manchester.ac.uk>

"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pyir_spectralcollection


class PyIR_PCA:
    """PyIR_PCA is a class to be used mainly for the
    usage of PCA tools on the spectral data. This class
    uses super() inheritance to take all functions from Doug_Toolbox().
    Most functions are wrappers of the sklearn.decomposition module
    paired with other toolbox functions.

    """
    def __init__(self, *args, **kwargs):

        self.ubermask = []
        self.pca_module = []
        self.pca_scores = []
        self.pca_loadings = []
        super().__init__(*args, **kwargs)

    def get_vals(self, toolbox_object):
        """ Inherits required data from a specfied toolbox_object. Wavenumbers
        and image dimensions are inherited.

        """
        self.wavenumbers = toolbox_object.wavenumbers
        self.xpixels = toolbox_object.xpixels
        self.ypixels = toolbox_object.ypixels

    def get_labels(self, toolbox_object):
        """ Inherits label objects from a specfied toolbox_object.

        """
        self.class_labels = toolbox_object.class_labels
        self.class_label_names = toolbox_object.class_label_names

    def set_labels(self, Labels):
        """ Sets the pca labels based upon a labels object
        """

        self.class_label_names  = Labels[0]
        self.class_labels= Labels[1]

    def fit_PCA(self, data, n_comp = 50):
        """Fits a PCA model to the data, using n_components.

        :param data: Data to fit PCA to.
        :type data: numpy.array
        :param n_comp: Numper of principal components to fit to. Default = 50
        :type n_comp: int


        :returns: object.pca_module (.PCA()), object.pca_loadings (numpy.array)

        """
        #Checks to see if input is 3D.
        if data.ndim == 3:
            dims, data = pyir_spectralcollection.reshaper_3D(data)
        self.pca_module = sklearn.decomposition.PCA(n_components=n_comp).fit(data)
        self.pca_loadings = self.pca_module.components_
        self.pca_scores = np.dot(data,
                                 self.pca_module.components_[0:n_comp,:].T)

    def plot_pca_image(self, prin_comp = 1, norm = True):
        """Plots an image of specific principal component scores.

        :param prin_comp: Which principal component to use in plot. Default = 1
        :type prin_comp: int
        :param norm: Additional argument to normalise the loadings. Default = True
        :type norm: Bool


        :returns: matplotlib.pyplot plot

        """
        data = np.array(self.pca_module.components_[prin_comp-1,:])
        if norm == True:
            data = 2*((data-np.min(data))/(np.max(data)-np.min(data))) - 1

        data = data.reshape(self.ypixels, self.xpixels)

        title = ('Score on Principal Component ' + str(prin_comp))
        self.disp_image(data, title, colour_bar = True)

    def plot_scores(self, data, prin_comp1, prin_comp2):
        """Plots principal component scores on a matplotlib.pyplot scatter plot.

        :param data: Original data to be transformed by each princ_comp.
        :type data: np.array
        :param prin_comp1: Which principal component to use in plot x-axis.
        :type prin_comp1: int
        :param prin_comp2: Which principal component to use in plot y-axis.
        :type prin_comp2: int


        :returns: matplotlib.pyplot plot

        """
        plt.figure()
        colour_map = plt.cm.get_cmap('rainbow')
        norm = plt.Normalize(vmin= self.class_labels.min(),
                             vmax = self.class_labels.max())
        self.class_labels = (self.class_labels.astype('float64')).ravel()

        x_all = np.dot(data, self.pca_module.components_[prin_comp1-1,:])
        y_all = np.dot(data, self.pca_module.components_[prin_comp2-1,:])

        x_std = np.std(x_all)
        y_std = np.std(y_all)

        #need it to loop through for each "group"
        temp_low = 0
        for i in range(0, len(self.class_label_names)):
            temp_high = temp_low + np.sum(self.class_labels == i)
            plt.scatter(x_all[temp_low:temp_high], y_all[temp_low:temp_high],
                    c = colour_map(norm(self.class_labels)[temp_low:temp_high]),
                    label = self.class_label_names[i][:])
            temp_low = temp_high

        plt.xlabel('PC ' + str(prin_comp1) + ' (' +
                   str(self.pca_module.explained_variance_ratio_
                       [prin_comp1-1]*100)+ ') % var')
        plt.ylabel('PC ' + str(prin_comp2) + ' (' +
                   str(self.pca_module.explained_variance_ratio_
                       [prin_comp2-1]*100)+ ') % var')
        plt.ylim(np.min(y_all) - y_std,
                 np.max(y_all) + y_std)
        plt.xlim(np.min(x_all) - x_std,
                 np.max(x_all) + x_std)

        plt.legend(loc = 'best')
        plt.show()

    def plot_loading(self, prin_comp, ax, wavenums=0):
        """Plots principal component loading across the wavenumber range.

        :param prin_comp: Which principal component to use in plot y-axis.
        :type prin_comp: int
        :param wavenums: Wavenumber array to plot loading against.
        :type wavenums: numpy.array


        :returns: matplotlib.pyplot plot

        """

        temp=0
        if type(wavenums) == int:
            temp = 1
            wavenums = np.arange(0, self.pca_loadings[prin_comp-1,:].shape[0])

        #plt.figure()
        #plt.plot(np.ravel(wavenums), self.pca_loadings[prin_comp-1,:])
        ax.plot(np.ravel(wavenums), self.pca_loadings[prin_comp-1,:])
        #plt.ylabel('weight')
        ax.set_ylabel('wavenumber cm-1')
        #plt.xlabel('wavenumber cm-1')
        ax.set_xlabel('weight')
        if temp == 1:
        #    plt.xlabel('array position')
            ax.set_xlabel('array position')
        ax.set_title('PCA Loading for Principal Component ' + str(prin_comp))
        #plt.show()
        #return ax



    def plot_cum_explained_var(self, max_prin_comps = 15, axes=None,title=""):
        """Plots cumulative explained variance of the principal components.

        :param max_prin_comps: Maximum principal components. Default = 15.
        :type max_prin_comps: int

        :returns: matplotlib.pyplot plot

        """

        if not axes:
            fig = plt.figure()
            ax = plt.axes()
        else:
            ax = axes
        cumsum = np.cumsum(self.pca_module.explained_variance_ratio_[0:max_prin_comps])
        ax.plot(cumsum)
        ax.axvline(x = 15, color = 'g', label = f'{cumsum[15]*100:.2f}% Variance at 16 PC')
        for xval in range(max_prin_comps):
            if cumsum[xval] > 0.9:
                ax.axvline(x = xval, color = 'r', label = f'90% Variance at {xval+1} PC')
                break
        ax.set_xlabel('number of components')
        ax.set_title(title)
        ax.legend()
        ax.set_ylabel('cumulative explained variance')
        if not axes:
            plt.show()


    def dimension_reduce(self, data, n_comps=20):
        """Reduces dataset to user defined n_components.

        :param data: Original data to be transformed by each princ_comp.
        :type data: np.array
        :param n_comps: Number of principal components. Default = 20.
        :type n_comps: int

        :returns: numpy array

        """
        if type(self.pca_module) == list:
            self.fit_PCA(data)

        #Reduce eigenvector matrix down to number of comps selected
        temp_components = self.pca_module.components_
        self.pca_module.components_ = self.pca_module.components_[0:n_comps,:]
        data = self.pca_module.transform(data)

        #return the original loadings just in case
        self.pca_module.components_ = temp_components

        return data

    def noise_reduction(self, data, n_comps=10):
        """Utilises PCA noise reduction technique.

        :param data: Original data to be transformed by each princ_comp.
        :type data: np.array
        :param n_comps: Number of principal components used in noise reduction. Default = 10.
        :type n_comps: int

        :returns: numpy array

        """
        return np.dot(self.pca_scores[:,0:n_comps],
                          self.pca_loadings[0:n_comps,:])

    def kernel_PCA(self, n_comps=None, kernel='linear', **kwargs):
        """Creates a KernelPCA object for non-linear dimensionality reduction using
        KernalPCA module. Once created, the KernelPCA functions can be then used.
        Refer to scikitlearn KernelPCA documentation for additional commands.

        :param n_comps: Number of principal components used.
        :type n_comps: int
        :param kernel: Kernel to be used from 'linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’
        :type kernel: string
        :param **kwargs: Additional inputs of KernelPCA module.
        :type **kwargs: multiple.

        :returns: decomposition.KernalPCA object. Refer to scikitlearn docs.

        """

        return sklearn.decomposition.KernelPCA(n_components = n_comps, kernel = kernel, **kwargs)

    def transform_dims(self, data, pca_mean, loadings, n_comp=0):
        """Transforms the dataset to a specified number of components from
        an external loadings array. In essence generating score values for
        each loading. It is important
        that the inputted data is not mean centered yet and the user provides
        the mean from the trained pca model to subtract such that the
        smoothing operation works correctly.

        :param data: Original data to be dimensionally transformed.
        :type data: np.array
        :param pca_mean: The subtratced mean from the external pca model.
        :type pca_mean: np.array
        :param loadings: Loadings array from a Doug_PCA object.
        :type loadings: np.array
        :param n_comp: User defined number of components, default=0.
        :type n_comp: int


        :returns: numpy array

        """

        if n_comp !=0:
            #Reduce eigenvector matrix down to number of comps selected
            loadings = loadings[0:n_comp]

        #Project data into now dimensional space
        data = np.dot(data-pca_mean, loadings.T)

        return data

    def pca_smoothing(self, data, pca_mean, loadings, n_comp=0):
        """Performs PCA smoothing to a user defined number of principal
        components using an external pca model's loadings. It is important
        that the inputted data is not mean centered yet and the user provides
        the mean from the trained pca model to subtract such that the
        smoothing operation works correctly.

        :param data: Original data to be smoothed.
        :type data: np.array
        :param pca_mean: The subtratced mean from the external pca model.
        :type pca_mean: np.array
        :param loadings: Loadings array.
        :type loadings: np.array
        :param n_comp: User defined number of components to smooth to, default=0.
        :type n_comp: int


        :returns: numpy array

        """

        if n_comp ==0:
            #If no argument presented, assume maximum number of n_components.
            #Assuming loading array is (n_comp, wavenumber) dimensions
            n_comp = min(loadings.shape[0], loadings.shape[1])


        #Step one is to generate the scores array by first subtracting the
        #external pca model's mean from the data, and then multiplying the data
        #by the external data's loadings. This is done using the .transform_dims
        #function.

        scores = self.transform_dims(data, pca_mean, loadings, n_comp)

        #Step two is to then smooth the data by multiply the scores and loading
        #arrays to obtain a smoothed dataset of the same dimensions of the
        #original data, then add the subtracted mean back

        return (np.dot(scores[:,0:n_comp], loadings[0:n_comp,:]))+pca_mean
