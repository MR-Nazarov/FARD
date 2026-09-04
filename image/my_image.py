import numpy as np
from skimage import io
from image.my_dicom import myDicom
import matplotlib.pyplot as plt
# import av
import scipy.misc
from skimage.util import view_as_windows as viewW
from matplotlib.widgets import Slider
# import astra
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.ndimage as ndimage
from PIL import Image
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
# from mask_CT import find_mask
from scipy.ndimage import affine_transform as Affine


class myImage:

    def __init__(self, im=None, name=None, convertFromPacs=False, cmap=None, clim=None, mask=None, maskTh=None,
                 affineTransform=None):
        self.dicom = myDicom(convertFromPacs=convertFromPacs)
        self.maxSlicesTwoShow = 10
        self.showNoisy = False
        self.pos = None
        self.cmap = cmap
        self.affine = affineTransform
        self.mask = mask
        self.interpolation = None
        self.maskTh = maskTh
        self.clim = clim
        self.ROIS = []
        if (not im is None):
            self.dicom.checkSquareImage(im)  # check and make image square(runs over current version)
            self.set_image(im=im, name=name)

    def set_image(self, im, name=None, slices=None):
        if (isinstance(im, str)):
            self.read_im(im, name=None, slices=slices)
            self.sourcePath = im
        else:
            self.I = im
            if (not self.affine is None):
                self.I = Affine(self.I, np.linalg.inv(self.affine))
            if (not slices is None):
                self.I = self.I[slices, :, :]
            self.imShape = im.shape
            if (name is None):
                self.imName = 'image'
            else:
                self.imName = name
            if (not self.mask is None and self.mask):
                M = find_mask(im=self.I, Th=self.maskTh)
                self.I = self.I * M

    def read_im(self, im, dtype=None, slices=None, name=None):
        self.dtype = dtype
        if (isinstance(im, str)):
            self.sourcePath = im
            if ('jpg' in im or 'png' in im or 'JPG' in im or 'jpeg' in im or 'JPEG' in im):
                self.I = ndimage.imread(im)
            elif ('tif' in im):
                self.I = io.imread(im)
            elif ('dcm' in im):
                self.I = self.dicom.readDicom(im)
            else:
                self.I, self.meta = self.dicom.readDicomSeriesWithMeta(im)
            if (name is None):
                self.imName = im.split('/')[-1]
            else:
                self.imName = name
        else:
            self.I = im
            if (name is None):
                self.imName = 'image'
            else:
                self.imName = name

        if (not self.affine is None):
            self.I = Affine(self.I, np.linalg.inv(self.affine))

        if (not self.dtype is None):
            self.I = self.I.astype(dtype=self.dtype)

        if (not slices is None):
            self.I = self.I[slices, :, :]
        self.imShape = self.I.shape

        if (not self.mask is None and self.mask):
            M = find_mask(im=self.I, Th=self.maskTh)
            self.I = self.I * M

        return self.I

    def set_reconstructed(self, im, dtype=None, slices=None):
        self.dtype = dtype
        if (isinstance(im, str)):
            if ('dcm' in im):
                self.reconstructed = self.dicom.readDicom(im)
            else:
                self.reconstructed, self.meta = self.dicom.readDicomSeriesWithMeta(im)
        else:
            self.reconstructed = im
        if (not self.dtype is None):
            self.reconstructed = self.reconstructed.astype(dtype=self.dtype)

        if (not slices is None):
            self.reconstructed = self.reconstructed[slices, :, :]

        return self.reconstructed

    def sliceNumThreshold(self, Th):
        self.I = self.I[0:Th, :, :]

    def set_noisy(self, noisy):
        self.noisy = noisy

    def clear_ROIS(self):
        self.ROIS = []

    def calc_dose_reduction(self, sliceNum=0, resize=1):
        if (len(self.imShape) > 2):
            im = self.I[sliceNum, :, :]
            imNoisy = self.noisy[sliceNum, :, :]
        else:
            im = self.I
            imNoisy = self.noisy
        im = scipy.misc.imresize(im, 100 * resize)
        imNoisy = scipy.misc.imresize(imNoisy, 100 * resize)
        if (not hasattr(self, 'noisy')):
            self.add_gaussian_noise()
        croppedI, pos = self.crop_image(im)
        croppedNoisy, __ = self.crop_image(imNoisy, pos)
        stdI = np.std(croppedI.flatten())
        stdNoisy = np.std(croppedNoisy.flatten())
        noiseIncrease = stdNoisy / stdI
        reduction = 100 * (1 - 1 / (noiseIncrease ** 2))
        print('reduction is %0.2f' % reduction)
        return reduction

    # def write_video(self,im,path):
    #     output = av.open(path, 'w')
    #     stream = output.add_stream('mpeg4', '15')
    #     stream.bit_rate = 8000000
    #     stream.pix_fmt = 'rgb24'
    #     stream.height = im.shape[1]
    #     stream.width = im.shape[2]
    #
    #     for sliceNum in range(im.shape[0]):
    #
    #         frame = av.VideoFrame.from_ndarray(im[sliceNum,:,:], format='rgb24')
    #         packet = stream.encode(frame)
    #         output.mux(packet)
    #
    #     output.close()
    #
    # def create_backprojection(self, sliceNum, algorithm = 'FBP'):
    #
    #     newSino = blur(self.sinogram[sliceNum,...], sigma = 2)
    #     vol_geom = astra.create_vol_geom(self.imShape[1], self.imShape[2])
    #     reconstruction_id = astra.data2d.create('-vol', vol_geom)
    #     cfg = astra.astra_dict('{}_CUDA'.format(algorithm))
    #     cfg['ReconstructionDataId'] = reconstruction_id
    #     astra.data2d.store(self.sinogram_id[sliceNum], newSino)
    #     cfg['ProjectionDataId'] = self.sinogram_id[sliceNum]
    #     cfg['ProjectorId'] = self.project_id[sliceNum]
    #     algorithm_id = astra.algorithm.create(cfg)
    #     astra.algorithm.run(i = algorithm_id, iterations = 1)
    #     reconstruction = astra.data2d.get(reconstruction_id)
    #
    #     return reconstruction

    def write_image_(self, im, path, source_path='', description='', interp=1):
        if ('avi' in path):
            self.write_video(im.astype(np.uint8), path)
        if ('tif' in path):
            io.imsave(path, im.astype(np.uint8))
        elif ('dcm' in path):
            if (source_path == ''):
                raise RuntimeError('No source path given for writing dcm, aborting!')
            self.dicom.writeDicom(I=im, source_path=source_path, output_path=path, description=description,
                                  interp=interp)
        else:
            if (source_path == ''):
                raise RuntimeError('No source path given for writing dcm, aborting!')
            self.dicom.writeDicomSeries(I=im, source_path=source_path, output_path=path, description=description,
                                        interp=interp)

    def write_sinogram(self, path, im=None, source_path='', description=''):
        if (im is None):
            im = self.sinogram
        self.dicom.writeDicomSeriesOnlyImages(output_path=path, source_path=source_path, I=im, description=description)

    def add_gaussian_noise(self, mu=0, sigma=10, add_to_sinogram=True, sliceNum=None):
        self.noiseModel = 'gaussian'
        self.mu = mu
        self.sigma = sigma
        if (add_to_sinogram):
            if (sliceNum is None):
                self.add_noise_to_3d_sinogram()
            else:
                self.add_noise_to_sinogram(sliceNum)
        else:
            N = np.random.normal(mu, sigma, self.imShape)
            self.noisy = self.I + N

    def add_poisson_noise(self, IO=1e4, add_to_sinogram=True, sliceNum=None):
        self.noiseModel = 'poisson'
        self.IO = IO
        if (add_to_sinogram):
            if (sliceNum is None):
                self.add_noise_to_3d_sinogram()
            else:
                self.add_noise_to_sinogram(sliceNum)
        else:
            N = np.random.poisson(self.I, self.imShape)
            self.noisy = self.I + N

    # def add_noise_to_sinogram(self, sliceNum):
    #     tmp = copy.deepcopy(self.I)
    #     # Create geometries and projector.
    #     vol_geom = astra.create_vol_geom(self.imShape[1], self.imShape[2])
    #     angles = np.linspace(0, np.pi, 360, endpoint=False)
    #     proj_geom = astra.create_proj_geom('parallel', 1., 512, angles)
    #     projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
    #     # create sinogram
    #     sinogram_id, sino = astra.create_sino(data=self.I[sliceNum,:,:], proj_id=projector_id, gpuIndex = 0)
    #     # add noise
    #     if(self.noiseModel == 'poisson'):
    #         sinoNoisy = astra.functions.add_noise_to_sino(sinogram_in = sino, I0 = self.IO)
    #     elif(self.noiseModel == 'gaussian'):
    #         sinoNoisy = self.add_gaussian_noise_(im = sino,mu = self.mu,sigma = self.sigma)
    #     astra.data2d.store(sinogram_id, sinoNoisy)
    #     # Create reconstruction.
    #     reconstruction_id = astra.data2d.create('-vol', vol_geom)
    #     cfg = astra.astra_dict('FBP_CUDA')
    #     cfg['ReconstructionDataId'] = reconstruction_id
    #     cfg['ProjectionDataId'] = sinogram_id
    #     cfg['ProjectorId'] = projector_id
    #     algorithm_id = astra.algorithm.create(cfg)
    #     astra.algorithm.run(i = algorithm_id, iterations = 1)
    #     reconstruction = astra.data2d.get(reconstruction_id)
    #     reconstruction[self.I[sliceNum,:,:] == 0] = 0
    #     reconstruction[reconstruction < 0] = 0
    #     if( not (hasattr('self','noisy'))):
    #         self.noisy = self.I
    #     self.noisy[sliceNum,:,:] = reconstruction
    #     self.I = tmp

    # def create_sinogram_slice(self, sliceNum):
    #     vol_geom = astra.create_vol_geom(self.imShape[1], self.imShape[2])
    #     angles = np.linspace(0, 2*np.pi, 2*360, endpoint=False)
    #     sourceDetectorDistance = 950
    #     sourcePatientDistance = 541
    #     detectorPatientDistance = sourceDetectorDistance - sourcePatientDistance
    #     det_width = 1. # Size of a detector pixel.
    #     det_count = 1500 # Number of detector pixels.
    #     source_origin = sourcePatientDistance #Position of the source.
    #     origin_det = detectorPatientDistance #Position of the detector
    #     proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
    #     projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
    #     # create sinogram
    #     sinogram_id, sino = astra.create_sino(data=self.I[sliceNum,:,:], proj_id=projector_id, gpuIndex = 0)
    #     return sino, sinogram_id, projector_id

    def create_sinogram_3D_fan_beam(self):

        self.sinogram_id = []
        self.project_id = []
        for sliceNum in range(self.imShape[0]):
            if (not sliceNum):
                self.sinogram, sin_id, proj_id = self.create_sinogram_slice(sliceNum=sliceNum)
                self.sinogram = self.sinogram[None, ...]
            else:
                sinogram_, sin_id, proj_id = self.create_sinogram_slice(sliceNum=sliceNum)
                self.sinogram = np.concatenate((self.sinogram, sinogram_[None, ...]), axis=0)
            self.sinogram_id.append(sin_id)
            self.project_id.append(proj_id)
        return self.sinogram

    # def add_noise_to_3d_sinogram(self):
    #
    #     tmp = copy.deepcopy(self.I)
    #     IforSino = self.I
    #     # Create geometries and projector.
    #     vol_geom = astra.create_vol_geom(IforSino.shape[1],IforSino.shape[2],IforSino.shape[0])
    #     angles = np.linspace(0, np.pi, 360, endpoint=False)
    #     pixelSpacing = self.meta[0]['0x28','0x30'].value
    #     proj_geom = astra.create_proj_geom('parallel3d', 1., 1., IforSino.shape[0],IforSino.shape[1], angles)
    #     vol_id = astra.data3d.link('-vol', vol_geom, np.ascontiguousarray(IforSino.astype(np.float32)))
    #     # create sinogram
    #     sinogram_id, sino = astra.create_sino3d_gpu(data = vol_id, proj_geom = proj_geom, vol_geom = vol_geom, gpuIndex = 0)
    #     self.sinogram = sino
    #     # add noise
    #     if(self.noiseModel == 'poisson'):
    #         sinoNoisy = astra.functions.add_noise_to_sino(sinogram_in=sino, I0=self.IO)
    #     elif(self.noiseModel == 'gaussian'):
    #         sinoNoisy = self.add_gaussian_noise_(im=sino, mu=self.mu, sigma=self.sigma)
    #     self.noisySinogram = sinoNoisy
    #     astra.data3d.store(sinogram_id, sinoNoisy)
    #     # Create reconstruction.
    #     rec_id = astra.data3d.create('-vol', vol_geom)
    #     cfg = astra.astra_dict('CGLS3D_CUDA')
    #     cfg['ReconstructionDataId'] = rec_id
    #     cfg['ProjectionDataId'] = sinogram_id
    #     alg_id = astra.algorithm.create(cfg)
    #     astra.algorithm.run(i = alg_id, iterations = 100)
    #     reconstruction = astra.data3d.get(rec_id)
    #     astra.data3d.delete(rec_id)
    #     astra.data3d.delete(vol_id)
    #     astra.data3d.delete(sinogram_id)
    #     astra.algorithm.delete(alg_id)
    #     reconstruction[IforSino == 0] = 0
    #     reconstruction[reconstruction < 0] = 0
    #     self.noisy = reconstruction
    #     self.I = tmp

    def add_gaussian_noise_to_sinogram(self, mu, sigma):

        pass

    def add_gaussian_noise_(self, im, mu, sigma):
        if (len(im.shape) == 2):
            N = np.random.normal(mu, sigma, (im.shape[0], im.shape[1]))
        else:
            N = np.random.normal(mu, sigma, (im.shape[0], im.shape[1], im.shape[2]))
        noisy_ = im + N
        return noisy_

    # def create_sinogram_3D(self):
    #     tmp = copy.deepcopy(self.I)
    #     IforSino = self.I
    #     # Create geometries and projector.
    #     vol_geom = astra.create_vol_geom(IforSino.shape[1], IforSino.shape[2], IforSino.shape[0])
    #     angles = np.linspace(0, np.pi, 360, endpoint=False)
    #     proj_geom = astra.create_proj_geom('parallel3d', 1., 1., IforSino.shape[0], IforSino.shape[1], angles)
    #     vol_id = astra.data3d.link('-vol', vol_geom, np.ascontiguousarray(IforSino.astype(np.float32)))
    #     # create sinogram
    #     _, self.sinogram = astra.create_sino3d_gpu(data=vol_id, proj_geom=proj_geom, vol_geom=vol_geom, gpuIndex=0)
    #     self.I = tmp
    #     return self.sinogram

    def write_noisy_image(self, path, source_patch='', description=''):
        self.write_image_(self.noisy, path, source_patch, description)

    def write_image(self, path, source_patch='', description=''):
        self.write_image_(self.I, path, source_patch, description)

    def write_sino_noisy(self, path, source_path='', description=''):
        self.write_image_(self.noisySinogram, path, source_path, description)

    def crop(self, pos=None, initial=None, useSelfPos=None, resize=None):
        if (not useSelfPos is None and useSelfPos):
            pos = self.pos
        self.initial = initial
        self.I = self.crop_(self.I, pos)
        if (not self.dtype is None):
            self.I = self.I.astype(dtype=self.dtype)
        if (hasattr(self, 'reconstructed')):
            self.reconstructed = self.crop_(self.reconstructed, self.pos)
        if (hasattr(self, 'noisy')):
            self.noisy = self.crop_(self.noisy, self.pos)
        if (not resize is None):
            self.resize(resize)
        self.imShape = self.I.shape

    def crop_(self, im, pos=None):
        if (len(self.imShape) > 2):
            tmp, pos = self.crop_scan(im, pos)
        else:
            tmp, pos = self.crop_image(im, pos)
        self.pos = pos
        return tmp

    def im2col(self, I=None, blockSize=(5, 5), stepsize=1, sliceNum=None):
        if (I is None):
            if (sliceNum is None):
                patches = viewW(self.I, (blockSize[0], blockSize[1])).reshape(-1, blockSize[0] * blockSize[1],
                                                                              order='F')[:, ::stepsize]
            else:
                patches = viewW(self.I[sliceNum, :, :], (blockSize[0], blockSize[1])).reshape(-1,
                                                                                              blockSize[0] * blockSize[
                                                                                                  1], order='F')[:,
                          ::stepsize]
        else:
            if (sliceNum is None):
                patches = viewW(I, (blockSize[0], blockSize[1])).reshape(-1, blockSize[0] * blockSize[1], order='F')[:,
                          ::stepsize]
            else:
                patches = viewW(I[sliceNum, :, :], (blockSize[0], blockSize[1])).reshape(-1,
                                                                                         blockSize[0] * blockSize[1],
                                                                                         order='F')[:, ::stepsize]
        return patches

    def im3col(self, I=None, blockSize=(5, 5, 5)):
        if (I is None):
            patches = viewW(self.I.transpose((0, 2, 1)), (blockSize[0], blockSize[1], blockSize[2])).reshape(
                (-1, blockSize[0] * blockSize[1] * blockSize[2]))
        else:
            patches = viewW(I.transpose((0, 2, 1)), (blockSize[0], blockSize[1], blockSize[2])).reshape(
                (-1, blockSize[0] * blockSize[1] * blockSize[2]))
        return patches

    def col2im(self, B, blockSize, imageSize):
        m, n = blockSize
        mm, nn = imageSize
        return B.reshape((mm - m + 1, nn - n + 1), order='F')

    def crop_image(self, im=None, pos=None, cropSelf=False, cmap='gray', resize=None):

        if (im is None):
            im = self.I
        if (pos is None):
            pos = self.get_pos(im=im, cmap=cmap)
            # pos = cv2.selectROI('selectROI', im.astype(np.uint8), fromCenter = False, showCrosshair = False)
        if (not hasattr(self, 'initial') or self.initial is None):
            # cropped = im[int(pos[1]):int(pos[1] + pos[3]), int(pos[0]):int(pos[0] + pos[2])]
            # BUG FIX: to support non-square crops
            cropped = im[int(pos[0]):int(pos[0] + pos[2]), int(pos[1]):int(pos[1] + pos[3])]
        else:
            cropped = im[int(pos[1]):int(pos[1] + self.initial[3]), int(pos[0]):int(pos[0] + self.initial[2])]
        if (cropSelf):
            self.set_image(cropped)
        if (not resize is None):
            cropped = self.resize_(cropped, resize)
        return cropped, pos

    def crop_scan(self, im=None, pos=None, useSelfPos=False, initialCrop=None):

        self.initial = initialCrop
        if (useSelfPos):
            pos = self.pos
        if (im is None):
            im = self.I

        if isinstance(im, list):
            dtype = im[0].dtype
        else:
            dtype = im.dtype

        if (len(self.imShape) == 2):
            croppedImage, pos = self.crop_image(im)
            return croppedImage, pos
        else:
            for i in range(self.imShape[0]):
                if (i == 0 and pos is None):
                    if isinstance(im, list):
                        croppedImage, pos = self.crop_image(im[i])
                    else:
                        croppedImage, pos = self.crop_image(im[i, :, :])
                    self.pos = pos
                    croppedScan = np.zeros((self.imShape[0], croppedImage.shape[0], croppedImage.shape[1]), dtype=dtype)
                elif (not pos is None and i == 0):
                    if isinstance(im, list):
                        croppedImage, __ = self.crop_image(im[i], pos)
                    else:
                        croppedImage, __ = self.crop_image(im[i, :, :], pos)
                    croppedScan = np.zeros((self.imShape[0], croppedImage.shape[0], croppedImage.shape[1]), dtype=dtype)
                else:
                    if isinstance(im, list):
                        croppedImage, __ = self.crop_image(im[i], pos)
                    else:
                        croppedImage, __ = self.crop_image(im[i, :, :], pos)
                croppedScan[i, :, :] = croppedImage
        return croppedScan, pos

    def get_pos_(self, im=None):
        if (im is None):
            im = self.I
        _, pos = self.crop_image(im)
        return pos

    def show_(self, im=None):
        if (im is None):
            im = self.I
        if (len(im.shape) > 2 and im.shape[2] != 3):
            cnt = 0
            for sliceNum in range(self.startSlice, self.imShape[0]):
                cnt += 1
                if (cnt > self.maxSlicesTwoShow):
                    break
                if (not self.showNoisy):
                    fig = plt.figure()
                    fig.canvas.set_window_title(self.imName)
                    plt.imshow(im[sliceNum, :, :], cmap=self.cmap, clim=self.clim, aspect=self.aspect,
                               interpolation=self.interpolation)
                    plt.title(self.title + ' Slice Number is ' + str(sliceNum))
                else:
                    fig, axes = plt.subplots(2, 1)
                    fig.canvas.set_window_title(self.imName + ' slice number is ' + str(sliceNum) + ' norm SSD = ' +
                                                str(np.sum(np.abs(
                                                    im[sliceNum, :, :] - self.noisy[sliceNum, :, :])) / self.noisy[
                                                                                                        sliceNum, :,
                                                                                                        :].size))
                    axes[0].imshow(im[sliceNum, :, :], cmap=self.cmap, clim=self.clim, interpolation=self.interpolation)
                    axes[0].set_title('image')
                    axes[1].imshow(self.noisy[sliceNum, :, :], cmap=self.cmap, clim=self.clim,
                                   interpolation=self.interpolation)
                    axes[1].set_title('image noisy')
        else:
            if (not self.showNoisy):
                fig = plt.figure()
                fig.canvas.set_window_title(self.imName)
                plt.imshow(im, cmap=self.cmap, clim=self.clim, aspect=self.aspect, interpolation=self.interpolation)
                plt.title(self.title)
            else:
                fig, axes = plt.subplots(2, 1)
                fig.canvas.set_window_title(self.imName + self.title + ' norm SSD = ' +
                                            str(np.sum(np.abs(im - self.noisy)) / self.noisy.size))
                axes[0].imshow(im, cmap=self.cmap, clim=self.clim, interpolation=self.interpolation)
                axes[0].set_title('image')
                axes[1].imshow(self.noisy, cmap=self.cmap, clim=self.clim, interpolation=self.interpolation)
                axes[1].set_title('image noisy')

    def show_sino_(self):
        if (len(self.imShape) > 2):
            cnt = 0
            for sliceNum in range(self.startSlice, self.imShape[0]):
                cnt += 1
                if (cnt > self.maxSlicesTwoShow):
                    break
                if (not self.showNoisy):
                    fig = plt.figure()
                    fig.canvas.set_window_title(self.imName)
                    plt.imshow(self.sinogram[sliceNum, :, :], cmap='gray', clim=self.clim)
                    plt.title(self.title + ' Slice Number is ' + str(sliceNum))
                else:
                    fig, axes = plt.subplots(2, 1)
                    fig.canvas.set_window_title(self.imName + ' slice number is ' + str(sliceNum) + ' norm SSD = ' +
                                                str(np.sum(np.abs(
                                                    self.sinogram[sliceNum, :, :] - self.noisySinogram[sliceNum, :,
                                                                                    :])) /
                                                    self.noisySinogram[sliceNum, :, :].size))
                    axes[0].imshow(self.sinogram[sliceNum, :, :], cmap='gray', clim=self.clim)
                    axes[0].set_title('image')
                    axes[1].imshow(self.noisySinogram[sliceNum, :, :], cmap='gray', clim=self.clim)
                    axes[1].set_title('image noisy')
        else:
            if (not self.showNoisy):
                fig = plt.figure()
                fig.canvas.set_window_title(self.imName)
                plt.imshow(self.sinogram, cmap='gray', clim=self.clim)
                plt.title(self.title)
            else:
                fig, axes = plt.subplots(2, 1)
                fig.canvas.set_window_title(self.imName + self.title + ' norm SSD = ' +
                                            str(np.sum(
                                                np.abs(self.sinogram - self.noisySinogram)) / self.noisySinogram.size))
                axes[0].imshow(self.sinogram, cmap='gray', clim=self.clim)
                axes[0].set_title('image')
                axes[1].imshow(self.noisySinogram, cmap='gray', clim=self.clim)
                axes[1].set_title('image noisy')

    def resize_(self, im, scale, new_size=None):
        # if new_size if given, scale is not used. new_size is the new image size
        newIm = Image.fromarray(im)
        newIm = newIm.resize((np.int(im.shape[1] * scale), np.int(im.shape[0] * scale)), Image.BICUBIC)
        if new_size is not None:
            newIm = newIm.resize((np.int(new_size), np.int(new_size)), Image.BICUBIC)
        newIm = np.asarray(newIm, order='F')

        return newIm

    def resize(self, scale, new_size=None):

        if (len(self.imShape) == 2 or self.imShape[2] == 3):
            self.I = self.resize_(self.I, scale, new_size)
        else:
            for sliceNum in range(self.imShape[0]):
                im = self.resize_(self.I[sliceNum, :, :], scale, new_size)
                if (sliceNum == 0):
                    newIm = np.zeros((self.imShape[0], im.shape[0], im.shape[1]),
                                     dtype=self.I.dtype)  # was self.imShape[0], im.shape[1], im.shape[2]), dtype = self.I.dtype
                newIm[sliceNum, :, :] = im
            self.I = newIm
            del newIm

    def show(self, title='', dontShow=False, maxSlicesoShow=None, startSlice=None, closeWithMouseClick=False,
             aspect=None, im=None, save=None, interpolation=None):
        self.aspect = aspect
        self.interpolation = interpolation
        if (not startSlice is None):
            self.startSlice = startSlice
        else:
            self.startSlice = 0
        if (not maxSlicesoShow is None):
            self.maxSlicesTwoShow = maxSlicesoShow
        self.useClim = False
        if (hasattr(self, 'noisy')):
            self.showNoisy = True
        else:
            self.showNoisy = False
        self.title = title
        self.show_(im=im)
        if (not save is None):
            plt.savefig('/media/pihash/DATA/Research/Michael_G/python_codes/CT/FIGS/' + save + '.tif')
        if (closeWithMouseClick):
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()
        if (not dontShow):
            plt.show()
        else:
            plt.draw()

    def show_sino(self, title='', dontShow=False, maxSlicesoShow=None, startSlice=None):
        if (not hasattr(self, 'sinogram')):
            return
        if (not startSlice is None):
            self.startSlice = startSlice
        else:
            self.startSlice = 0
        if (not maxSlicesoShow is None):
            self.maxSlicesTwoShow = maxSlicesoShow
        self.useClim = False
        if (hasattr(self, 'noisy')):
            self.showNoisy = True
        else:
            self.showNoisy = False
        self.title = title
        self.show_sino_()
        if (not dontShow):
            plt.show()

    def show_with_map(self, im=None):

        if (im is None):
            im = self.I

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.25, bottom=0.25)
        min0 = 0
        max0 = 25000

        im1 = ax.imshow(im)
        fig.colorbar(im1)

        axcolor = 'lightgoldenrodyellow'
        axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

        smin = Slider(axmin, 'Min', 0, 30000, valinit=min0)
        smax = Slider(axmax, 'Max', 0, 30000, valinit=max0)

        def update(val):
            im1.set_clim([smin.val, smax.val])
            fig.canvas.draw()

        smin.on_changed(update)
        smax.on_changed(update)

        plt.show()

    def show_ROI(self, pos=None, im=None, dont_show=False, title='ROI', edgecolor='r', save=None, interpolation=None):
        if (pos is None and len(self.ROIS) == 0):
            return
        if (im is None):
            im = self.I
        # Create figure and axes
        fig, ax = plt.subplots(1)
        ax.imshow(im, cmap=self.cmap, clim=self.clim, interpolation=interpolation)
        if (not isinstance(edgecolor, list) and len(self.ROIS) > 0):
            edgecolor = [edgecolor] * len(self.ROIS)
        # Create a Rectangle data
        if (pos is None and len(self.ROIS) > 0):
            for i, ROI in enumerate(self.ROIS):
                rect = patches.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], linewidth=3, edgecolor=edgecolor[i],
                                         facecolor='none')
                # Add the data to the Axes
                ax.add_patch(rect)
        elif (not pos is None):
            rect = patches.Rectangle((pos[0], pos[1]), pos[2], pos[3], linewidth=3, edgecolor=edgecolor,
                                     facecolor='none')
            # Add the data to the Axes
            ax.add_patch(rect)
        plt.title(title)
        if (not save is None):
            plt.savefig('/media/pihash/DATA/Research/Michael_G/python_codes/CT/FIGS/' + save + '.tif')
        if (not dont_show):
            plt.show()
        else:
            plt.draw()

    def compute_ssim_(self):

        SSIM = []
        dataRange = 4000
        for sliceNum in range(self.imShape[0]):
            SSIM.append(ssim(self.I[sliceNum, :, :], self.reconstructed[sliceNum, :, :], data_range=dataRange))

        return np.mean(SSIM)

    def compute_psnr_(self):

        PSNR = []
        dataRange = 4000
        for sliceNum in range(self.imShape[0]):
            PSNR.append(psnr(self.I[sliceNum, :, :], self.reconstructed[sliceNum, :, :], data_range=dataRange))

        return np.mean(PSNR)

    def compute_fsim_(self):

        from for_articles.FSIM import FSIM
        fsim = FSIM(self.I, self.reconstructed)
        return np.mean(fsim)

    def get_loc(self, sliceNum=None, numLocs=None, title=None):
        if ((len(self.imShape) > 2 and self.imShape[2] > 3) or not sliceNum is None):
            plt.imshow(self.I[sliceNum, :, :], clim=self.clim, cmap=self.cmap)
        else:
            plt.imshow(self.I, clim=self.clim, cmap=self.cmap)
        if (not title is None):
            plt.title(title)
        loc = plt.ginput(numLocs)
        plt.close()
        return loc

    def compute_quality_index(self, index):

        if (index == 'ssim'):
            result = self.compute_ssim_()
        elif (index == 'psnr'):
            result = self.compute_psnr_()
        elif (index == 'fsim'):
            result = self.compute_fsim_()

        return result

    def close(self):
        plt.close()

    def add_ROI(self, pos):
        self.ROIS.append(pos)

    def get_pos(self, im=None, cmap=None, initialPos=None):

        if (im is None):
            im = self.I

        def line_select_callback(eclick, erelease):
            # 'eclick and erelease are the press and release events'
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            pos = np.array([x1, y1, x2 - x1, y2 - y1])
            self.pos = [np.int(x) for x in pos]

        def toggle_selector(event):
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                toggle_selector.RS.set_active(True)

        def press(event):
            print('pressed', event.key)
            if event.key == 'enter' or event.key == 'space':
                plt.close()

        fig, current_ax = plt.subplots()
        current_ax.imshow(im, cmap=cmap, clim=self.clim)
        # drawtype is 'box' or 'line' or 'none'
        toggle_selector.RS = RectangleSelector(current_ax, line_select_callback, drawtype='box', useblit=False,
                                               button=[1, 3], spancoords='pixels', interactive=True)

        plt.connect('key_press_event', toggle_selector)
        plt.connect('key_press_event', press)

        if (not initialPos is None):
            toggle_selector.RS.to_draw.set_visible(True)
            toggle_selector.RS.extents = (0, initialPos[0], 0, initialPos[1])

        plt.show()
        return self.pos

    def anonymize_series(self, output_path, new_person_name="anonymous", new_patient_id='', description=''):

        for i in range(len(self.meta)):
            self.meta[i] = self.dicom.anonymize(self.meta[i], new_person_name, new_patient_id)

        self.dicom.writeDicomSeriesUsingMeta(self.I, self.meta, output_path, description)

    def convert_axial_to_sagittal(self, dest_path):
        "convert an axial dicom series to sagittal and saves"
        sagittal_img = np.zeros((self.imShape[2], self.imShape[0], self.imShape[1]), dtype=self.I.dtype)
        for sliceNum in range(self.imShape[2]):
            sagittal_img[sliceNum, :, :] = self.I[:, :, sliceNum]
        sagittal_img = np.flip(sagittal_img, axis=1)
        self.dicom.change_meta_from_axial_to_sagittal(sagittal_img, self.meta, dest_path,
                                                      description='Converted to Sagittal')  # edits meta fields saves dicom
