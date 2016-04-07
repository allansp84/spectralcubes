# -*- coding: utf-8 -*-

import os
import sys
import cv2
import bob.io.video as video
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import ndimage
from scipy import mgrid
from scipy import stats
from scipy.signal import *
import skimage.color as skcolor
from sklearn.metrics import mutual_info_score
from antispoofing.spectralcubes.utils import *
# import bob.ip.facedetect as facedetect


class LowLevelFeatures(object):

    _MAX_FRAME_NUMBERS = 1000
    debug = False

    def __init__(self, dataset_path, output_path, input_fname, roi, measure_type,
                 filter_type='gauss',
                 kernel_width=KERNEL_WIDTH,
                 sigma=SIGMA,
                 n_cuboids=N_CUBOID,
                 cuboid_width=CUBOID_WIDTH,
                 cuboid_depth=CUBOID_DEPTH,
                 analize=False,
                 color=True,
                 flatten=True,
                 only_face=False,
                 frame_numbers=0,
                 seed=SEED,
                 facelocations_path=None,
                 get_faceloc=None):

        # -- private attributes
        self.__dataset_path = ""
        self.__output_path = ""
        self.__input_fname = ""
        self.__measure_type = ""
        self.__roi = ""
        self.__frame_numbers = 0

        # -- public attributes
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.input_fname = input_fname
        self.roi = roi
        self.measure_type = measure_type
        self.filter_type = filter_type
        self.kernel_width = kernel_width
        self.sigma = sigma
        self.n_cuboids = n_cuboids
        self.cuboid_width = cuboid_width
        self.cuboid_depth = cuboid_depth
        self.seed = seed
        self.analize = analize
        self.color = color
        self.flatten = flatten
        self.only_face = only_face
        self.frame_numbers = frame_numbers
        self.operation = IMG_OP
        self.box_shape = BOX_SHAPE
        self.get_faceloc = get_faceloc
        self.facelocations_path = facelocations_path

    @property
    def dataset_path(self):
        return self.__dataset_path

    @dataset_path.setter
    def dataset_path(self, path):
        self.__dataset_path = os.path.abspath(path)

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, value):
        self.__output_path = os.path.abspath(value)

    @property
    def input_fname(self):
        return self.__input_fname

    @input_fname.setter
    def input_fname(self, value):
        self.__input_fname = os.path.abspath(value)

    @property
    def measure_type(self):
        return self.__measure_type

    @measure_type.setter
    def measure_type(self, value):
        try:
            self.__measure_type = measures_available[value]
        except KeyError:
            raise ValueError('invalid measure option')

    @property
    def roi(self):
        return self.__roi

    @roi.setter
    def roi(self, value):
        try:
            self.__roi = rois_available[value]
        except KeyError:
            raise ValueError('invalid roi option')

    @property
    def frame_numbers(self):
        return self.__frame_numbers

    @frame_numbers.setter
    def frame_numbers(self, frame_numbers):
        if frame_numbers > 0:
            self.__frame_numbers = frame_numbers
        else:
            self.__frame_numbers = self._MAX_FRAME_NUMBERS

    def extracting_face_region(self, frames):

        face_boxes = self.__detecting_faces(frames)
        # face_boxes = self.face_detection()
        faces, boxes_adjusted = self.__normalized_faces(face_boxes, frames)

        return faces

    def load_video(self):

        input_video = video.reader(self.input_fname)
        frames = input_video.load()
        # frames = bob.io.load(self.input_fname)
        frames = np.rollaxis(frames, 1, 4)

        if self.frame_numbers:
            frames = frames[:self.frame_numbers, :, :, :]

        n_frames, n_rows, n_cols, n_channels = frames.shape

        if not self.color:

            gray_frames = np.zeros((n_frames, n_rows, n_cols), dtype=frames.dtype)

            for f in xrange(n_frames):
                # gray_frames[f] = skcolor.rgb2hsv(frames[f])[:,:,2] * 255
                gray_frames[f] = skcolor.rgb2gray(frames[f]) * 255

            frames = gray_frames[:, :, :, np.newaxis]

        # else:
        #     for f in xrange(n_frames):
        #         frames[f] = skcolor.rgb2hsv(frames[f]) * 255

        results = frames

        return results

    @staticmethod
    def __lower_limit(value, threshould):
        if value < threshould:
            return threshould
        else:
            return value

    def __detecting_faces(self, imgs_):

        face_boxes = []
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        imgs = imgs_.copy()
        n_frames, n_rows, n_cols, n_channels = imgs.shape

        for i, img in enumerate(imgs):

            gray = img

            if n_channels == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray_eq = cv2.equalizeHist(gray)

            face_box = face_cascade.detectMultiScale(gray_eq, SCALE_FACTOR, MIN_NEIGHBORS, minSize=(30, 30),
                                                     flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            # flags=cv2.cv.CV_HAAR_DO_CANNY_PRUNING | cv2.cv.CV_HAAR_DO_ROUGH_SEARCH | cv2.cv.CV_HAAR_FEATURE_MAX |
            # cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV_HAAR_SCALE_IMAGE

            y, x = gray.shape[0] / 2, gray.shape[1] / 2
            h, w = self.box_shape[0], self.box_shape[1]

            if len(face_box) != 0:
                x, y, w, h = face_box[0]

            face_boxes.append([i, x, y, w, h])

        return np.array(face_boxes, np.int)

    def __normalized_faces(self, boxes, imgs):
        norm_faces = []
        boxes_faces = []

        for box, img in zip(boxes, imgs):
            x, y, w, h = box
            if self.operation == 'crop':
                new_img, box_adjusted = self.__crop_face(img, y, x, max(w, h))

            elif self.operation == 'resize':
                new_img, box_adjusted = self.__resize_face(img, y, x, max(w, h))

            elif self.operation == 'None':
                new_img = img
                box_adjusted = box

            else:
                print 'Operation is not applicable !'
                sys.exit(0)

            norm_faces.append(new_img)
            boxes_faces.append(box_adjusted)

        return np.array(norm_faces), np.array(boxes_faces)

    def __resize_face(self, face, y, x, dim):
        new_height = self.box_shape[0]
        new_width = self.box_shape[1]

        dim_ = dim - dim * 0.8

        x += dim_ / 2
        y += dim_ / 2
        dim -= dim_

        if (y + dim) > face.shape[0]:
            shift = (y + dim) - face.shape[0]
            y -= shift

        if (x + dim) > face.shape[1]:
            shift = (x + dim) - face.shape[1]
            x -= shift

        y = self.__lower_limit(y, 0)
        x = self.__lower_limit(x, 0)

        roi = face[y:y + dim, x:x + dim, :]
        box = [x, y, dim, dim]

        resized_face = cv2.resize(roi, (new_width, new_height))

        if not self.color:
            resized_face = resized_face[:, :, np.newaxis]

        return resized_face, box

    def __crop_face(self, face, y, x, dim):
        new_height = self.box_shape[0]
        new_width = self.box_shape[1]

        y -= (new_height - dim) / 2
        x -= (new_width - dim) / 2

        if (y + new_height) > face.shape[0]:
            shift = (y + new_height) - face.shape[0]
            y -= shift

        if (x + new_width) > face.shape[1]:
            shift = (x + new_width) - face.shape[1]
            x -= shift

        y = self.__lower_limit(y, 0)
        x = self.__lower_limit(x, 0)
        box = [x, y, new_width, new_height]

        croped_face = face[y:y + new_height, x:x + new_width, :]

        return croped_face, box

    # def __get_faces_from_locations(self, faces):
    #     norm_faces = []
    #     for box, face in zip(boxes, faces):
    #         x, y, w, h = box
    #
    #         if self.operation == 'crop':
    #             new_img = self.__crop_face(face, y, x, max(w, h))
    #             norm_faces.append(new_img)
    #         else:
    #             new_img = face[:, y:y + h, x:x + w]
    #             norm_faces.append(new_img)
    #
    #     return np.array(norm_faces)

    def get_faces(self, frames):

        # n_frames, n_rows, n_cols, n_channels = frames.shape
        # faces = np.zeros((n_frames, 100, 100, n_channels), dtype=frames.dtype)

        fname_faceloc = self.input_fname.replace('replayattack', 'replayattack/face-locations')
        fname_faceloc = os.path.splitext(fname_faceloc)[0] + '.face'

        boxes = np.loadtxt(fname_faceloc, dtype=np.int, delimiter=" ")

        norm_faces = []
        for f, (box, face) in enumerate(zip(boxes, frames)):
            x, y, w, h = box[1], box[2], box[3], box[4]

            new_img = self.__crop_face(face, y, x, max(w, h))
            norm_faces.append(new_img)

        return np.array(norm_faces)

    def __gauss_kernel(self):

        # kernel_half_width = np.int32(self.kernel_width / 2)
        kernel_half_width = self.kernel_width / 2

        y, x = mgrid[-kernel_half_width:kernel_half_width + 1, -kernel_half_width:kernel_half_width + 1]

        kernel_not_normalized = np.exp((-(x ** 2 + y ** 2)) / (2. * self.sigma ** 2))
        normalization_constant = np.float32(1) / np.sum(kernel_not_normalized)
        kernel = (normalization_constant * kernel_not_normalized).astype(np.float32)

        return kernel

    def gaussian_filter(self, frame):

        kernel = self.__gauss_kernel()
        conv_frame = ndimage.convolve(frame, kernel, mode='mirror')

        return conv_frame

    def residual_noise_video(self, frames):

        if self.debug:
            print "\t- computing residual noise video ..."
            sys.stdout.flush()

        n_frames, n_row, n_col, n_channels = frames.shape
        filtered_frames = np.zeros(frames.shape, dtype=frames.dtype)
        residual_noise = np.zeros(frames.shape, dtype=frames.dtype)

        if "gauss" in self.filter_type:
            for f in xrange(n_frames):
                for c in xrange(n_channels):
                    filtered_frames[f, :, :, c] = self.gaussian_filter(frames[f, :, :, c])

        elif "median" in self.filter_type:
            for f in xrange(n_frames):
                for c in xrange(n_channels):
                    filtered_frames[f, :, :, c] = medfilt2d(frames[f, :, :, c], self.kernel_width)

        elif "wiener" in self.filter_type:
            for f in xrange(n_frames):
                for c in xrange(n_channels):
                    filtered_frames[f, :, :, c] = np.abs(wiener(frames[f, :, :, c], (self.kernel_width, self.kernel_width)))
        else:
            raise ValueError('filter not implemented.')

        for f in xrange(n_frames):
            for c in xrange(n_channels):
                noise = frames[f, :, :, c] - filtered_frames[f, :, :, c]
                residual_noise[f, :, :, c] = np.abs(noise)

        return residual_noise

    @staticmethod
    def normalization(arrs):

        n_frames, n_row, n_col, n_channels = arrs.shape
        arrs_norm = np.zeros(arrs.shape, dtype=arrs.dtype)

        for f in xrange(n_frames):
            for c in xrange(n_channels):
                maxmin = (arrs[f, :, :, c].max() - arrs[f, :, :, c].min())
                maxmin = 1. if maxmin == 0 else maxmin
                arr_minmax = (arrs[f, :, :, c] - arrs[f, :, :, c].min()) / maxmin
                arrs_norm[f, :, :, c] = 255 * arr_minmax

        return arrs_norm

    def spectral_analysis(self, frames):

        if self.debug:
            print "\t- computing spectral video ..."
            sys.stdout.flush()

        n_frames, n_row, n_col, n_channels = frames.shape
        spectra = np.zeros(frames.shape, dtype=np.float32)
        pwr2 = np.array([2], dtype=np.float32)

        if "phase" in self.measure_type:
            for f in xrange(n_frames):
                for c in xrange(n_channels):

                    fft_frames = np.fft.fft2(frames[f, :, :, c].astype(np.float32))
                    angle = np.angle(fft_frames)
                    angle_shift = np.fft.fftshift(angle)

                    if self.analize:
                        spectra[f, :, :, c] = angle_shift
                    else:
                        if "energy" in self.measure_type:
                            spectra[f, :, :, c] = np.power(angle_shift, pwr2)

                        elif ("entropy" in self.measure_type) or ("mutual" in self.measure_type):
                            angle_shift += 1.
                            angle_shift[angle_shift == 0] = 1.
                            angle_log = np.log(np.abs(angle_shift))
                            max_min = (angle_log.max() - angle_log.min())
                            max_min = 1. if max_min == 0 else max_min
                            angle_minmax = (angle_log - angle_log.min()) / max_min
                            spectra[f, :, :, c] = 255 * angle_minmax

                        else:
                            max_min = (angle_shift.max() - angle_shift.min())
                            max_min = 1. if max_min == 0 else max_min
                            spectra[f, :, :, c] = (angle_shift - angle_shift.min()) / max_min

        elif "mag" in self.measure_type:
            for f in xrange(n_frames):
                for c in xrange(n_channels):

                    fft_frames = np.fft.fft2(frames[f, :, :, c].astype(np.float32))
                    mag = np.absolute(fft_frames)
                    mag_shift = np.fft.fftshift(mag)

                    if self.analize:
                        spectra[f, :, :, c] = mag_shift

                    else:

                        if "energy" in self.measure_type:
                            spectra[f, :, :, c] = np.power(mag_shift, pwr2)

                        elif ("entropy" in self.measure_type) or ("mutual" in self.measure_type):
                            mag_shift += 1.
                            mag_shift[mag_shift == 0] = 1.
                            mag_log = np.log(np.abs(mag_shift))
                            mag_minmax = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min())
                            spectra[f, :, :, c] = 255 * mag_minmax

                        else:
                            max_min = (mag_shift.max() - mag_shift.min())
                            max_min = 1. if max_min == 0 else max_min
                            spectra[f, :, :, c] = (mag_shift - mag_shift.min()) / max_min

        else:
            print "error: spectral_analysis(self, frames)"
            sys.exit(0)

        return spectra

    @staticmethod
    def compute_log(feats):

        arrs = feats.copy()
        arrs += 1.
        arrs[arrs <= 0] = 1.
        arrs = np.log(arrs)

        return arrs

    @staticmethod
    def compute_energy(cuboids):

        n_cuboids, cuboid_depth, cuboid_width, cuboid_width, n_channels = cuboids.shape
        energies = np.zeros((n_cuboids, cuboid_depth, n_channels), dtype=np.float)

        for n_cuboid in xrange(n_cuboids):
            for depth in xrange(cuboid_depth):
                for channel in xrange(n_channels):
                    energies[n_cuboid, depth, channel] = np.sum(cuboids[n_cuboid, depth, :, :, channel])

        return energies

    @staticmethod
    def shannon_entropy(counts_):

        # counts /= counts.sum()
        counts = counts_.copy()
        probs = counts[counts != 0]
        h = stats.entropy(probs, base=2)

        return h

    def compute_entropy(self, cuboids):
        # base = 2
        n_cuboids, cuboid_depth, cuboid_width, cuboid_width, n_channels = cuboids.shape
        entropies = np.zeros((n_cuboids, cuboid_depth, n_channels), dtype=np.float)

        for n_cuboid in xrange(n_cuboids):
            for depth in xrange(cuboid_depth):
                for channel in xrange(n_channels):
                    x = cuboids[n_cuboid, depth, :, :, channel].ravel()
                    counts = np.histogram(x, bins=256, range=(0, 255))[0]
                    entropies[n_cuboid, depth, channel] = self.shannon_entropy(counts)

        return entropies

    @staticmethod
    def mutual_info(x, y):
        # bins = 256
        c_xy = np.histogram2d(x, y, bins=(256, 256), range=((0, 255), (0, 255)))[0]
        mi = mutual_info_score(None, None, contingency=c_xy)

        return mi

    @staticmethod
    def compute_temporal_correlation(cuboids):

        np.seterr(all='raise')

        n_cuboids, cuboid_depth, cuboid_width, cuboid_width, n_channels = cuboids.shape
        corr = np.zeros((n_cuboids, cuboid_depth - 1, n_channels), dtype=np.float)

        for n_cuboid in xrange(n_cuboids):
            for depth in xrange(cuboid_depth - 1):
                for channel in xrange(n_channels):

                    x = cuboids[n_cuboid, depth, :, :, channel].ravel()
                    y = cuboids[n_cuboid, depth + 1, :, :, channel].ravel()

                    try:
                        corr[n_cuboid, depth, channel] = stats.pearsonr(x, y)[0]
                    except FloatingPointError:
                        corr[n_cuboid, depth, channel] = 0.
                        pass

        return corr

    def compute_temporal_mutual_info(self, cuboids):

        n_cuboids, cuboid_depth, cuboid_width, cuboid_width, n_channels = cuboids.shape
        mi = np.zeros((n_cuboids, cuboid_depth - 1, n_channels), dtype=np.float)

        for n_cuboid in xrange(n_cuboids):
            for depth in xrange(cuboid_depth - 1):
                for channel in xrange(n_channels):
                    x = cuboids[n_cuboid, depth, :, :, channel].ravel()
                    y = cuboids[n_cuboid, depth + 1, :, :, channel].ravel()
                    mi[n_cuboid, depth, channel] = self.mutual_info(x, y)

        return mi

    def get_roi(self, frames):

        n_frames, row, col, n_channels = frames.shape

        if self.roi == "centerframe":
            return frames[:, (row / 4):(row - row / 4), (col / 4):(col - col / 4), :]
        else:
            return frames

    def get_cuboid(self, frames):

        if self.debug:
            print "\t- getting cuboids ..."
            sys.stdout.flush()

        n_frames, n_rows, n_cols, n_channels = frames.shape
        cuboids = np.zeros((self.n_cuboids, self.cuboid_depth, self.cuboid_width, self.cuboid_width, n_channels),
                           dtype=frames.dtype)

        n_frames_rd = np.random.uniform(0, (n_frames - self.cuboid_depth - 1), size=self.n_cuboids)

        n_rows_rd = np.random.uniform(0, (n_rows - self.cuboid_width + 1), size=self.n_cuboids)

        n_cols_rd = np.random.uniform(0, (n_cols - self.cuboid_width + 1), size=self.n_cuboids)

        for n_cuboid in xrange(self.n_cuboids):
            x = n_frames_rd[n_cuboid]
            y = n_rows_rd[n_cuboid]
            z = n_cols_rd[n_cuboid]
            cuboids[n_cuboid, :, :, :, :] = frames[x:x + self.cuboid_depth, y:y + self.cuboid_width, z:z + self.cuboid_width, :]

        return cuboids

    def compute_measure(self, cuboids):

        if self.debug:
            print "\t- computing measures from cuboids ..."
            sys.stdout.flush()

        if "energy" in self.measure_type:
            measures = self.compute_energy(cuboids)

        elif "entropy" in self.measure_type:
            measures = self.compute_entropy(cuboids)

        elif "mutualinfo" in self.measure_type:
            measures = self.compute_temporal_mutual_info(cuboids)

        elif "correlation" in self.measure_type:
            measures = self.compute_temporal_correlation(cuboids)

        else:
            print 'error: compute_measure(freq_domain_frames)'
            sys.exit(0)

        return measures

    def save_low_level_features(self, output_fname, measures):

        if self.debug:
            print "\t- saving low level features ..."
            sys.stdout.flush()

        try:
            os.makedirs(os.path.dirname(output_fname))
        except OSError:
            pass

        n_cuboids, depth, channel = measures.shape

        if self.color:
            if self.flatten:
                measures = measures.reshape(n_cuboids, -1)
            else:
                measures = np.rollaxis(measures, 2, 1)
                measures = measures.reshape(-1, depth)
        else:
            measures = measures.reshape(n_cuboids, -1)

        np.save(output_fname, measures)

        # if measures_output is not None:
        #     measures_output = measures.copy()

        return True

    def save_bounding_box(self, output_fname, bounding_box):

        if self.debug:
            print "\t- saving low level features ..."
            sys.stdout.flush()

        try:
            os.makedirs(os.path.dirname(output_fname))
        except OSError:
            pass

        output_fname = output_fname.replace('.npy', '.face')
        np.savetxt(output_fname, bounding_box, fmt="%d", delimiter=" ")

        return True

    @staticmethod
    def face_detection():
        # input_video = video.reader(self.input_fname)
        # frames = input_video.load()
        # # n_frames, n_row, n_col, n_channels = frames.shape
        bounding_box = []
        # for i, face_image in enumerate(frames):
        #     bbox, quality = facedetect.detect_single_face(face_image)
        #     box = [val for val in (bbox.topleft + bbox.size)]
        #     if box:
        #         box = [i] + box
        #         bounding_box.append(box)
        return bounding_box

    def extract_low_level_features(self):

        if self.get_faceloc:

            rel_fname = os.path.relpath(self.input_fname, self.dataset_path)
            rel_fname = os.path.splitext(rel_fname)[0] + '.face'
            output_fname = os.path.join(self.output_path, rel_fname)

            if os.path.exists(output_fname):
                return True

            bounding_box = self.face_detection()
            self.save_bounding_box(output_fname, bounding_box)

        else:

            rel_fname = os.path.relpath(self.input_fname, self.dataset_path)
            rel_fname = os.path.splitext(rel_fname)[0] + '.npy'
            output_fname = os.path.join(self.output_path, rel_fname)

            if os.path.exists(output_fname):
                return True

            frames = self.load_video()

            if self.only_face:
                faces = self.extracting_face_region(frames)
                frames = faces

            residual_noise_frames = self.residual_noise_video(frames)
            del frames

            freq_domain_frames = self.spectral_analysis(residual_noise_frames)
            del residual_noise_frames

            freq_domain_frames = self.get_roi(freq_domain_frames)
            cuboids = self.get_cuboid(freq_domain_frames)
            del freq_domain_frames

            measures = self.compute_measure(cuboids)
            del cuboids

            # import pdb
            # pdb.set_trace()

            self.save_low_level_features(output_fname, measures)

            # import pdb
            # pdb.set_trace()

        return True

    # def show_in_axes(self, frames, residual_noise_frames, mag_spec_norm, phase_spec_norm):
    #
    #     for i in range(frames.shape[0]):
    #         # cmap = "jet"
    #
    #         # # Four axes, returned as a 2-d array
    #         # f, axarr = plt.subplots(2, 2)
    #
    #         # axarr[0, 0].imshow(frames[i], cmap=cmap, vmax=255, vmin=0)
    #         # axarr[0, 0].set_title('Frame %.3d'% (i), fontsize=16)
    #         # # axarr[0, 0].cbar_axes[0].colorbar(im0)
    #
    #         # axarr[0, 1].imshow(residual_noise_frames[i], cmap=cmap, vmax=255, vmin=0)
    #         # axarr[0, 1].set_title('Residual Noise', fontsize=16)
    #         # # grid.cbar_axes[1].colorbar(im1)
    #
    #         # axarr[1, 0].imshow(mag_spec_norm[i], cmap=cmap, vmax=255, vmin=0)
    #         # axarr[1, 0].set_title('Mag Spectrum', fontsize=16)
    #         # # grid.cbar_axes[2].colorbar(im2)
    #
    #         # im = axarr[1, 1].imshow(phase_spec_norm[i], cmap=cmap, vmax=255, vmin=0)
    #         # axarr[1, 1].set_title('Phase Spectrum', fontsize=16)
    #         # # grid.cbar_axes[3].colorbar(im3)
    #
    #         # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #         # for ax in axarr.flat:
    #         #     plt.setp(ax.get_xticklabels(), visible=False)
    #         #     plt.setp(ax.get_yticklabels(), visible=False)
    #
    #         # plt.tight_layout(pad=3.)
    #
    #         # # Make an axis for the colorbar on the right side
    #         # cax = f.add_axes([0.92, 0.08, 0.025, 0.8])
    #         # f.colorbar(im, cax=cax)
    #
    #         # fname = "{0}/{1:03d}.png".format(output_fname, i)
    #         # plt.savefig(fname)
    #         pass

    @staticmethod
    def show_in_grid(frames, residual_noise_frames, mag_spec_norm, phase_spec_norm, output_fname):

        for i in range(frames.shape[0]):

            fig = plt.figure(figsize=(10, 10), dpi=100)

            cmap = "jet"

            grid = AxesGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5, label_mode="1", share_all=True,
                            cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad="2%")

            im0 = grid[0].imshow(frames[i], cmap=cmap, vmax=255, vmin=0)
            grid.cbar_axes[0].colorbar(im0), grid[0].set_title('Frame %.3d' % i, fontsize=16)

            im1 = grid[1].imshow(residual_noise_frames[i], cmap=cmap, vmax=255, vmin=0)
            grid.cbar_axes[1].colorbar(im1), grid[1].set_title('Residual Noise', fontsize=16)

            im2 = grid[2].imshow(mag_spec_norm[i], cmap=cmap, vmax=255, vmin=0)
            grid.cbar_axes[2].colorbar(im2), grid[2].set_title('Mag Spectrum', fontsize=16)

            im3 = grid[3].imshow(phase_spec_norm[i], cmap=cmap, vmax=255, vmin=0)
            grid.cbar_axes[3].colorbar(im3), grid[3].set_title('Phase Spectrum', fontsize=16)

            for ax in grid:
                ax.set_yticks([]), ax.set_xticks([])

            plt.tight_layout(pad=2.5)

            fname = "{0}/{1:03d}.png".format(output_fname, i)
            plt.savefig(fname)

    @staticmethod
    def save_figure_as_image(file_name, fig=None, **kwargs):

        fig_size = fig.get_size_inches()
        w, h = fig_size[0], fig_size[1]
        fig.patch.set_alpha(0)
        if 'orig_size' in kwargs:  # Aspect ratio scaling if required
            w, h = kwargs['orig_size']
            w2, h2 = fig_size[0], fig_size[1]
            fig.set_size_inches([(w2 / w) * w, (w2 / w) * h])
            fig.set_dpi((w2 / w) * fig.get_dpi())
        a = fig.gca()
        a.set_frame_on(False)
        a.set_xticks([])
        a.set_yticks([])
        plt.axis('off')
        plt.xlim(0, h)
        plt.ylim(w, 0)
        fig.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0)

    def analize_one_sample(self):

        self.color = False
        # self.only_face = True
        self.measure_type = 3

        frames = self.load_video()

        # if self.only_face:
        #     faces = self.extracting_face_region(frames)
        #     frames = faces

        residual_noise_frames = self.residual_noise_video(frames)
        mag_spec = self.spectral_analysis(residual_noise_frames)
        mag_spec = self.compute_log(mag_spec)
        mag_spec_norm = self.normalization(mag_spec)

        # padded = np.lib.pad(mag_spec_norm, 2, padwithtens)

        mag_spec_norm = np.squeeze(mag_spec_norm)
        border_width = 5
        paddeds = []
        for mag in mag_spec_norm:
            n_rows, n_cols = mag.shape
            padded = np.zeros((n_rows + (2 * border_width), n_cols + (2 * border_width)), dtype=mag.dtype)
            padded[border_width:(border_width + n_rows), border_width:(border_width + n_cols)] = mag[:, :]
            paddeds.append(padded)
        img = mosaic(10, np.array(paddeds))

        h, w = img.shape[0], img.shape[1]

        rel_fname = os.path.relpath(self.input_fname, self.dataset_path)
        rel_fname = os.path.splitext(rel_fname)[0]
        fname = "working/ritmo{0}.png".format(rel_fname)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        # ax.show()
        # fig.savefig(fname)

        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass

        self.save_figure_as_image(fname, plt.gcf(), orig_size=(h, w))

        # bob.io.base.save(img.astype('uint8'), fname)
        # plt.savefig(fname)

        # return True

        matplotlib.rcParams['figure.max_open_warning'] = 8

        rel_fname = os.path.relpath(self.input_fname, self.dataset_path)
        rel_fname = os.path.splitext(rel_fname)[0]
        output_fname = os.path.join(self.output_path, rel_fname)

        try:
            os.makedirs(output_fname)
        except OSError:
            pass

        # self.color = True
        # frames_color = self.load_video()[:4]

        # kernel, sigma = self.kernel_width, self.sigma
        # self.kernel_width, self.sigma = 5, 2
        # n_frames, n_row, n_col, n_channels = frames_color.shape
        # filtered_frames = np.zeros(frames_color.shape, dtype=frames_color.dtype)
        # for f in xrange(n_frames):
        #     for c in xrange(n_channels):
        #         filtered_frames[f,:,:,c] = self.gaussian_filter(frames_color[f,:,:,c])
        # self.kernel_width, self.sigma = kernel, sigma
        # frames_color = filtered_frames.copy()

        # self.color = False
        # frames = self.load_video()[:4]

        # kernel, sigma = self.kernel_width, self.sigma
        # self.kernel_width, self.sigma = 5, 2
        # n_frames, n_row, n_col, n_channels = frames.shape
        # filtered_frames = np.zeros(frames.shape, dtype=frames.dtype)
        # for f in xrange(n_frames):
        #     for c in xrange(n_channels):
        #         filtered_frames[f,:,:,c] = self.gaussian_filter(frames[f,:,:,c])
        # self.kernel_width, self.sigma = kernel, sigma
        # frames = filtered_frames.copy()

        self.color = False

        frames = self.load_video()

        if self.only_face:
            face_boxes = self.__detecting_faces(frames)
            frames, boxes_adjusted = self.__normalized_faces(face_boxes, frames)

        # import pdb; pdb.set_trace()
        residual_noise_frames = self.residual_noise_video(frames)

        self.measure_type = 1
        phase_spec = self.spectral_analysis(residual_noise_frames)
        phase_spec_norm = self.normalization(phase_spec)

        self.measure_type = 3
        mag_spec = self.spectral_analysis(residual_noise_frames)
        mag_spec = self.compute_log(mag_spec)
        mag_spec_norm = self.normalization(mag_spec)

        if not self.color:
            frames = np.squeeze(frames, axis=(3,))
            residual_noise_frames = np.squeeze(residual_noise_frames, axis=(3,))
            phase_spec_norm = np.squeeze(phase_spec_norm, axis=(3,))
            mag_spec_norm = np.squeeze(mag_spec_norm, axis=(3,))

        self.show_in_grid(frames, residual_noise_frames, mag_spec_norm, phase_spec_norm, output_fname)
        # exit(0)

        # n_frames, n_rows, n_cols = frames.shape[:3]
        # cmap = cm.jet

        # for i in range(n_frames):

        #     fig1 = plt.figure(figsize=(14,4), facecolor='w', dpi=150)

        #     # fig1 = plt.figure(figsize=(8,4), facecolor='w', dpi=150)

        #     ax1 = fig1.add_subplot(141)
        #     im1 = ax1.imshow(frames[i], cmap=cmap, vmax=255, vmin=0)
        #     ax1.set_title('Frame %.3d'% (i), fontsize=16)
        #     plt.setp(ax1.get_xticklabels(), visible=False)
        #     plt.setp(ax1.get_yticklabels(), visible=False)
        #     ax1.figure.colorbar(im1, orientation='vertical', shrink=0.7)

        #     ax2 = fig1.add_subplot(142)
        #     im2 = ax2.imshow(residual_noise_frames[i], cmap=cmap, vmax=255, vmin=0)
        #     ax2.set_title('Residual Noise', fontsize=16)
        #     plt.setp(ax2.get_xticklabels(), visible=False)
        #     plt.setp(ax2.get_yticklabels(), visible=False)
        #     ax1.figure.colorbar(im2, orientation='vertical', shrink=0.7)

        #     ax3 = fig1.add_subplot(143)
        #     im3 = ax3.imshow(mag_spec_norm[i], cmap=cmap, vmax=255, vmin=0)
        #     ax3.set_title('Mag Spectrum', fontsize=16)
        #     plt.setp(ax3.get_xticklabels(), visible=False)
        #     ax3.figure.colorbar(im3, orientation='vertical', shrink=0.7)

        #     ax4 = fig1.add_subplot(144)
        #     im4 = ax4.imshow(phase_spec_norm[i], cmap=cmap, vmax=255, vmin=0)
        #     ax4.set_title('Phase Spectrum', fontsize=16)
        #     plt.setp(ax4.get_xticklabels(), visible=False)
        #     ax4.figure.colorbar(im4, orientation='vertical', shrink=0.7)

        #     fname = "{0}/{1:03d}_flat.png".format(output_fname, i)

        #     plt.tight_layout(pad=3.)
        #     plt.savefig(fname)

        # fname = "{0}/{1:03d}_frame.png".format(output_fname, i)
        # misc.imsave(fname, frames_color[i])

        # fig = plt.figure(figsize=(4,6), facecolor='w', dpi=150)

        # data = mag_spec_norm[i][::1,::1]
        # ax5 = fig.add_subplot(211,projection='3d')
        # x,y = np.mgrid[:data.shape[0],:data.shape[1]]
        # im5 = ax5.plot_surface(x,y,data,
        #     cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=True)
        # # ax5.set_title('Mag Spectrum (Frame %.3d)'% (i), fontsize=12)
        # ax5.set_zlim3d(0,255)
        # plt.setp(ax5.get_xticklabels(), visible=False)
        # plt.setp(ax5.get_yticklabels(), visible=False)
        # plt.setp(ax5.get_zticklabels(), visible=False)
        # ax5.figure.colorbar(im5, orientation='vertical', shrink=0.7)
        # im5.set_clim([200,255])

        # data = phase_spec_norm[i][::1,::1]
        # ax6 = fig.add_subplot(212,projection='3d')
        # x,y = np.mgrid[:data.shape[0],:data.shape[1]]
        # im6 = ax6.plot_surface(x,y,data,
        #     cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=True)
        # # ax6.set_title('Phase Spectrum (Frame %.3d)'% (i), fontsize=12)
        # ax6.set_zlim3d(0,255)
        # plt.setp(ax6.get_xticklabels(), visible=False)
        # plt.setp(ax6.get_yticklabels(), visible=False)
        # plt.setp(ax6.get_zticklabels(), visible=False)
        # ax6.figure.colorbar(im6, orientation='vertical', shrink=0.7)
        # im6.set_clim([200,255])

        # fname = "{0}/{1:03d}_surface.png".format(output_fname, i)
        # plt.tight_layout(pad=3.)
        # plt.savefig(fname)

    def run(self):
        """ Title
        Description
        Args:
        Returns:
        Raises:
        """

        np.random.RandomState(self.seed)
        np.random.seed(self.seed)

        if not self.analize:
            self.extract_low_level_features()
        else:
            self.analize_one_sample()

        return True


if __name__ == "__main__":
    start = get_time()

    LowLevelFeatures(dataset_path="../../tests/data", output_path="../../tests/output",
                     input_fname="../../tests/data/test_1.mov", roi=1, measure_type=0,
                     filter_type="gauss", frame_numbers=10).run()

    total_time_elapsed(start, get_time())
