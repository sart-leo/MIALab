"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import numpy as np


class ImageNormalizationLog(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a Log normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The Log-normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        regularization = 0.00000001
        img_arr = np.log(img_arr + regularization)

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalizationLog:\n' \
            .format(self=self)


class ImageNormalizationZScore(pymia_fltr.Filter):
    """Represents a Z-score normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # Normalize image subtracting the mean intensity and dividing by the standard deviation
        img_arr = (img_arr - np.mean(img_arr))/np.std(img_arr)

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

            Returns:
        str: String representation.
        """
        return 'ImageNormalizationZScore:\n' \
            .format(self=self)


class ImageNormalizationMinMax(pymia_fltr.Filter):
    """Represents a MinMax normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

            Returns:
        str: String representation.
        """
        return 'ImageNormalizationMinMax:\n' \
            .format(self=self)


class ImageNormalizationClipping(pymia_fltr.Filter):
    """Represents a Clipping normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        img_arr[img_arr > 0.95 * np.max(img_arr)] = 0.95 * np.max(img_arr)
        img_arr[img_arr < 1.05 * np.min(img_arr)] = 1.05 * np.min(img_arr)

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

            Returns:
        str: String representation.
        """
        return 'ImageNormalizationClipping:\n' \
            .format(self=self)


class CMeansParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the CMeansParameters

        Args:
            img_mask (sitk.Image): The White Matter mask image.
        """
        self.img_mask = img_mask


class ImageNormalizationCMeans(pymia_fltr.Filter):
    """Represents a Clipping normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: CMeansParameters = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask
        mask = sitk.Resample(mask, image)

        img_arr = sitk.GetArrayFromImage(image)
        mask_arr = sitk.GetArrayFromImage(mask)
        mask_arr[mask_arr != 1] = 0

        mean = np.sum(img_arr[mask_arr == 1])/np.sum(mask_arr[mask_arr == 1])
        cValue = 1000
        img_arr = cValue*img_arr/mean

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

            Returns:
        str: String representation.
        """
        return 'ImageNormalizationCmeans:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The masked image.
        """
        mask = params.img_mask  # the brain mask

        sitk.Mask(image, mask)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """
        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different
        if is_ground_truth:
            # apply transformation to ground truth and brain mask using nearest neighbor interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkNearestNeighbor, 0,
                                  image.GetPixelIDValue())
        else:
            # apply transformation to T1w and T2w images using linear interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkLinear, 0.0,
                                  image.GetPixelIDValue())

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
