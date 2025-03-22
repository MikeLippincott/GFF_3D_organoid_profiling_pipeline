import cucim.skimage.measure
import cupy
import cupyx
import cupyx.scipy.ndimage
import scipy
import skimage


def linear_costes_gpu(fi, si, scale_max=255, fast_costes="Accurate"):
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0.
    """
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    xvar = cupy.var(fi[non_zero], axis=0, ddof=1)
    yvar = cupy.var(si[non_zero], axis=0, ddof=1)

    xmean = cupy.mean(fi[non_zero], axis=0)
    ymean = cupy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = cupy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + cupy.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(fi.max(), si.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    fi_max = fi.max()
    si_max = si.max()

    # Initialise without a threshold
    fi = fi.get()
    si = si.get()
    costReg, _ = scipy.stats.pearsonr(fi, si)
    fi = cupy.array(fi)
    si = cupy.array(si)
    costReg = cupy.array(costReg)
    thr_fi_c = i
    thr_si_c = (a * i) + b
    while i > fi_max and (a * i) + b > si_max:
        i -= i_step
    while i > i_step:
        thr_fi_c = i
        thr_si_c = (a * i) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := cupy.count_nonzero(combt)) != num_true:
                # convert to numpy array since this module does not exist in cupy
                fi = fi.get()
                si = si.get()
                combt = combt.get()
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                # convert back to a cupy array
                fi = cupy.array(fi)
                si = cupy.array(si)
                combt = cupy.array(combt)

                num_true = positives

            if costReg <= 0:
                break
            elif fast_costes == "Accurate" or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_fi_c, thr_si_c


def bisection_costes_gpu(fi, si, scale_max=255):
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (fi > 0) | (si > 0)
    xvar = cupy.var(fi[non_zero], axis=0, ddof=1)
    yvar = cupy.var(si[non_zero], axis=0, ddof=1)

    xmean = cupy.mean(fi[non_zero], axis=0)
    ymean = cupy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = cupy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + cupy.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        if cupy.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                fi = fi.get()
                si = si.get()
                combt = combt.get()
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                # convert back to a cupy array
                fi = cupy.array(fi)
                si = cupy.array(si)
                combt = cupy.array(combt)
                costReg = cupy.array(costReg)
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c


def select_objects_from_label_gpu(label_image, object_ids):
    label_image = label_image.copy()
    label_image[label_image != object_ids] = 0
    return label_image


def expand_box_gpu(min_coor, max_coord, current_min, current_max, expand_by):
    if max_coord - min_coor - (current_max - current_min) < expand_by:
        return ValueError("Cannot expand box by the requested amount")
    while expand_by > 0:
        if current_min > min_coor:
            current_min -= 1
            expand_by -= 1
        elif current_max < max_coord:
            current_max += 1
            expand_by -= 1

    return current_min, current_max


def new_crop_border_gpu(bbox1, bbox2, image):
    i1z1, i1y1, i1x1, i1z2, i1y2, i1x2 = bbox1
    i2z1, i2y1, i2x1, i2z2, i2y2, i2x2 = bbox2
    z_range1 = i1z2 - i1z1
    y_range1 = i1y2 - i1y1
    x_range1 = i1x2 - i1x1
    z_range2 = i2z2 - i2z1
    y_range2 = i2y2 - i2y1
    x_range2 = i2x2 - i2x1
    z_diff = cupy.abs(z_range1 - z_range2)
    y_diff = cupy.abs(y_range1 - y_range2)
    x_diff = cupy.abs(x_range1 - x_range2)
    min_z_coord = 0
    max_z_coord = image.shape[0]
    min_y_coord = 0
    max_y_coord = image.shape[1]
    min_x_coord = 0
    max_x_coord = image.shape[2]
    if z_range1 < z_range2:
        i1z1, i1z2 = expand_box_gpu(
            min_coor=min_z_coord,
            max_coord=max_z_coord,
            current_min=i1z1,
            current_max=i1z2,
            expand_by=z_diff,
        )
    elif z_range1 > z_range2:
        i2z1, i2z2 = expand_box_gpu(
            min_coor=min_z_coord,
            max_coord=max_z_coord,
            current_min=i2z1,
            current_max=i2z2,
            expand_by=z_diff,
        )
    if y_range1 < y_range2:
        i1y1, i1y2 = expand_box_gpu(
            min_coor=min_y_coord,
            max_coord=max_y_coord,
            current_min=i1y1,
            current_max=i1y2,
            expand_by=y_diff,
        )
    elif y_range1 > y_range2:
        i2y1, i2y2 = expand_box_gpu(
            min_coor=min_y_coord,
            max_coord=max_y_coord,
            current_min=i2y1,
            current_max=i2y2,
            expand_by=y_diff,
        )
    if x_range1 < x_range2:
        i1x1, i1x2 = expand_box_gpu(
            min_coor=min_x_coord,
            max_coord=max_x_coord,
            current_min=i1x1,
            current_max=i1x2,
            expand_by=x_diff,
        )
    elif x_range1 > x_range2:
        i2x1, i2x2 = expand_box_gpu(
            min_coor=min_x_coord,
            max_coord=max_x_coord,
            current_min=i2x1,
            current_max=i2x2,
            expand_by=x_diff,
        )
    return (i1z1, i1y1, i1x1, i1z2, i1y2, i1x2), (i2z1, i2y1, i2x1, i2z2, i2y2, i2x2)


# crop the image to the bbox of the mask
def crop_3D_image_gpu(image, bbox):
    z1, y1, x1, z2, y2, x2 = bbox
    return image[z1:z2, y1:y2, x1:x2]


def prepare_two_images_for_colocalization_gpu(
    label_object1, label_object2, image_object1, image_object2, object_id1, object_id2
):
    label_object1 = cupy.array(label_object1)
    label_object2 = cupy.array(label_object2)
    object_id1 = cupy.array(object_id1)
    object_id2 = cupy.array(object_id2)
    label_object1 = select_objects_from_label_gpu(label_object1, object_id1)
    label_object2 = select_objects_from_label_gpu(label_object2, object_id2)
    # get the image bbox
    props_image1 = cucim.skimage.measure.regionprops_table(
        label_object1, properties=["bbox"]
    )
    bbox_image1 = (
        props_image1["bbox-0"][0].item(),
        props_image1["bbox-1"][0].item(),
        props_image1["bbox-2"][0].item(),
        props_image1["bbox-3"][0].item(),
        props_image1["bbox-4"][0].item(),
        props_image1["bbox-5"][0].item(),
    )

    props_image2 = cucim.skimage.measure.regionprops_table(
        label_object2, properties=["bbox"]
    )
    bbox_image2 = (
        props_image2["bbox-0"][0].item(),
        props_image2["bbox-1"][0].item(),
        props_image2["bbox-2"][0].item(),
        props_image2["bbox-3"][0].item(),
        props_image2["bbox-4"][0].item(),
        props_image2["bbox-5"][0].item(),
    )
    new_bbox1, new_bbox2 = new_crop_border_gpu(bbox_image1, bbox_image2, image_object1)

    cropped_image_1 = cupy.asarray(crop_3D_image_gpu(image_object1, new_bbox1))
    cropped_image_2 = cupy.asarray(crop_3D_image_gpu(image_object2, new_bbox2))

    return cropped_image_1, cropped_image_2


def measure_3D_colocalization_gpu(
    cropped_image_1, cropped_image_2, thr=15, fast_costes="Accurate"
):
    results = {}
    thr = 15
    ################################################################################################
    # Calculate the correlation coefficient between the two images
    ################################################################################################
    mean1 = cupyx.scipy.ndimage.mean(cropped_image_1)
    mean2 = cupyx.scipy.ndimage.mean(cropped_image_2)
    std1 = cupy.sqrt(cupyx.scipy.ndimage.sum((cropped_image_1 - mean1) ** 2))
    std2 = cupy.sqrt(cupyx.scipy.ndimage.sum((cropped_image_2 - mean2) ** 2))
    x = cropped_image_1 - mean1
    y = cropped_image_2 - mean2
    corr = cupyx.scipy.ndimage.sum(x * y / (std1 * std2))

    ################################################################################################
    # Calculate the Manders' coefficients
    ################################################################################################

    # Threshold as percentage of maximum intensity of objects in each channel
    tff = (thr / 100) * cupyx.scipy.ndimage.maximum(cropped_image_1)
    tss = (thr / 100) * cupyx.scipy.ndimage.maximum(cropped_image_2)

    combined_thresh = (cropped_image_1 >= tff) & (cropped_image_2 >= tss)

    fi_thresh = cropped_image_1[combined_thresh]
    si_thresh = cropped_image_2[combined_thresh]
    tot_fi_thr = cupyx.scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= tff],
    )
    tot_si_thr = cupyx.scipy.ndimage.sum(cropped_image_2[cropped_image_2 >= tss])

    M1 = cupyx.scipy.ndimage.sum(fi_thresh) / cupy.array(tot_fi_thr)
    M2 = cupyx.scipy.ndimage.sum(si_thresh) / cupy.array(tot_si_thr)

    ################################################################################################
    # Calculate the overlap coefficient
    ################################################################################################

    fpsq = cupyx.scipy.ndimage.sum(
        cropped_image_1[combined_thresh] ** 2,
    )
    spsq = cupyx.scipy.ndimage.sum(
        cropped_image_2[combined_thresh] ** 2,
    )
    pdt = cupy.sqrt(cupy.array(fpsq) * cupy.array(spsq))
    overlap = (
        cupyx.scipy.ndimage.sum(
            cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
        )
        / pdt
    )
    K1 = cupyx.scipy.ndimage.sum(
        cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
    ) / (cupy.array(fpsq))
    K2 = cupyx.scipy.ndimage.sum(
        cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
    ) / (cupy.array(spsq))

    ################################################################################################
    # Calculate the Costes' coefficient
    ################################################################################################

    # Orthogonal Regression for Costes' automated threshold
    if cupy.max(cropped_image_1) > 255 or cupy.max(cropped_image_2) > 255:
        scale = 65535
    else:
        scale = 255

    if fast_costes == "Accurate":
        thr_fi_c, thr_si_c = bisection_costes_gpu(
            cropped_image_1, cropped_image_2, scale
        )
    else:
        thr_fi_c, thr_si_c = linear_costes_gpu(
            cropped_image_1, cropped_image_2, scale, fast_costes
        )

    # Costes' thershold for entire image is applied to each object
    fi_above_thr = cropped_image_1 > thr_fi_c
    si_above_thr = cropped_image_2 > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = cropped_image_1[combined_thresh_c]
    si_thresh_c = cropped_image_2[combined_thresh_c]

    tot_fi_thr_c = cupyx.scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= thr_fi_c],
    )

    tot_si_thr_c = cupyx.scipy.ndimage.sum(
        cropped_image_2[cropped_image_2 >= thr_si_c],
    )
    C1 = cupyx.scipy.ndimage.sum(fi_thresh_c) / cupy.array(tot_fi_thr_c)
    C2 = cupyx.scipy.ndimage.sum(si_thresh_c) / cupy.array(tot_si_thr_c)

    ################################################################################################
    # write the results to the output dictionary
    ################################################################################################

    results["MEAN.CORRELATION.COEFF"] = cupy.mean(corr).get()
    results["MEDIAN.CORRELATION.COEFF"] = cupy.median(corr).get()
    results["MIN.CORRELATION.COEFF"] = cupy.min(corr).get()
    results["MAX.CORRELATION.COEFF"] = cupy.max(corr).get()
    results["MEAN.MANDERS.COEFF.M1"] = cupy.mean(M1).get()
    results["MEDIAN.MANDERS.COEFF.M1"] = cupy.median(M1).get()
    results["MIN.MANDERS.COEFF.M1"] = cupy.min(M1).get()
    results["MAX.MANDERS.COEFF.M1"] = cupy.max(M1).get()
    results["MEAN.MANDERS.COEFF.M2"] = cupy.mean(M2).get()
    results["MEDIAN.MANDERS.COEFF.M2"] = cupy.median(M2).get()
    results["MIN.MANDERS.COEFF.M2"] = cupy.min(M2).get()
    results["MAX.MANDERS.COEFF.M2"] = cupy.max(M2).get()
    results["MEAN.OVERLAP.COEFF"] = cupy.mean(overlap).get()
    results["MEDIAN.OVERLAP.COEFF"] = cupy.median(overlap).get()
    results["MIN.OVERLAP.COEFF"] = cupy.min(overlap).get()
    results["MAX.OVERLAP.COEFF"] = cupy.max(overlap).get()
    results["MEAN.K1"] = cupy.mean(K1).get()
    results["MEDIAN.K1"] = cupy.median(K1).get()
    results["MIN.K1"] = cupy.min(K1).get()
    results["MAX.K1"] = cupy.max(K1).get()
    results["MEAN.K2"] = cupy.mean(K2).get()
    results["MEDIAN.K2"] = cupy.median(K2).get()
    results["MIN.K2"] = cupy.min(K2).get()
    results["MAX.K2"] = cupy.max(K2).get()
    results["MEAN.MANDERS.COEFF.COSTES.M1"] = cupy.mean(C1).get()
    results["MEDIAN.MANDERS.COEFF.COSTES.M1"] = cupy.median(C1).get()
    results["MIN.MANDERS.COEFF.COSTES.M1"] = cupy.min(C1).get()
    results["MAX.MANDERS.COEFF.COSTES.M1"] = cupy.max(C1).get()
    results["MEAN.MANDERS.COEFF.COSTES.M2"] = cupy.mean(C2).get()
    results["MEDIAN.MANDERS.COEFF.COSTES.M2"] = cupy.median(C2).get()
    results["MIN.MANDERS.COEFF.COSTES.M2"] = cupy.min(C2).get()
    results["MAX.MANDERS.COEFF.COSTES.M2"] = cupy.max(C2).get()
    return results
