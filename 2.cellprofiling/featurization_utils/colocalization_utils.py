import numpy
import scipy.ndimage
import skimage


def linear_costes(fi, si, scale_max=255, fast_costes="Accurate"):
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0.
    """
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
    yvar = numpy.var(si[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(fi[non_zero], axis=0)
    ymean = numpy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(fi.max(), si.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    fi_max = fi.max()
    si_max = si.max()

    # Initialise without a threshold
    costReg, _ = scipy.stats.pearsonr(fi, si)
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
            if (positives := numpy.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
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


def bisection_costes(fi, si, scale_max=255):
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (fi > 0) | (si > 0)
    xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
    yvar = numpy.var(si[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(fi[non_zero], axis=0)
    ymean = numpy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
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
        if numpy.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
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


def select_objects_from_label(label_image, object_ids):
    label_image = label_image.copy()
    label_image[label_image != object_ids] = 0
    return label_image


def expand_box(min_coor, max_coord, current_min, current_max, expand_by):
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


def new_crop_border(bbox1, bbox2, image):
    i1z1, i1y1, i1x1, i1z2, i1y2, i1x2 = bbox1
    i2z1, i2y1, i2x1, i2z2, i2y2, i2x2 = bbox2
    z_range1 = i1z2 - i1z1
    y_range1 = i1y2 - i1y1
    x_range1 = i1x2 - i1x1
    z_range2 = i2z2 - i2z1
    y_range2 = i2y2 - i2y1
    x_range2 = i2x2 - i2x1
    z_diff = numpy.abs(z_range1 - z_range2)
    y_diff = numpy.abs(y_range1 - y_range2)
    x_diff = numpy.abs(x_range1 - x_range2)
    min_z_coord = 0
    max_z_coord = image.shape[0]
    min_y_coord = 0
    max_y_coord = image.shape[1]
    min_x_coord = 0
    max_x_coord = image.shape[2]
    if z_range1 < z_range2:
        i1z1, i1z2 = expand_box(
            min_coor=min_z_coord,
            max_coord=max_z_coord,
            current_min=i1z1,
            current_max=i1z2,
            expand_by=z_diff,
        )
    elif z_range1 > z_range2:
        i2z1, i2z2 = expand_box(
            min_coor=min_z_coord,
            max_coord=max_z_coord,
            current_min=i2z1,
            current_max=i2z2,
            expand_by=z_diff,
        )
    if y_range1 < y_range2:
        i1y1, i1y2 = expand_box(
            min_coor=min_y_coord,
            max_coord=max_y_coord,
            current_min=i1y1,
            current_max=i1y2,
            expand_by=y_diff,
        )
    elif y_range1 > y_range2:
        i2y1, i2y2 = expand_box(
            min_coor=min_y_coord,
            max_coord=max_y_coord,
            current_min=i2y1,
            current_max=i2y2,
            expand_by=y_diff,
        )
    if x_range1 < x_range2:
        i1x1, i1x2 = expand_box(
            min_coor=min_x_coord,
            max_coord=max_x_coord,
            current_min=i1x1,
            current_max=i1x2,
            expand_by=x_diff,
        )
    elif x_range1 > x_range2:
        i2x1, i2x2 = expand_box(
            min_coor=min_x_coord,
            max_coord=max_x_coord,
            current_min=i2x1,
            current_max=i2x2,
            expand_by=x_diff,
        )
    return (i1z1, i1y1, i1x1, i1z2, i1y2, i1x2), (i2z1, i2y1, i2x1, i2z2, i2y2, i2x2)


# crop the image to the bbox of the mask
def crop_3D_image(image, bbox):
    z1, y1, x1, z2, y2, x2 = bbox
    return image[z1:z2, y1:y2, x1:x2]


def prepare_two_images_for_colocalization(
    label_object1, label_object2, image_object1, image_object2, object_id1, object_id2
):
    label_object1 = select_objects_from_label(label_object1, object_id1)
    label_object2 = select_objects_from_label(label_object2, object_id2)
    # get the image bbox
    props_image1 = skimage.measure.regionprops_table(label_object1, properties=["bbox"])
    bbox_image1 = (
        props_image1["bbox-0"][0],
        props_image1["bbox-1"][0],
        props_image1["bbox-2"][0],
        props_image1["bbox-3"][0],
        props_image1["bbox-4"][0],
        props_image1["bbox-5"][0],
    )

    props_image2 = skimage.measure.regionprops_table(label_object2, properties=["bbox"])
    bbox_image2 = (
        props_image2["bbox-0"][0],
        props_image2["bbox-1"][0],
        props_image2["bbox-2"][0],
        props_image2["bbox-3"][0],
        props_image2["bbox-4"][0],
        props_image2["bbox-5"][0],
    )

    new_bbox1, new_bbox2 = new_crop_border(bbox_image1, bbox_image2, image_object1)

    cropped_image_1 = crop_3D_image(image_object1, new_bbox1)
    cropped_image_2 = crop_3D_image(image_object2, new_bbox2)
    return cropped_image_1, cropped_image_2


def measure_3D_colocalization(
    cropped_image_1, cropped_image_2, thr=15, fast_costes="Accurate"
):
    results = {}
    thr = 15
    ################################################################################################
    # Calculate the correlation coefficient between the two images
    ################################################################################################
    mean1 = scipy.ndimage.mean(cropped_image_1, 1)
    mean2 = scipy.ndimage.mean(cropped_image_2, 1)
    std1 = numpy.sqrt(scipy.ndimage.sum((cropped_image_1 - mean1) ** 2))
    std2 = numpy.sqrt(scipy.ndimage.sum((cropped_image_2 - mean2) ** 2))
    x = cropped_image_1 - mean1
    y = cropped_image_2 - mean2
    corr = scipy.ndimage.sum(x * y / (std1 * std2))

    ################################################################################################
    # Calculate the Manders' coefficients
    ################################################################################################

    # Threshold as percentage of maximum intensity of objects in each channel
    tff = (thr / 100) * scipy.ndimage.maximum(cropped_image_1)
    tss = (thr / 100) * scipy.ndimage.maximum(cropped_image_2)

    combined_thresh = (cropped_image_1 >= tff) & (cropped_image_2 >= tss)

    fi_thresh = cropped_image_1[combined_thresh]
    si_thresh = cropped_image_2[combined_thresh]
    tot_fi_thr = scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= tff],
    )
    tot_si_thr = scipy.ndimage.sum(cropped_image_2[cropped_image_2 >= tss])

    M1 = scipy.ndimage.sum(fi_thresh) / numpy.array(tot_fi_thr)
    M2 = scipy.ndimage.sum(si_thresh) / numpy.array(tot_si_thr)

    ################################################################################################
    # Calculate the overlap coefficient
    ################################################################################################

    fpsq = scipy.ndimage.sum(
        cropped_image_1[combined_thresh] ** 2,
    )
    spsq = scipy.ndimage.sum(
        cropped_image_2[combined_thresh] ** 2,
    )
    pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))
    overlap = (
        scipy.ndimage.sum(
            cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
        )
        / pdt
    )
    K1 = scipy.ndimage.sum(
        cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
    ) / (numpy.array(fpsq))
    K2 = scipy.ndimage.sum(
        cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
    ) / (numpy.array(spsq))

    ################################################################################################
    # Calculate the Costes' coefficient
    ################################################################################################

    # Orthogonal Regression for Costes' automated threshold
    if numpy.max(cropped_image_1) > 255 or numpy.max(cropped_image_2) > 255:
        scale = 65535
    else:
        scale = 255

    if fast_costes == "Accurate":
        thr_fi_c, thr_si_c = bisection_costes(cropped_image_1, cropped_image_2, scale)
    else:
        thr_fi_c, thr_si_c = linear_costes(
            cropped_image_1, cropped_image_2, scale, fast_costes
        )

    # Costes' thershold for entire image is applied to each object
    fi_above_thr = cropped_image_1 > thr_fi_c
    si_above_thr = cropped_image_2 > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = cropped_image_1[combined_thresh_c]
    si_thresh_c = cropped_image_2[combined_thresh_c]

    tot_fi_thr_c = scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= thr_fi_c],
    )

    tot_si_thr_c = scipy.ndimage.sum(
        cropped_image_2[cropped_image_2 >= thr_si_c],
    )
    C1 = scipy.ndimage.sum(fi_thresh_c) / numpy.array(tot_fi_thr_c)
    C2 = scipy.ndimage.sum(si_thresh_c) / numpy.array(tot_si_thr_c)

    ################################################################################################
    # write the results to the output dictionary
    ################################################################################################

    results["MEAN.CORRELATION.COEFF"] = numpy.mean(corr)
    results["MEDIAN.CORRELATION.COEFF"] = numpy.median(corr)
    results["MIN.CORRELATION.COEFF"] = numpy.min(corr)
    results["MAX.CORRELATION.COEFF"] = numpy.max(corr)
    results["MEAN.MANDERS.COEFF.M1"] = numpy.mean(M1)
    results["MEDIAN.MANDERS.COEFF.M1"] = numpy.median(M1)
    results["MIN.MANDERS.COEFF.M1"] = numpy.min(M1)
    results["MAX.MANDERS.COEFF.M1"] = numpy.max(M1)
    results["MEAN.MANDERS.COEFF.M2"] = numpy.mean(M2)
    results["MEDIAN.MANDERS.COEFF.M2"] = numpy.median(M2)
    results["MIN.MANDERS.COEFF.M2"] = numpy.min(M2)
    results["MAX.MANDERS.COEFF.M2"] = numpy.max(M2)
    results["MEAN.OVERLAP.COEFF"] = numpy.mean(overlap)
    results["MEDIAN.OVERLAP.COEFF"] = numpy.median(overlap)
    results["MIN.OVERLAP.COEFF"] = numpy.min(overlap)
    results["MAX.OVERLAP.COEFF"] = numpy.max(overlap)
    results["MEAN.K1"] = numpy.mean(K1)
    results["MEDIAN.K1"] = numpy.median(K1)
    results["MIN.K1"] = numpy.min(K1)
    results["MAX.K1"] = numpy.max(K1)
    results["MEAN.K2"] = numpy.mean(K2)
    results["MEDIAN.K2"] = numpy.median(K2)
    results["MIN.K2"] = numpy.min(K2)
    results["MAX.K2"] = numpy.max(K2)
    results["MEAN.MANDERS.COEFF.COSTES.M1"] = numpy.mean(C1)
    results["MEDIAN.MANDERS.COEFF.COSTES.M1"] = numpy.median(C1)
    results["MIN.MANDERS.COEFF.COSTES.M1"] = numpy.min(C1)
    results["MAX.MANDERS.COEFF.COSTES.M1"] = numpy.max(C1)
    results["MEAN.MANDERS.COEFF.COSTES.M2"] = numpy.mean(C2)
    results["MEDIAN.MANDERS.COEFF.COSTES.M2"] = numpy.median(C2)
    results["MIN.MANDERS.COEFF.COSTES.M2"] = numpy.min(C2)
    results["MAX.MANDERS.COEFF.COSTES.M2"] = numpy.max(C2)
    return results
