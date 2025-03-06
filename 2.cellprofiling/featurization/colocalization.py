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


def calculate_3D_colocalization(
    croppped_image_1, croppped_image_2, thr=15, fast_costes="Accurate"
):
    results = {}
    thr = 15
    ################################################################################################
    # Calculate the correlation coefficient between the two images
    ################################################################################################
    mean1 = scipy.ndimage.mean(croppped_image_1, 1)
    mean2 = scipy.ndimage.mean(croppped_image_2, 1)
    std1 = numpy.sqrt(scipy.ndimage.sum((croppped_image_1 - mean1) ** 2))
    std2 = numpy.sqrt(scipy.ndimage.sum((croppped_image_2 - mean2) ** 2))
    x = croppped_image_1 - mean1
    y = croppped_image_2 - mean2
    corr = scipy.ndimage.sum(x * y / (std1 * std2))

    ################################################################################################
    # Calculate the Manders' coefficients
    ################################################################################################

    # Threshold as percentage of maximum intensity of objects in each channel
    tff = (thr / 100) * scipy.ndimage.maximum(croppped_image_1)
    tss = (thr / 100) * scipy.ndimage.maximum(croppped_image_2)

    combined_thresh = (croppped_image_1 >= tff) & (croppped_image_2 >= tss)

    fi_thresh = croppped_image_1[combined_thresh]
    si_thresh = croppped_image_2[combined_thresh]
    tot_fi_thr = scipy.ndimage.sum(
        croppped_image_1[croppped_image_1 >= tff],
    )
    tot_si_thr = scipy.ndimage.sum(croppped_image_2[croppped_image_2 >= tss])

    M1 = scipy.ndimage.sum(fi_thresh) / numpy.array(tot_fi_thr)
    M2 = scipy.ndimage.sum(si_thresh) / numpy.array(tot_si_thr)

    ################################################################################################
    # Calculate the overlap coefficient
    ################################################################################################

    fpsq = scipy.ndimage.sum(
        croppped_image_1[combined_thresh] ** 2,
    )
    spsq = scipy.ndimage.sum(
        croppped_image_2[combined_thresh] ** 2,
    )
    pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))
    overlap = (
        scipy.ndimage.sum(
            croppped_image_1[combined_thresh] * croppped_image_2[combined_thresh],
        )
        / pdt
    )
    K1 = scipy.ndimage.sum(
        croppped_image_1[combined_thresh] * croppped_image_2[combined_thresh],
    ) / (numpy.array(fpsq))
    K2 = scipy.ndimage.sum(
        croppped_image_1[combined_thresh] * croppped_image_2[combined_thresh],
    ) / (numpy.array(spsq))

    ################################################################################################
    # Calculate the Costes' coefficient
    ################################################################################################

    # Orthogonal Regression for Costes' automated threshold
    if numpy.max(croppped_image_1) > 255 or numpy.max(croppped_image_2) > 255:
        scale = 65535
    else:
        scale = 255

    if fast_costes == "Accurate":
        thr_fi_c, thr_si_c = bisection_costes(croppped_image_1, croppped_image_2, scale)
    else:
        thr_fi_c, thr_si_c = linear_costes(
            croppped_image_1, croppped_image_2, scale, fast_costes
        )

    # Costes' thershold for entire image is applied to each object
    fi_above_thr = croppped_image_1 > thr_fi_c
    si_above_thr = croppped_image_2 > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = croppped_image_1[combined_thresh_c]
    si_thresh_c = croppped_image_2[combined_thresh_c]

    tot_fi_thr_c = scipy.ndimage.sum(
        croppped_image_1[croppped_image_1 >= thr_fi_c],
    )

    tot_si_thr_c = scipy.ndimage.sum(
        croppped_image_2[croppped_image_2 >= thr_si_c],
    )
    C1 = scipy.ndimage.sum(fi_thresh_c) / numpy.array(tot_fi_thr_c)
    C2 = scipy.ndimage.sum(si_thresh_c) / numpy.array(tot_si_thr_c)

    ################################################################################################
    # write the results to the output dictionary
    ################################################################################################

    results["Mean Correlation coeff"] = numpy.mean(corr)
    results["Median Correlation coeff"] = numpy.median(corr)
    results["Min Correlation coeff"] = numpy.min(corr)
    results["Max Correlation coeff"] = numpy.max(corr)
    results["Mean Manders coeff_M1"] = numpy.mean(M1)
    results["Median Manders coeff_M1"] = numpy.median(M1)
    results["Min Manders coeff_M1"] = numpy.min(M1)
    results["Max Manders coeff_M1"] = numpy.max(M1)
    results["Mean Manders coeff_M2"] = numpy.mean(M2)
    results["Median Manders coeff_M2"] = numpy.median(M2)
    results["Min Manders coeff_M2"] = numpy.min(M2)
    results["Max Manders coeff_M2"] = numpy.max(M2)
    results["Mean_overlap_coeff"] = numpy.mean(overlap)
    results["Median_overlap_coeff"] = numpy.median(overlap)
    results["Min_overlap_coeff"] = numpy.min(overlap)
    results["Max_overlap_coeff"] = numpy.max(overlap)
    results["Mean_K1"] = numpy.mean(K1)
    results["Median_K1"] = numpy.median(K1)
    results["Min_K1"] = numpy.min(K1)
    results["Max_K1"] = numpy.max(K1)
    results["Mean_K2"] = numpy.mean(K2)
    results["Median_K2"] = numpy.median(K2)
    results["Min_K2"] = numpy.min(K2)
    results["Max_K2"] = numpy.max(K2)
    results["Mean_Manders_Coeff(costes)_M1"] = numpy.mean(C1)
    results["Median_Manders_Coeff(costes)_M1"] = numpy.median(C1)
    results["Min_Manders_Coeff(costes)_M1"] = numpy.min(C1)
    results["Max_Manders_Coeff(costes)_M1"] = numpy.max(C1)
    results["Mean_Manders_Coeff(costes)_M2"] = numpy.mean(C2)
    results["Median_Manders_Coeff(costes)_M2"] = numpy.median(C2)
    results["Min_Manders_Coeff(costes)_M2"] = numpy.min(C2)
    results["Max_Manders_Coeff(costes)_M2"] = numpy.max(C2)
    return results
