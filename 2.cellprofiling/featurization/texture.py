import mahotas


def measure_3D_texture(
    image_object,
    distance=1,
):
    # Display results
    feature_names = [
        "Angular Second Moment",
        "Contrast",
        "Correlation",
        "Variance",
        "Inverse Difference Moment",
        "Sum Average",
        "Sum Variance",
        "Sum Entropy",
        "Entropy",
        "Difference Variance",
        "Difference Entropy",
        "Information Measure of Correlation 1",
        "Information Measure of Correlation 2",
        "Maximal Correlation Coefficient",
    ]
    haralick_features = mahotas.features.haralick(
        ignore_zeros=False,
        f=image_object,
        distance=distance,
        compute_14th_feature=False,
    )
    haralick_mean = haralick_features.mean(axis=0)
    output_dict = dict(zip(feature_names, haralick_mean))
    return output_dict
