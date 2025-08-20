import argparse


def check_for_missing_args(**kwargs):
    missing_args = []
    for arg, value in kwargs.items():
        if value is None:
            missing_args.append(arg)
    if missing_args:
        raise ValueError(
            f"Missing required arguments: {', '.join(missing_args)}. "
            "Please provide all required arguments."
        )


def parse_segmentation_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default=None,
        help="Well and field of view to process, e.g. 'A01-1'",
    )
    argparser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Patient ID, e.g. 'NF0014'",
    )
    argparser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for image processing, e.g. 3",
    )
    argparser.add_argument(
        "--clip_limit",
        type=float,
        default=None,
        help="Clip limit for contrast enhancement, e.g. 0.05",
    )
    argparser.add_argument(
        "--compartment",
        type=str,
        default=None,
        help="Compartment to process, e.g. 'Nuclei'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
    window_size = args.window_size
    clip_limit = args.clip_limit
    compartment = args.compartment
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        window_size=window_size,
        clip_limit=clip_limit,
        compartment=compartment,
    )
    return {
        "well_fov": well_fov,
        "patient": patient,
        "window_size": window_size,
        "clip_limit": clip_limit,
        "compartment": compartment,
    }
