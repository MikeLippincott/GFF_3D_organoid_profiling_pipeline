#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib

# In[ ]:


argparser = argparse.ArgumentParser(
    description="Generate a Python script to print the contents of a file."
)
argparser.add_argument(
    "--patient",
    type=pathlib.Path,
    required=True,
    help="Path to the patient file to be read.",
)

args = argparser.parse_args()
patient = args.patient


# In[ ]:


data_path = pathlib.Path(f"../../data/{patient}").resolve()
data_path = data_path.glob("*")
data_path = [x for x in data_path if x.is_dir()]
data_path = sorted(data_path)


patient_well_fov_list = []
for well_fov in pathlib.Path(patient / "zstack_images").glob("*"):
    patient_well_fov = f"{patient.parts[-1]}_{well_fov.parts[-1]}"
    patient_well_fov_list.append(patient_well_fov)
# write to file
# with two columns separated by a tab
# column one is the patient
# column two is the well_fov
save_path = pathlib.Path(f"../load_files/{patient.name}_well_fov.tsv").resolve()
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    f.write("patient\twell_fov\n")
    for patient_well_fov in patient_well_fov_list:  # wite the first two lines to test
        f.write(f"{patient_well_fov.split('_')[0]}\t{patient_well_fov.split('_')[1]}\n")
