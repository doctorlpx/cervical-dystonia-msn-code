# Prepare local inputs

Use Python 3.13 and the packages in `requirements.txt` for analysis. Run the commands from the repository root, replacing the uppercase path names with local directories.

1. Generate the parcel labels and spatial rotations using the CAT12 12.8.1 (r2043) templates and the official [Schaefer fsaverage 7-network annotations](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label).
   ```sh
   python -m pip install neuromaps==0.0.7
   python preparation/atlas.py CAT12 ANNOTATIONS prepared/reference
   ```
2. Process the OpenNeuro T1 scans in SPM12/CAT12. Place each participant’s CAT12 XML report, log, central surfaces, registered spheres and thickness files under `CAT_ROOT/<participant_id>/anat`. Extract the cohort and covariates, then build the networks in MATLAB R2024b with the Statistics and Machine Learning Toolbox.
   ```sh
   python preparation/prepare_cohort.py participants.tsv CAT_ROOT prepared
   ```
   ```matlab
   addpath('preparation')
   build_networks('CAT_ROOT','SPM12','prepared')
   ```
3. In a separate environment, install `requirements-abagen.txt`. Download the six normalized Allen donor archives into `DONOR_ROOT/normalized_microarray_donor<ID>`.
   ```sh
   python preparation/expression.py DONOR_ROOT CAT12/templates_MNI152NLin2009cAsym prepared
   ```
4. In the analysis environment, project the PET source images listed in `study_settings.py` using the neuromaps 164k registration-fusion coordinates. Build gene definitions from the [HGNC complete set](https://www.genenames.org/download/archive/), the five [Enrichr GMT libraries](https://maayanlab.cloud/Enrichr/#libraries) named in the same file and the [cell-type marker list](https://doi.org/10.1038/s41467-020-17051-5).
   ```sh
   python preparation/receptor_maps.py PET_IMAGES REGISTRATION_FUSION prepared/reference
   python preparation/rebuild_gene_definitions.py GENE_RESOURCES prepared/expression/schaefer400/expression.csv prepared/reference
   ```

For optional map-based runs, put local group-contrast CSVs in `prepared/group_maps/<branch>.csv` using the `REGIONAL_GLM.csv` format from a full run. To summarize separate positive and negative Metascape exports, run `python preparation/read_metascape.py POSITIVE.zip NEGATIVE.zip summary`.

