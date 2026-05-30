# Contributing

Contributions are welcome through issues and pull requests.

## Scope

Please keep changes scoped to one component where possible:

- hardware design files in `Structure/`, `Electronics/`, or `Zemax/`;
- embedded firmware in `Firmware/`;
- acquisition software in `DAQ_software/`;
- calcium processing in `Software/Airscope_ca_processing/`;
- behavioural segmentation in `Software/SAM2Mice/`;
- neural decoding in `Software/Neuron_BERT/`;
- documentation and released-data descriptors in `README.md` and
  `Data_release/`.

## Pull Requests

Before opening a pull request:

1. Describe the scientific or engineering motivation for the change.
2. List the affected hardware, firmware, software, or data components.
3. Include the commands, notebooks, or manual checks used to validate the
   change.
4. Avoid committing generated files, local caches, model checkpoints,
   downloaded dependencies, or raw datasets unless they are intentionally part
   of the public release.
5. Preserve existing third-party license notices and add new notices when
   introducing third-party code, weights, hardware footprints, or datasets.

## Licensing

By contributing, you agree that your contribution is licensed under the license
that applies to the target directory. See the top-level `LICENSE` file for the
license map.

If your contribution cannot be distributed under the target directory license,
state this explicitly before submitting it.

## Data and Model Weights

Large files should normally be provided through a documented download location
instead of being committed to git. If a model checkpoint or dataset is required
for reproducibility, include:

- source URL or accession;
- version or date downloaded;
- checksum when practical;
- license or reuse terms;
- expected local path used by the code.
