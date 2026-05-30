# Notices

Airscope combines original hardware designs, firmware, host acquisition
software, analysis code, documentation, and released data. The repository uses
multiple licenses; see `LICENSE` for the directory-level license map.

Airscope software in `DAQ_software/`, `Firmware/`, and `Software/` is
distributed under the GNU General Public License v3.0 only unless a more
specific third-party file-level notice applies.

## Third-Party Components

- `Software/Airscope_ca_processing/` includes or adapts code from CaImAn,
  Suite2p, SIMA, scikit-image, and related scientific-imaging projects. See
  `Software/Airscope_ca_processing/THIRD_PARTY_NOTICES.md` and file-level
  notices in that directory.
- `Software/SAM2Mice/` is based on the SAM 2 ecosystem and includes additional
  third-party notices and license files in that directory, including
  `Software/SAM2Mice/LICENSE_cctorch`. File-level Apache-2.0 notices from the
  upstream SAM 2 demo code are retained for provenance.
- External model checkpoints and datasets may be subject to their own license
  terms and should be downloaded from the sources documented in the relevant
  module README files.

## Large Artifacts

Large generated artifacts, model weights, logs, caches, and downloaded
dependencies are not required for source-code review and should generally be
distributed through release assets, documented download links, or external data
repositories rather than committed directly to git.
