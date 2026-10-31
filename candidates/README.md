# Candidate Catalog

This directory contains the RSI-selected archival PSB/E+A candidate catalog.

## Files

- `psb_candidates.csv`: object-level catalog after 3 arcsec consolidation.
- `psb_candidate_spectra.csv`: spectrum-level companion table for the retained
  LAMOST spectra associated with the candidate objects.

The object-level table uses one representative spectrum per candidate object.
The spectrum-level table preserves all retained spectra, so multiple rows may
share the same `candidate_id`.

The broad catalog contains 133 objects and 144 retained spectra. Its nested
conservative core contains 73 objects and 80 spectra. Previously reported
PSB/E+A sources are excluded through 3 arcsec matching to the literature
compilation and known-positive benchmark, supplemented by catalog queries.
Candidate objects are numbered consecutively from RSI-PSB-0001 to RSI-PSB-0133.

## Columns

- `candidate_id`: RSI candidate identifier.
- `ra_deg`, `dec_deg`, `redshift`: sky position and redshift.
- `representative_spectrum_id`, `representative_obsid`: representative LAMOST
  spectrum and observation identifiers in `psb_candidates.csv`.
- `spectrum_id`, `obsid`: LAMOST spectrum and observation identifiers in
  `psb_candidate_spectra.csv`.
- `conservative_core`: marks membership in the nested conservative subset
  defined by the combined criteria described in the paper.
- `visual_assessment_grade`: auxiliary spectrum-level grade from full-spectrum
  visual inspection (`A`: clean, `B`: plausible, `C`: borderline/contaminated,
  `D`: weak visual support). It is not a confirmation label.
- `n_retained_spectra`: number of retained spectra associated with the
  candidate object.
- `snr_4000_5000`: median-flux S/N proxy over 4000-5000 Angstrom in the rest
  frame.
- `ew_oii_3727`, `ew_hdelta_4101`, `ew_hgamma_4340`,
  `ew_hbeta_4861`, `ew_halpha_6563`: Goto-style equivalent-width
  measurements in Angstrom. Absorption is positive and emission is negative.

These tables list candidate objects.

## Measurements

EWs are measured on the original spectra over the full adopted line and
continuum windows. The [O II] blue continuum window spans 3653-3713 Angstrom.
The measurements follow the same prescription as the companion RSI-toolkit
analysis release. Hbeta is the measured line EW without an emission-infill
correction.

## License

These candidate tables and their data documentation use CC BY 4.0.
See [LICENSE.txt](LICENSE.txt). Original survey spectra are not included
or relicensed.
