# Candidate Catalog

This directory contains the RSI-selected archival PSB/E+A candidate catalog.

## Files

- `psb_candidates.csv`: object-level catalog after 3 arcsec consolidation.
- `psb_candidate_spectra.csv`: spectrum-level companion table for the retained
  LAMOST spectra associated with the candidate objects.

The object-level table uses one representative spectrum per candidate object.
The spectrum-level table preserves all retained spectra, so multiple rows may
share the same `candidate_id`.

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

These tables list candidate objects; confirmation requires follow-up observations.

## Measurement update (2026-10-03)

[O II] EWs were corrected using the full adopted blue continuum window,
3653-3713 Angstrom, rather than truncating it at 3700 Angstrom before
measurement. Other line EWs, SNR, candidate IDs, representative spectra,
and memberships are unchanged: 145 objects, 160 retained spectra and the
nested 90-spectrum core. Both tables correspond to the updated full-window
measurements in the companion RSI-toolkit analysis release.
