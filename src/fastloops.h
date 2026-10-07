/*
 * fastloops.h -- cached facet topology and parallel facet loops.
 *
 * Not part of the original Surface Evolver; see fastloops.c.
 */
#ifndef FASTLOOPS_H
#define FASTLOOPS_H

/* Three corner vertices per facet ordinal (tail order), or NULL when the
   cache doesn't apply (it is only built for soapfilm surfaces). */
vertex_id *fl_facet_corners(void);
/* Facets in FOR_ALL_FACETS order; *n gets their number. */
facet_id *fl_facet_list(long *n);
/* Nonzero when PYSE_CHECK_FACET_CACHE is set: verify every cached answer. */
int fl_check(void);

/* Classic facet volumes for linear soapfilm surfaces, computed in parallel
   with per-thread sums merged in thread order (agrees with the serial loop
   to round-off).
   Returns 0, doing nothing, when the case isn't covered. */
int fl_facet_volumes(void);

/* Facet surface energies (area, density, gravity) for linear soapfilm
   surfaces, as facet_energy_l(f_id,ALL_ENERGIES) for every facet: areas
   are stored, and the area and energy sums accumulated per thread.
   Returns 0, doing nothing, when the case isn't covered. */
int fl_facet_energies(void);

/* Facet tension and gravity forces for linear soapfilm surfaces, as
   facet_force_l() for every facet: areas stored, forces added to the
   vertices via per-thread arrays. Returns 0, doing nothing, when not covered. */
int fl_facet_forces(void);

/* Number of threads for the parallel loops (1 without OpenMP). */
int fl_threads(void);
void fl_set_threads(int n);

#endif
