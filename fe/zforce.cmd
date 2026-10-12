// zforce.cmd

// Some ways to calculate vertical force.
// zforce4 is the preferred way.

dz := 1e-5 * height // perturbation size

// Single difference, moving just top pad
zforce1 := { old_energy := total_energy;
             height := height + dz;
             recalc;
             new_energy := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             zforce := -(new_energy - old_energy)/dz;
             printf "zforce: %18.15f  (single difference, top pad only)\n",
		zforce;
             height := height - dz; // restore original
             recalc;
           }

// Central difference, moving just top pad
zforce2 := { 
             height := height + dz;
             recalc;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             height := height - 2*dz;
             recalc;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             zforce := -(new_energy_1 - new_energy_2)/2/dz;
             printf "zforce: %18.15f  (central difference, top pad only)\n",
	       zforce;
             height := height + dz; // restore original
             recalc;
           }

// Single difference, moving linearly
zforce3 := { old_energy := total_energy;
	     set vertex z z + dz*z/height; // do this before changing height
             height := height + dz;
             recalc;
             new_energy := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             zforce := -(new_energy - old_energy)/dz;
             printf "zforce: %18.15f  (single difference, linear move)\n",
		zforce;
	     set vertex z z - dz*z/height; 
             height := height - dz; // restore original
             recalc;
           }

// Central difference, moving linearly
zforce4 := { 
	     set vertex z z + dz*z/height; // do this before changing height
             height := height + dz;
             recalc;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
	     set vertex z z - 2*dz*z/height; 
             height := height - 2*dz;
             recalc;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             zforce := -(new_energy_1 - new_energy_2)/2/dz;
             printf "zforce: %18.15f  (central difference, linear move)\n",
	       zforce;
	     set vertex z z + dz*z/height; 
             height := height + dz; // restore original
             recalc;
           }


// Central difference, moving linearly, with re-convergence,
// so don't need volume corrections.
zforce5 := { 
	     set vertex z z + dz*z/height; // do this before changing height
             height := height + dz;
             recalc; hessian; hessian;
             new_energy_1 := total_energy; 
	     set vertex z z - 2*dz*z/height; 
             height := height - 2*dz;
             recalc; hessian; hessian;
             new_energy_2 := total_energy;
             zforce := -(new_energy_1 - new_energy_2)/2/dz;
             printf "zforce: %18.15f  (central difference, linear move, reconverge)\n",
	       zforce;
	     set vertex z z + dz*z/height; 
             height := height + dz; // restore original
             recalc; hessian; hessian;  // reconverge
           }

// Geometrical calculation of forces
// (not accurate since it leaves out effect of facets touching
// top pad at only one vertex)

zforce6 := { // first, pressure on top, accounting for gravity 
	     zforce := sum(facet where on_constraint 2,area) *
               (body[1].pressure - gravity_constant*height*SOLDER_DENSITY);
	     // add tension due to facets pulling on top
	     foreach edge ee where on_constraint 2 do
	     { zforce := zforce - sum(ee.facet ff where not fixed,
		  ff.tension*ee.length*sqrt(ff.x^2+ff.y^2)/area);
             };
             printf "zforce: %18.15f  (geometric)\n", zforce;
           }

