// xzforce.cmd
// Commands for use with datafiles with lateral offset.

// command to smoothly change x offset.
// Use: set new_offset to desired value, then do do_offset.
new_offset := offset;
do_offset := { doff := new_offset - offset;
	       offset := new_offset;
	       set vertex x x + doff*z/height;
	       recalc;
             }

// Central difference, moving linearly
dx := 1e-5;
do_xforce := { 
             new_offset := offset + dx; // do this first so constraint in right place
	     do_offset;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             new_offset := offset - 2*dx;
	     do_offset;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             xforce := -(new_energy_1 - new_energy_2)/2/dx;
             printf "xforce: %18.15f  (central difference, linear move)\n", xforce;
             new_offset := offset + dx; // restore original
	     do_offset;
           }

// Central difference, moving linearly
dz := 1e-5;
do_zforce := { 
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

