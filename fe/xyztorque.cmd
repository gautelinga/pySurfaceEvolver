// Commands to calculate x, y, and z forces, and tilt torque.

// command to smoothly change x offset.
// Use: set new_x_offset to desired value, then do do_x_offset.
new_x_offset := x_offset;
do_x_offset := { doff := new_x_offset - x_offset;
	       x_offset := new_x_offset;
               slope := tan(tilt*pi/180);
	       set vertex x x + doff*z/(height + slope*(y - y_offset));
	       recalc;
             }
// Central difference, moving linearly
dx := 1e-5;
do_xforce := { 
             new_x_offset := x_offset + dx; // do this first so constraint in right place
	     do_x_offset;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             new_x_offset := x_offset - 2*dx;
	     do_x_offset;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             xforce := -(new_energy_1 - new_energy_2)/2/dx;
             printf "xforce: %18.15f  (central difference, linear move)\n", xforce;
             new_x_offset := x_offset + dx; // restore original
	     do_x_offset;
           }

// command to smoothly change y offset.
// Use: set new_y_offset to desired value, then do do_y_offset.
new_y_offset := y_offset;
do_y_offset := { doff := new_y_offset - y_offset;
	       y_offset := new_y_offset;
	       set vertex y y + doff*z/(height + slope*(y - y_offset));
	       recalc;
             }
// Central difference, moving linearly
dy := 1e-5;
do_yforce := { 
             new_y_offset := y_offset + dy; // do this first so constraint in right place
	     do_y_offset;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             new_y_offset := y_offset - 2*dy;
	     do_y_offset;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             yforce := -(new_energy_1 - new_energy_2)/2/dy;
             printf "yforce: %18.15f  (central difference, linear move)\n", yforce;
             new_y_offset := y_offset + dy; // restore original
	     do_y_offset;
           }

// Central difference, moving linearly
dz := 1e-5;
do_zforce := { 
	     set vertex z z + dz*z/height; // do this before changing height
             height := height + dz;
             recalc;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
	     set vertex z z - 2*dz*z/(height + slope*(y - y_offset)); 
             height := height - 2*dz;
             recalc;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             zforce := -(new_energy_1 - new_energy_2)/2/dz;
             printf "zforce: %18.15f  (central difference, linear move)\n",
	       zforce;
	     set vertex z z + dz*z/(height + slope*(y - y_offset)); 
             height := height + dz; // restore original
             recalc;
           }

// Torque
// First, a smooth tilting
new_tilt := tilt;
do_tilt := { dtilt := (new_tilt - tilt)*pi/180;
             slope := tan(tilt);
	     tilt := new_tilt;
             foreach vertex vv do {
	       beta := z/(height + slope*(y - y_offset));
	       vv.y := y_offset + cos(beta*dtilt)*(y - y_offset) 
		   - sin(beta*dtilt)*(z - beta*height);
	       vv.z := beta*height + sin(beta*dtilt)*(y - y_offset) 
		   + cos(beta*dtilt)*(z - beta*height);
	       }
           }

// Torque by central difference, moving linearly
dangle := 1e-4; // degrees
do_torque := { 
             new_tilt := tilt + dangle; // do this first so constraint in right place
	     do_tilt;
             new_energy_1 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             new_tilt := tilt - 2*dangle;
	     do_tilt;
             new_energy_2 := total_energy - body[1].pressure *
                      (body[1].volume - body[1].target);
             torque := -(new_energy_1 - new_energy_2)/2/(dangle*pi/180);
             printf "torque: %18.15f  (central difference, linear move)\n", torque;
             new_tilt := tilt + dangle; // restore original
	     do_tilt;
           }

forces := { do_xforce; do_yforce; do_zforce; do_torque; }	     

