// runforce.cmd

// Script to automatically run a two-parameter set of force
// calculations and print the results to a text file suitable
// for import into another program.

// Usage: Initialize the bga to the minimum height and volume desired.
// Refine and evolve to the desired fineness.  Set delta_h, delta_vol,
// max_h, max_vol, and outfile if the defaults below are not acceptable.
// Run "runforce".

outfile := "runforce.out";    // default output file
delta_h := .01;               // height increment
delta_vol := .0005;           // volume increment
max_h := .30;                 // maximum height
max_vol := 0.01;              // maximum volume
define vertex attribute old_coord real[3]  // for saving old coordinates.

runforce := {
   start_height := height;
   // print header info
   printf "volume                  height\n                      " >>> outfile; 
   ht := height;
   while ( ht < max_h ) do 
   { printf "%20.15f ",ht >> outfile; ht := ht + delta_h; };
   printf "\n" >> outfile;
   while ( body[1].target < max_vol ) do
   { printf "%20.15f  ", body[1].target >> outfile;
     set vertex old_coord[1] x;  // save original coordinates
     set vertex old_coord[2] y;
     set vertex old_coord[3] z;
     eigen_flag := 0;
     while ( height < max_h ) do
     { 
       // Use hessian to converge.  Test each time for onset of instability.
       { hessian; 
         if ( eigen_neg == 0 ) then eigen_flag := 1; // onset of stability
         if ( eigen_neg && (eigen_flag==1) ) then break;  // out of while
       } 3;
       if ( eigen_neg == 0 ) then  // skip unstable configuration
       { zforce4; printf "%20.15f ",zforce >> outfile; }
       else { printf "                     " >> outfile;};
       set vertex z z+delta_h*z/height;  // linear interpolation of stretch
       height := height + delta_h;
       recalc;
     };
     printf "\n" >> outfile;
     height := start_height;   // prepare for next go-around
     set vertex x old_coord[1];
     set vertex y old_coord[2];
     set vertex z old_coord[3];
     body[1].target := body[1].target + delta_vol;
     hessian; hessian;
   }
}
     
    
