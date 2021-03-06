/***********************************************************************
 *
 * Copyright (C) 2014 Carsten Urbach
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 *
 * invert wrapper for using tmLQCD as a library
 *
 * Author: Carsten Urbach
 *         curbach@gmx.de
 *
 *******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#ifdef TM_USE_MPI
#include <mpi.h>
#endif
#ifdef TM_USE_OMP
# include <omp.h>
#endif
#include "global.h"
#include "git_hash.h"
#include "getopt.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#ifdef TM_USE_MPI
#include "xchange/xchange.h"
#endif
#include <io/utils.h>
#include <io/gauge.h>
#include "read_input.h"
#include "mpi_init.h"
#include "init/init.h"
#include "sighandler.h"
#include "boundary.h"
#include "invert_eo.h"
#include "start.h"
#include "operator.h"
#include "operator/clovertm_operators.h"
#include "operator/clovertm_operators_32.h"
#include "operator/clover_leaf.h"
#include "solver/solver_types.h"
#include "measure_gauge_action.h"
#include "linalg/convert_eo_to_lexic.h"
#include "operator/Hopping_Matrix.h"
#include "include/tmLQCD.h"

#ifdef HAVE_GPU
extern void init_mixedsolve_eo(su3** gf);
extern void init_mixedsolve(su3** gf);
extern void finalize_mixedsolve();
extern void init_gpu_fields(int need_momenta);
extern void finalize_gpu_fields();
#include "GPU/cudadefs.h"
#  ifdef TEMPORALGAUGE
#  include "temporalgauge.h" 
#  endif
#endif


static int tmLQCD_invert_initialised = 0;

int tmLQCD_invert_init(int argc, char *argv[], const int _verbose) {

  DUM_DERI = 8;
  DUM_MATRIX = DUM_DERI + 5;
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
  //4 extra fields (corresponding to DUM_MATRIX+0..5) for deg. and ND matrix mult.  
  NO_OF_SPINORFIELDS_32 = 6;

  // in read_input.h
  verbose = _verbose;
  g_use_clover_flag = 0;

#ifdef TM_USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#else
  g_proc_id = 0;
#endif

  /* Read the input file */
  if( (read_input("invert.input")) != 0) {
    fprintf(stderr, "tmLQCD_init_invert: Could not find input file: invert.input\nAborting...");
  }

#ifdef TM_USE_OMP
  init_openmp();
#endif

  tmlqcd_mpi_init(argc, argv);
  g_dbw2rand = 0;
  for(int j = 0; j < no_operators; j++) if(!operator_list[j].even_odd_flag) even_odd_flag = 0;

#ifdef _GAUGE_COPY
  int j = init_gauge_field(VOLUMEPLUSRAND, 1);
  j += init_gauge_field_32(VOLUMEPLUSRAND, 1);
#else
  int j = init_gauge_field(VOLUMEPLUSRAND, 0);
  j += init_gauge_field_32(VOLUMEPLUSRAND, 0);  
#endif
  if (j != 0) {
    fprintf(stderr, "tmLQCD_init_invert: Not enough memory for gauge_fields! Aborting...\n");
    return(-1);
  }
  j = init_geometry_indices(VOLUMEPLUSRAND);
  if (j != 0) {
    fprintf(stderr, "tmLQCD_init_invert: Not enough memory for geometry indices! Aborting...\n");
    return(-1);
  }
  if (even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
    j += init_spinor_field_32(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS_32);   
  }
  else {
    j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
    j += init_spinor_field_32(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS_32);   
  }
  if (j != 0) {
    fprintf(stderr, "tmLQCD_init_invert: Not enough memory for spinor fields! Aborting...\n");
    return(-1);
  }
  // define the geometry
  geometry();

  // initialise the operators
  init_operators();
  
#ifdef HAVE_GPU
  if(usegpu_flag){
    if(even_odd_flag){
      init_mixedsolve_eo(g_gauge_field);
    }
    else{
      init_mixedsolve(g_gauge_field);
    }
#  ifdef GPU_DOUBLE
    /*init double fields w/o momenta*/
    init_gpu_fields(0);
#  endif
#  ifdef TEMPORALGAUGE
    int retval;
    if((retval=init_temporalgauge(VOLUME, g_gauge_field)) !=0){
      if(g_proc_id == 0) printf("tmLQCD_init_invert: Error while initializing temporal gauge. Aborting...\n");   
      exit(200);
    }
#  endif
  }//usegpu_flag
#endif  


#ifdef _USE_HALFSPINOR
  j = init_dirac_halfspinor();
  if (j != 0) {
    fprintf(stderr, "tmLQCD_init_invert: Not enough memory for halffield! Aborting...\n");
    return(-1);
  }
  /* for mixed precision solvers, single precisio halfspinor field must always be there! */
  j = init_dirac_halfspinor32();
  if (j != 0) {
    fprintf(stderr, "tmLQCD_init_invert: Not enough memory for 32-bit halffield! Aborting...\n");
    return(-1);
  }
#  if (defined _PERSISTENT)
  if (even_odd_flag)
    init_xchange_halffield();
#  endif
#endif
  tmLQCD_invert_initialised = 1;  
  return(0);
}  /* end of tmLQCD_invert_init */

/***************************************************************************************
 *
 ***************************************************************************************/
int tmLQCD_read_gauge(const int nconfig) {
  char conf_filename[500];
  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "[tmLQCD_read_gauge] tmLQCD_read_gauge: tmLQCD_inver_init must be called first. Aborting...\n");
    return(-1);
  }

  if ( strcmp( gauge_input_filename, "unit_gauge_field") == 0 ) {
    if (g_cart_id == 0) {
      printf("# [tmLQCD_read_gauge] \n# [tmLQCD_read_gauge] Initializing unit gauge field\n");
      fflush(stdout);
    }
    unit_g_gauge_field();
    if (g_cart_id == 0) {
      printf("# [tmLQCD_read_gauge] Finished reading gauge field.\n");
      fflush(stdout);
    }
  } else {
    sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nconfig);
    int j=0;
    if (g_cart_id == 0) {
      printf("# [tmLQCD_read_gauge]\n# [tmLQCD_read_gauge] Trying to read gauge field from file %s.\n", conf_filename);
      fflush(stdout);
    }
    if( (j = read_gauge_field(conf_filename,g_gauge_field)) !=0) {
      fprintf(stderr, "[tmLQCD_read_gauge] tmLQCD_read_gauge: Error %d while reading gauge field from %s\n ...\n", j, conf_filename);
      return(-1);
    }
    if (g_cart_id == 0) {
      printf("# [tmLQCD_read_gauge] Finished reading gauge field.\n");
      fflush(stdout);
    }
  }  /* end of if unit_gauge_field */
#ifdef TM_USE_MPI
  xchange_gauge(g_gauge_field);
#endif
  convert_32_gauge_field(g_gauge_field_32, g_gauge_field, VOLUMEPLUSRAND);

  double plaquette = measure_plaquette( (const su3** const) g_gauge_field)/(6.*VOLUME*g_nproc);
  if (g_cart_id == 0) {
    printf("# [tmLQCD_read_gauge] The computed plaquette value is %.16e.\n", plaquette);
  }
  return(0);
}  /* end of tmLQCD_read_gauge */


/***************************************************************************************
 *
 ***************************************************************************************/
int tmLQCD_invert(double * const propagator, double * const source, 
		  const int op_id, const int write_prop) {
  unsigned int index_start = 0;
  g_mu = 0.;

  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "tmLQCD_invert: tmLQCD_inver_init must be called first. Aborting...\n");
    return(-1);
  }

  if(op_id < 0 || op_id >= no_operators) {
    fprintf(stderr, "tmLQCD_invert: op_id=%d not in valid range. Aborting...\n", op_id);
    return(-1);
  }

  operator_list[op_id].sr0 = g_spinor_field[0];
  operator_list[op_id].sr1 = g_spinor_field[1];
  operator_list[op_id].prop0 = g_spinor_field[2];
  operator_list[op_id].prop1 = g_spinor_field[3];

  zero_spinor_field(operator_list[op_id].prop0, VOLUME / 2);
  zero_spinor_field(operator_list[op_id].prop1, VOLUME / 2);

  // convert to even/odd order
  convert_lexic_to_eo(operator_list[op_id].sr0, operator_list[op_id].sr1, (spinor*) source);
  
  // invert
  operator_list[op_id].inverter(op_id, index_start, write_prop);

  // convert back to lexicographic order
  convert_eo_to_lexic((spinor*) propagator, operator_list[op_id].prop0, operator_list[op_id].prop1);

  return(0);
}  /* end of tmLQCD_invert */

/***************************************************************************************
 * invert even-odd preconditioned operator directly
 *   propagator and source must be in even-odd ordering already
 ***************************************************************************************/
int tmLQCD_invert_eo(double * const propagator, double * const source, const int op_id) {
  int iter;
  spinor * const p_ptr = (spinor * const)propagator;
  spinor * const s_ptr = (spinor * const)source;


  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "[tmLQCD_invert_eo] tmLQCD_inver_init must be called first. Aborting...\n");
    fflush(stderr);
    return(-1);
  }

  if(op_id < 0 || op_id >= no_operators) {
    fprintf(stderr, "[tmLQCD_invert_eo] op_id=%d not in valid range. Aborting...\n", op_id);
    fflush(stderr);
    return(-1);
  }

  if (operator_list[op_id].solver != EXACTDEFLATEDCG ) {
    fprintf(stderr, "[tmLQCD_invert_eo] Error, wrong solver type in operator\n");
    fflush(stderr);
    return(-2);
  }

  if(g_proc_id == 0) {
    printf("# [tmLQCD_invert_eo] Using eo-prec., symmetrized, deflated cg!\n");
    fflush(stdout);
  }

  /****************************************
   * - initialize parameters; this is otherwise
   *   done in operator.c or invert.c
   * -
   ****************************************/
  boundary(operator_list[op_id].kappa);
  g_mu    = operator_list[op_id].mu;
  g_c_sw  = operator_list[op_id].c_sw;
  g_kappa = operator_list[op_id].kappa;

  if(operator_list[op_id].type == CLOVER) {
    if (g_cart_id == 0 && g_debug_level > 1) {
      printf("#\n# csw = %e, computing clover leafs\n", g_c_sw);
    }
    init_sw_fields(VOLUME);

    sw_term( (const su3**) g_gauge_field, operator_list[op_id].kappa, operator_list[op_id].c_sw);
    /* this must be EE here!   */
    /* to match clover_inv in Qsw_psi */
    sw_invert(EE, operator_list[op_id].mu);
    /* now copy double sw and sw_inv fields to 32bit versions */
    copy_32_sw_fields();
  }

  /* call invert */
  iter = exactdeflated_cg( operator_list[op_id].solver_params, operator_list[op_id].deflator_params, p_ptr, s_ptr);

  if(iter<0) {
    fprintf(stderr, "[tmLQCD_invert_eo] Error from deflated_cg, status was %d\n", iter);
    fflush(stderr);
    return(1);
  }

  return(0);
}  /* end of tmLQCD_invert_eo */

/***************************************************************************************
 *
 ***************************************************************************************/
int tmLQCD_finalise() {

#ifdef TM_USE_OMP
  free_omp_accumulators();
#endif

#ifdef HAVE_GPU
  if(usegpu_flag){
    finalize_mixedsolve();
#  ifdef GPU_DOUBLE
    finalize_gpu_fields();
#  endif
#  ifdef TEMPORALGAUGE
    finalize_temporalgauge();
#  endif
  }
#endif
  
  free_gauge_field();
  free_gauge_field_32();                                                                                                                                                                                                                     
  free_geometry_indices();
  free_spinor_field();
  free_spinor_field_32();
  free_moment_field();
  free_chi_spinor_field();
#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  return(0);
}  /* end of tmLQCD_finalise */


/***************************************************************************************
 *
 ***************************************************************************************/
int tmLQCD_get_lat_params(tmLQCD_lat_params * params) {
  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "tmLQCD_get_lat_params: tmLQCD_inver_init must be called first. Aborting...\n");
    return(-1);
  }

  params->LX = LX;
  params->LY = LY;
  params->LZ = LZ;
  params->T = T;
  params->nstore = nstore;
  params->nsave = Nsave;
  params->no_operators = no_operators;
  return(0);
}

int tmLQCD_get_mpi_params(tmLQCD_mpi_params * params) {
  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "tmLQCD_get_mpi_params: tmLQCD_inver_init must be called first. Aborting...\n");
    return(-1);
  }

  params->nproc = g_nproc;
  params->nproc_t = g_nproc_t;
  params->nproc_x = g_nproc_x;
  params->nproc_y = g_nproc_y;
  params->nproc_z = g_nproc_z;
  params->cart_id = g_cart_id;
  params->proc_id = g_proc_id;
  params->time_rank = g_mpi_time_rank;
  params->omp_num_threads = omp_num_threads;
  params->proc_coords[0] = g_proc_coords[0];
  params->proc_coords[1] = g_proc_coords[1];
  params->proc_coords[2] = g_proc_coords[2];
  params->proc_coords[3] = g_proc_coords[3];
#ifdef TM_USE_MPI
  MPI_Comm_dup(g_cart_grid, &(params->cart_grid) );
#endif

  return(0);
}

int tmLQCD_get_gauge_field_pointer(double ** gf) {
  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "tmLQCD_get_gauge_field_pointer: tmLQCD_invert_init must be called first. Aborting...\n");
    return(-1);
  }
#ifdef TM_USE_MPI
  xchange_gauge(g_gauge_field);
#endif

  *gf = (double*) g_gauge_field[0];

  return(0);
}

/***************************************************************************************
 * copy several deflator parameters
 ***************************************************************************************/
int tmLQCD_get_deflator_params(tmLQCD_deflator_params*params, const int op_id) {
  if(!tmLQCD_invert_initialised) {
    fprintf(stderr, "[tmLQCD_get_deflator_params] tmLQCD_invert_init must be called first. Aborting...\n");
    return(-1);
  }

  if(operator_list[op_id].deflator_params.nconv == -1) {
    fprintf(stderr, "[tmLQCD_get_deflator_params] deflator no initialized. Aborting...\n");
    return(-2);
  }

  strcpy(params->type_name,  operator_list[op_id].deflator_params.type_name);
  params->eoprec = operator_list[op_id].deflator_params.eoprec;
  params->evecs  = operator_list[op_id].deflator_params.evecs;
  params->evals  = operator_list[op_id].deflator_params.evals;
  params->prec   = operator_list[op_id].deflator_params.prec;
  params->nev    = operator_list[op_id].deflator_params.nconv;

  return(0);
}  /* end of tmLQCD_get_deflator_params */

/***************************************************************************************
 * call the deflator function
 ***************************************************************************************/
int tmLQCD_init_deflator(const int op_id) {
  int status;

  if(op_id < 0 || op_id >= no_operators) {
    fprintf(stderr, "[tmLQCD_init_deflator] op_id=%d not in valid range. Aborting...\n", op_id);
    return(1);
  }


  if(operator_list[op_id].deflator_params.nconv > -1) {
    fprintf(stderr, "[tmLQCD_init_deflator] deflator already initialized. Aborting...\n");
    return(2);
  }

  /****************************************
   * initialize parameters; this is otherwise
   * done in operator.c or invert.c
   ****************************************/
  boundary(operator_list[op_id].kappa);
  g_mu    = operator_list[op_id].mu;
  g_c_sw  = operator_list[op_id].c_sw;
  g_kappa = operator_list[op_id].kappa;
             
  if (g_cart_id == 0) {
    fprintf(stdout, "# [tmLQCD_init_deflator] 2 kappa mu = %e, kappa = %e, c_sw = %e\n", g_mu, g_kappa, g_c_sw);
  }

  if(operator_list[op_id].type == CLOVER) {
    if (g_cart_id == 0) {
      printf("# [tmLQCD_init_deflator]\n# [tmLQCD_init_deflator] csw = %e, computing clover leafs\n", g_c_sw);
    }
    init_sw_fields(VOLUME);

    sw_term( (const su3**) g_gauge_field, operator_list[op_id].kappa, operator_list[op_id].c_sw);
    /* this must be EE here!   */
    /* to match clover_inv in Qsw_psi */
    sw_invert(EE, operator_list[op_id].mu);
    /* now copy double sw and sw_inv fields to 32bit versions */
    copy_32_sw_fields();
  }


  status = operator_list[op_id].deflator_params.init( &(operator_list[op_id].deflator_params) );
  if(status != 0) {
    fprintf(stderr, "[tmLQCD_init_deflator] Error from deflator_init, status was %d\n", status);
    return(3);
  }

  return(0);
}  /* end of tmLQCD_init_deflator */

int tmLQCD_set_deflator_fields(const int op_id1, const int op_id2) {
  int status;

  if( op_id1 < 0 || op_id1 >= no_operators || op_id2 < 0 || op_id2 >= no_operators) {
    fprintf(stderr, "[tmLQCD_init_deflator] op_id out of range; aborting\n");
    return(1);
  }

  if( operator_list[op_id2].deflator_params.nconv <= 0 || operator_list[op_id2].deflator_params.evecs == NULL || operator_list[op_id2].deflator_params.evals == NULL ) {
    fprintf(stderr, "[tmLQCD_copy_deflator_fields] source deflator not initialized; aborting\n");
    return(2);
  }

  /****************************************
   * set deflator fields
   ****************************************/
  operator_list[op_id1].deflator_params.nconv = operator_list[op_id2].deflator_params.nconv;
  operator_list[op_id1].deflator_params.evecs = operator_list[op_id2].deflator_params.evecs;
  operator_list[op_id1].deflator_params.evals = operator_list[op_id2].deflator_params.evals;

  return(0);
}  /* end of tmLQCD_set_deflator_fields */

