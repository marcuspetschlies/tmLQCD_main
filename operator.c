/***********************************************************************
 *
 * Copyright (C) 2009 Carsten Urbach
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
 ***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef TM_USE_MPI
# include <mpi.h>
#endif
#include "global.h"
#include "default_input_values.h"
#include "read_input.h"
#include "su3.h"
#include "operator/tm_operators.h"
#include "linalg_eo.h"
#include "operator/D_psi.h"
#include "operator/Dov_psi.h"
#include "operator/tm_operators_nd.h"
#include "operator/Hopping_Matrix.h"
#include "invert_eo.h"
#include "invert_doublet_eo.h"
#include "invert_overlap.h"
#include "invert_clover_eo.h"
#include "boundary.h"
#include "start.h"
#include "solver/eigenvalues.h"
#include "solver/solver.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "test/overlaptests.h"
#include "solver/index_jd.h"
#include "solver/deflator.h"
#include "little_D.h"
#include "operator/clovertm_operators.h"
#include "operator/clovertm_operators_32.h"
#include "operator/clover_leaf.h"
#include "operator.h"
#include "gettime.h"
#ifdef QUDA
#  include "quda_interface.h"
#endif

void dummy_D(spinor * const, spinor * const);
void dummy_Mee(spinor * const, spinor * const, double const);
void dummy_M(spinor * const, spinor * const, spinor * const, spinor * const);
void dummy_DbD(spinor * const s, spinor * const r, spinor * const p, spinor * const q);
void op_invert(const int op_id, const int index_start, const int write_prop);
void op_write_prop(const int op_id, const int index_start, const int append_);

operator operator_list[max_no_operators];

int no_operators = 0;

int add_operator(const int type) {

  operator * optr = &operator_list[no_operators];
  if(no_operators == max_no_operators) {
    fprintf(stderr, "maximal number of operators %d exceeded!\n", max_no_operators);
    exit(-1);
  }
  optr->type = type;
  optr->kappa = _default_g_kappa;
  optr->mu = _default_g_mu;
  optr->c_sw = _default_c_sw;
  optr->sloppy_precision = _default_operator_sloppy_precision_flag;
  optr->compression_type = _default_compression_type;
  optr->coefs = NULL;
  optr->rel_prec = _default_g_relative_precision_flag;
  optr->eps_sq = _default_solver_precision;
  optr->maxiter = _default_max_solver_iterations;
  optr->even_odd_flag = _default_even_odd_flag;
  optr->solver = _default_solver_flag;
  optr->mubar = _default_g_mubar;
  optr->epsbar = _default_g_epsbar;
  optr->sr0 = NULL;
  optr->sr1 = NULL;
  optr->sr2 = NULL;
  optr->sr3 = NULL;
  optr->prop0 = NULL;
  optr->prop1 = NULL;
  optr->prop2 = NULL;
  optr->prop3 = NULL;
  optr->error_code = 0;
  optr->prop_precision = _default_prop_precision_flag;
  optr->no_flavours = 1;
  optr->DownProp = 0;
  optr->conf_input = _default_gauge_input_filename;
  optr->no_extra_masses = 0;

  optr->applyM = &dummy_M;
  optr->applyMee = &dummy_Mee;
  optr->applyMeeInv = &dummy_Mee;
  optr->applyQ = &dummy_M;
  (optr->solver_params).mcg_delta = _default_mixcg_innereps;
  optr->applyQp = &dummy_D;
  optr->applyQm = &dummy_D;
  optr->applyMp = &dummy_D;
  optr->applyMm = &dummy_D;
  optr->applyQsq = &dummy_D;
  optr->applyDbQsq = &dummy_DbD;

  optr->inverter = &op_invert;
  optr->write_prop = &op_write_prop;

  /* ARPACK based solvers */
  (optr->solver_params).arpackcg_nrhs=0;
  (optr->solver_params).arpackcg_nrhs1=0;
  (optr->solver_params).arpackcg_eps_sq1=0.0;
  (optr->solver_params).arpackcg_eps_sq=0.0;
  (optr->solver_params).arpackcg_res_eps_sq=0.0;
  (optr->solver_params).arpackcg_nev=0;
  (optr->solver_params).arpackcg_ncv=0;
  (optr->solver_params).arpackcg_evals_kind=0;
  (optr->solver_params).arpackcg_comp_evecs=0;
  strncpy((optr->solver_params).arpack_logfile,"default_arpack_log.dat",300);
  (optr->solver_params).arpackcg_eig_tol=0.0;
  (optr->solver_params).arpackcg_eig_maxiter=0;
  (optr->solver_params).arpackcg_read_ev=0;
  (optr->solver_params).arpackcg_write_ev=0;
  (optr->solver_params).projection_type=0;
  (optr->solver_params).arpack_evecs_writeprec=32;
  (optr->solver_params).rel_prec = _default_g_relative_precision_flag;
  (optr->solver_params).eps_sq   = _default_solver_precision;
  (optr->solver_params).maxiter  = _default_max_solver_iterations;

  strncpy((optr->solver_params).arpack_evecs_filename,"eigenvectors",300);
  strncpy((optr->solver_params).arpack_evecs_fileformat,"NA",4);

  /* chebyshev polynomial related parameters */
  (optr->solver_params).cheb_k=0;
  (optr->solver_params).op_evmin=0.0;
  (optr->solver_params).op_evmax=0.0;


  /* init deflator parameters */
  (optr->deflator_params).type            = -1;
  (optr->deflator_params).eoprec          = _default_even_odd_flag;
  (optr->deflator_params).f               = NULL;
  (optr->deflator_params).f32             = NULL;
  (optr->deflator_params).f_final         = NULL;
  (optr->deflator_params).f_initial       = NULL;
  (optr->deflator_params).evecs           = NULL;
  (optr->deflator_params).evals           = NULL;
  (optr->deflator_params).prec            = 64;
  (optr->deflator_params).nconv           = -1;
  (optr->deflator_params).nev             = 0;
  (optr->deflator_params).ncv             = 0;
  (optr->deflator_params).evals_kind      = 0;
  (optr->deflator_params).evecs_howmny    = 'P';
  (optr->deflator_params).comp_evecs      = 0;
  (optr->deflator_params).eig_tol         = 0.;
  (optr->deflator_params).eig_maxiter     = 0;
  (optr->deflator_params).use_acc         = 0;
  (optr->deflator_params).cheb_k          = 0;
  (optr->deflator_params).op_evmin        = 0.0;
  (optr->deflator_params).op_evmax        = 0.0;
  (optr->deflator_params).write_ev        = 0;
  (optr->deflator_params).read_ev         = 0;
  (optr->deflator_params).evecs_writeprec = 64;
  (optr->deflator_params).init            = NULL;
  (optr->deflator_params).fini            = NULL;
  (optr->deflator_params).symmetric_operator = 1;
  strcpy( (optr->deflator_params).type_name,       "NA");
  strcpy((optr->deflator_params).logfile,          "arpack_log" ); 
  strcpy((optr->deflator_params).evecs_filename,   "eigenvectors");
  strcpy((optr->deflator_params).evecs_fileformat, "single");


  /* Overlap needs special treatment */
  if(optr->type == OVERLAP) {
    optr->even_odd_flag = 0;
    optr->solver = 13;
    optr->no_ev = 10;
    optr->no_ev_index = 8;
    optr->ev_prec = 1.e-15;
    optr->ev_readwrite = 0;
    optr->deg_poly = 50;
    optr->s = 0.6;
    optr->m = 0.;
    optr->inverter = &op_invert;
  }
  if(optr->type == DBTMWILSON || optr->type == DBCLOVER) {
    optr->no_flavours = 2;
    g_running_phmc = 1;
  }
  
  optr->precWS=NULL;

  optr->initialised = 1;

  no_operators++;
  return(no_operators);
}  /* end of add_operator */

int init_operators() {
  static int oinit = 0;
  operator * optr;
  if(!oinit) {
    oinit = 1;
    for(int i = 0; i < no_operators; i++) {
      optr = operator_list + i;
      /* This is a hack, it should be set on an operator basis. */
      optr->rel_prec = g_relative_precision_flag;

      (optr->solver_params).rel_prec = optr->rel_prec;
      (optr->solver_params).eps_sq   = optr->eps_sq;
      (optr->solver_params).maxiter  = optr->maxiter;

      if(optr->solver == EXACTDEFLATEDCG) {
        (optr->deflator_params).eoprec = optr->even_odd_flag;
        (optr->deflator_params).type   = optr->type;
        (optr->deflator_params).init   = make_exactdeflator;
        (optr->deflator_params).fini   = fini_exactdeflator;
      }

      if(optr->type == TMWILSON || optr->type == WILSON) {
        if(optr->c_sw > 0) {
          init_sw_fields();
        }
        optr->applyM = &M_full;
        optr->applyQ = &Q_full;

        if(optr->solver == EXACTDEFLATEDCG) {
          if(optr->type == TMWILSON ) {
            strcpy( (optr->deflator_params).type_name, "TMWILSON" );
          } else if( optr->type == WILSON) {
            strcpy( (optr->deflator_params).type_name, "WILSON" );
          }
        }

        if(optr->even_odd_flag) {
          /* TMWILSON or WILSON with even-odd preconditioning */
          optr->applyMee    = &Mee_psi;
          optr->applyMeeInv = &Mee_inv_psi;
          optr->applyQp = &Qtm_plus_psi;
          optr->applyQm = &Qtm_minus_psi;
          optr->applyQsq = &Qtm_pm_psi;
          optr->applyMp = &Mtm_plus_psi;
          optr->applyMm = &Mtm_minus_psi;

          if(optr->solver == EXACTDEFLATEDCG) {
            /* the operators in the deflator */
            if( (optr->deflator_params).symmetric_operator ) {
              (optr->deflator_params).f         = &Qtm_pm_psi;
              (optr->deflator_params).f32       = &Qtm_pm_psi_32;
              (optr->deflator_params).f_final   = &Qtm_plus_psi;
              (optr->deflator_params).f_initial = &Qtm_minus_psi;
            } else {
              (optr->deflator_params).f         = &Qtm_plus_psi;
              (optr->deflator_params).f_final   = NULL;
              (optr->deflator_params).f_initial = NULL;
            }
          }

        }
        else {
          optr->applyQp = &Q_plus_psi;
          optr->applyQm = &Q_minus_psi;
          optr->applyQsq = &Q_pm_psi;
          optr->applyMp = &D_psi;
          optr->applyMm = &M_minus_psi;
 
          if(optr->solver == EXACTDEFLATEDCG) {
            /* the operators in the deflator */
            if( (optr->deflator_params).symmetric_operator ) {
              (optr->deflator_params).f         = &Q_pm_psi;
              (optr->deflator_params).f32       = &Q_pm_psi_32;
              (optr->deflator_params).f_final   = &Q_plus_psi;
              (optr->deflator_params).f_initial = &Q_minus_psi;
            } else {
              (optr->deflator_params).f         = &D_psi;
              (optr->deflator_params).f32       = NULL;
              (optr->deflator_params).f_final   = NULL;
              (optr->deflator_params).f_initial = NULL;
            }
          }

        }
        if(optr->solver == CGMMS) {
          if (g_cart_id == 0 && optr->even_odd_flag == 1)
            fprintf(stderr, "CG Multiple mass solver works only without even/odd! Forcing!\n");
          optr->even_odd_flag = 0;
          if (g_cart_id == 0 && optr->DownProp)
            fprintf(stderr, "CGMMS doesn't need AddDownPropagator! Switching Off!\n");
          optr->DownProp = 0;
        }
        if(optr->solver == INCREIGCG) {
          if (g_cart_id == 0 && optr->DownProp) {
            fprintf(stderr,"Warning: When even-odd preconditioning is used, the eigenvalues for +mu and -mu will be little different\n");
            fprintf(stderr,"Incremental EigCG solver will still work however.\n");
          }
          if (g_cart_id == 0 && optr->even_odd_flag == 0)
            fprintf(stderr,"Incremental EigCG solver is added only with Even-Odd preconditioning!. Forcing\n");
          optr->even_odd_flag = 1;
        }
      }
      else if(optr->type == CLOVER) {
        if(optr->c_sw > 0) {
          init_sw_fields();
        }
        optr->applyM = &Msw_full;
        optr->applyQ = &Qsw_full;

        if(optr->solver == EXACTDEFLATEDCG) {
          strcpy( (optr->deflator_params).type_name, "CLOVER" );
        }

        if(optr->even_odd_flag) {
          optr->applyMee    = &Mee_sw_psi;
          optr->applyMeeInv = &Mee_sw_inv_psi;
          optr->applyQp = &Qsw_plus_psi;
          optr->applyQm = &Qsw_minus_psi;
          optr->applyQsq = &Qsw_pm_psi;
          optr->applyMp = &Msw_plus_psi;
          optr->applyMm = &Msw_minus_psi;


          if(optr->solver == EXACTDEFLATEDCG) {
            /* the operators in the deflator */
            if( (optr->deflator_params).symmetric_operator ) {
              (optr->deflator_params).f         = &Qsw_pm_psi;
              (optr->deflator_params).f32       = &Qsw_pm_psi_32;
              (optr->deflator_params).f_final   = &Qsw_plus_psi;
              (optr->deflator_params).f_initial = &Qsw_minus_psi;
            } else {
              (optr->deflator_params).f         = &Qsw_plus_psi;
              (optr->deflator_params).f32       = NULL;
              (optr->deflator_params).f_final   = NULL;
              (optr->deflator_params).f_initial = NULL;
            }
          }

        }
        else {
          optr->applyQp = &Qsw_full_plus_psi;
          optr->applyQm = &Qsw_full_minus_psi;
          optr->applyQsq = &Qsw_full_pm_psi;
          optr->applyMp = &D_psi;
          optr->applyMm = &Msw_full_minus_psi;

          if(optr->solver == EXACTDEFLATEDCG) {
            /* the operators in the deflator */
            if( (optr->deflator_params).symmetric_operator ) { 
              (optr->deflator_params).f         = Qsw_full_pm_psi;
              (optr->deflator_params).f32       = NULL;
              (optr->deflator_params).f_final   = &Qsw_full_plus_psi;
              (optr->deflator_params).f_initial = &Qsw_full_minus_psi;
            } else {
              (optr->deflator_params).f         = &D_psi;
              (optr->deflator_params).f32       = NULL;
              (optr->deflator_params).f_final   = NULL;
              (optr->deflator_params).f_initial = NULL;
            }
          }

        }
        if(optr->solver == CGMMS) {
          if (g_cart_id == 0 && optr->even_odd_flag == 1)
            fprintf(stderr, "CG Multiple mass solver works only without even/odd! Forcing!\n");
          optr->even_odd_flag = 0;
          if (g_cart_id == 0 && optr->DownProp)
            fprintf(stderr, "CGMMS doesn't need AddDownPropagator! Switching Off!\n");
          optr->DownProp = 0;
        }
      }
      else if(optr->type == OVERLAP) {
        optr->even_odd_flag = 0;
        optr->applyMp = &Dov_psi;
        optr->applyQp = &Qov_psi;
      }
      else if(optr->type == DBTMWILSON) {
        optr->even_odd_flag = 1;
        optr->applyDbQsq = &Qtm_pm_ndpsi;
        /* TODO: this should be here!       */
        /* Chi`s-spinors  memory allocation */
        /*       if(init_chi_spinor_field(VOLUMEPLUSRAND/2, 20) != 0) { */
        /*   fprintf(stderr, "Not enough memory for 20 NDPHMC Chi fields! Aborting...\n"); */
        /*   exit(0); */
        /*       } */
      }
      else if(optr->type == DBCLOVER) {
        optr->even_odd_flag = 1;
        optr->applyDbQsq = &Qsw_pm_ndpsi;
        //  optr->applyDbQsq = &Qtm_pm_ndpsi;
      }
      if(optr->external_inverter==QUDA_INVERTER ) {
#ifdef QUDA
        _initQuda();
#else
        if(g_proc_id == 0) {
          fprintf(stderr, "Error: You're trying to use QUDA but this build was not configured for QUDA usage.\n");
          exit(-2);
        }
#endif
      }
    } /* loop over operators */
  }
  return(0);
}  /* end of init_operators */

void dummy_D(spinor * const s, spinor * const r) {
  if(g_proc_id == 0) {
    fprintf(stderr, "dummy_D was called. Was that really intended?\n");
  } 
  return;
}

void dummy_Mee(spinor * const s, spinor * const r, double const d) {
  if(g_proc_id == 0) {
    fprintf(stderr, "dummy_Mee was called. Was that really intended?\n");
  } 
  return;
}

void dummy_M(spinor * const s, spinor * const r, spinor * const t, spinor * const k) {
  if(g_proc_id == 0) {
    fprintf(stderr, "dummy_M was called. Was that really intended?\n");
  } 
  return;
}


void dummy_DbD(spinor * const s, spinor * const r, spinor * const p, spinor * const q) {
  if(g_proc_id == 0) {
    fprintf(stderr, "dummy_DbD was called. Was that really intended?\n");
  }
  return;
}



void op_invert(const int op_id, const int index_start, const int write_prop) {
  operator * optr = &operator_list[op_id];
  double atime = 0., etime = 0., nrm1 = 0., nrm2 = 0.;
  int i;
  optr->iterations = 0;
  optr->reached_prec = -1.;
  g_kappa = optr->kappa;
  boundary(g_kappa);
  
  atime = gettime();
  if(optr->type == TMWILSON || optr->type == WILSON || optr->type == CLOVER) {
    g_mu = optr->mu;
    g_c_sw = optr->c_sw;
    if(optr->type == CLOVER) {
      if (g_cart_id == 0 && g_debug_level > 1) {
        printf("#\n# csw = %e, computing clover leafs\n", g_c_sw);
      }
      init_sw_fields(VOLUME);
      
      sw_term( (const su3**) g_gauge_field, optr->kappa, optr->c_sw); 
      /* this must be EE here!   */
      /* to match clover_inv in Qsw_psi */
      sw_invert(EE, optr->mu);
      /* now copy double sw and sw_inv fields to 32bit versions */
      copy_32_sw_fields();
    }
    
    // this loop is for +mu (i=0) and -mu (i=1)
    // the latter if AddDownPropagator = yes is chosen
    for(i = 0; i < 2; i++) {
      // we need this here again for the sign switch at i == 1
      g_mu = optr->mu;
      if (g_cart_id == 0) {
        printf("#\n# 2 kappa mu = %e, kappa = %e, c_sw = %e\n", g_mu, g_kappa, g_c_sw);
      }
      if(i > 0) {
        zero_spinor_field(optr->prop0, VOLUME/2);
        zero_spinor_field(optr->prop1, VOLUME/2);
      }
      if(optr->type != CLOVER) {
        if(use_preconditioning){
          g_precWS=(void*)optr->precWS;
        }
        else {
          g_precWS=NULL;
        }
        optr->iterations = invert_eo( optr->prop0, optr->prop1, optr->sr0, optr->sr1,
                                      optr->eps_sq, optr->maxiter,
                                      optr->solver, optr->rel_prec,
                                      0, optr->even_odd_flag,optr->no_extra_masses,
                                      optr->extra_masses, optr->solver_params, optr->deflator_params,
                                      optr->id,
                                      optr->external_inverter, optr->sloppy_precision, optr->compression_type);

        /* check result */
        M_full(g_spinor_field[DUM_DERI], g_spinor_field[DUM_DERI+1], optr->prop0, optr->prop1);
      }
      else {
        /* this must be EE here!   */
        /* to match clover_inv in Qsw_psi */
        if(optr->even_odd_flag || optr->solver == DFLFGMRES || optr->solver == DFLGCR)
          sw_invert(EE, optr->mu); //this is needed only when we use even-odd preconditioning
        /* now copy double sw and sw_inv fields to 32bit versions */
        copy_32_sw_fields();
        
        optr->iterations = invert_clover_eo(optr->prop0, optr->prop1, optr->sr0, optr->sr1,
                                            optr->eps_sq, optr->maxiter,
                                            optr->solver, optr->rel_prec,
                                            optr->even_odd_flag, optr->solver_params,
                                            &g_gauge_field, optr->applyQsq, optr->applyQm,
                                            optr->external_inverter, optr->sloppy_precision, optr->compression_type);
        /* check result */
        optr->applyM(g_spinor_field[DUM_DERI], g_spinor_field[DUM_DERI+1], optr->prop0, optr->prop1);
      }

      diff(g_spinor_field[DUM_DERI], g_spinor_field[DUM_DERI], optr->sr0, VOLUME / 2);
      diff(g_spinor_field[DUM_DERI+1], g_spinor_field[DUM_DERI+1], optr->sr1, VOLUME / 2);

      nrm1 = square_norm(g_spinor_field[DUM_DERI], VOLUME / 2, 1);
      nrm2 = square_norm(g_spinor_field[DUM_DERI+1], VOLUME / 2, 1);
      optr->reached_prec = nrm1 + nrm2;

      /* convert to standard normalisation  */
      /* we have to mult. by 2*kappa        */
      if (optr->kappa != 0.) {
        mul_r(optr->prop0, (2*optr->kappa), optr->prop0, VOLUME / 2);
        mul_r(optr->prop1, (2*optr->kappa), optr->prop1, VOLUME / 2);
      }
      if (optr->solver != CGMMS && write_prop) /* CGMMS handles its own I/O */
        optr->write_prop(op_id, index_start, i);
      if(optr->DownProp) {
        optr->mu = -optr->mu;
        dfl_subspace_updated = 1;
      } 
      else 
        break;
    }
  } else if(optr->type == DBTMWILSON || optr->type == DBCLOVER) {
    g_mubar = optr->mubar;
    g_epsbar = optr->epsbar;
    g_c_sw = 0.;
    if(optr->type == DBCLOVER) {
      g_c_sw = optr->c_sw;
      if (g_cart_id == 0 && g_debug_level > 1) {
        printf("#\n# csw = %e, computing clover leafs\n", g_c_sw);
      }
      init_sw_fields(VOLUME);
      sw_term( (const su3**) g_gauge_field, optr->kappa, optr->c_sw); 
      sw_invert_nd(optr->mubar*optr->mubar-optr->epsbar*optr->epsbar);
      /* now copy double sw and sw_inv fields to 32bit versions */
      copy_32_sw_fields();
    }

    for(i = 0; i < SourceInfo.no_flavours; i++) {
      if(optr->type != DBCLOVER) {
        optr->iterations = invert_doublet_eo( optr->prop0, optr->prop1, optr->prop2, optr->prop3,
                                              optr->sr0, optr->sr1, optr->sr2, optr->sr3,
                                              optr->eps_sq, optr->maxiter,
                                              optr->solver, optr->rel_prec,
                                              optr->solver_params, optr->external_inverter, 
                                              optr->sloppy_precision, optr->compression_type);
      }
      else {
        optr->iterations = invert_cloverdoublet_eo( optr->prop0, optr->prop1, optr->prop2, optr->prop3,
                                                    optr->sr0, optr->sr1, optr->sr2, optr->sr3,
                                                    optr->eps_sq, optr->maxiter,
                                                    optr->solver, optr->rel_prec,
                                                    optr->solver_params, optr->external_inverter, 
                                                    optr->sloppy_precision, optr->compression_type);
      }
      g_mu = optr->mubar;
      if(optr->type != DBCLOVER) {
        M_full(g_spinor_field[DUM_DERI+1], g_spinor_field[DUM_DERI+2], optr->prop0, optr->prop1);
      }
      else {
        Msw_full(g_spinor_field[DUM_DERI+1], g_spinor_field[DUM_DERI+2], optr->prop0, optr->prop1);
      }
      assign_add_mul_r(g_spinor_field[DUM_DERI+1], optr->prop2, -optr->epsbar, VOLUME/2);
      assign_add_mul_r(g_spinor_field[DUM_DERI+2], optr->prop3, -optr->epsbar, VOLUME/2);
    
      g_mu = -g_mu;
      if(optr->type != DBCLOVER) {
        M_full(g_spinor_field[DUM_DERI+3], g_spinor_field[DUM_DERI+4], optr->prop2, optr->prop3);
      }
      else {
        Msw_full(g_spinor_field[DUM_DERI+3], g_spinor_field[DUM_DERI+4], optr->prop2, optr->prop3);
      }
      assign_add_mul_r(g_spinor_field[DUM_DERI+3], optr->prop0, -optr->epsbar, VOLUME/2);
      assign_add_mul_r(g_spinor_field[DUM_DERI+4], optr->prop1, -optr->epsbar, VOLUME/2);

      diff(g_spinor_field[DUM_DERI+1], g_spinor_field[DUM_DERI+1], optr->sr0, VOLUME/2); 
      diff(g_spinor_field[DUM_DERI+2], g_spinor_field[DUM_DERI+2], optr->sr1, VOLUME/2); 
      diff(g_spinor_field[DUM_DERI+3], g_spinor_field[DUM_DERI+3], optr->sr2, VOLUME/2); 
      diff(g_spinor_field[DUM_DERI+4], g_spinor_field[DUM_DERI+4], optr->sr3, VOLUME/2); 

      nrm1  = square_norm(g_spinor_field[DUM_DERI+1], VOLUME/2, 1); 
      nrm1 += square_norm(g_spinor_field[DUM_DERI+2], VOLUME/2, 1); 
      nrm1 += square_norm(g_spinor_field[DUM_DERI+3], VOLUME/2, 1); 
      nrm1 += square_norm(g_spinor_field[DUM_DERI+4], VOLUME/2, 1); 
      optr->reached_prec = nrm1;
      g_mu = g_mu1;
      /* For standard normalisation */
      /* we have to mult. by 2*kappa */
      mul_r(g_spinor_field[DUM_DERI], (2*optr->kappa), optr->prop0, VOLUME/2);
      mul_r(g_spinor_field[DUM_DERI+1], (2*optr->kappa), optr->prop1, VOLUME/2);
      mul_r(g_spinor_field[DUM_DERI+2], (2*optr->kappa), optr->prop2, VOLUME/2);
      mul_r(g_spinor_field[DUM_DERI+3], (2*optr->kappa), optr->prop3, VOLUME/2);
      /* the final result should be stored in the convention used in */
      /* hep-lat/0606011                                             */
      /* this requires multiplication of source with                 */
      /* (1+itau_2)/sqrt(2) and the result with (1-itau_2)/sqrt(2)   */

      mul_one_pm_itau2(optr->prop0, optr->prop2, g_spinor_field[DUM_DERI], 
                       g_spinor_field[DUM_DERI+2], -1., VOLUME/2);
      mul_one_pm_itau2(optr->prop1, optr->prop3, g_spinor_field[DUM_DERI+1], 
                       g_spinor_field[DUM_DERI+3], -1., VOLUME/2);
      /* write propagator */
      if(write_prop) optr->write_prop(op_id, index_start, i);

      mul_r(optr->prop0, 1./(2*optr->kappa), g_spinor_field[DUM_DERI], VOLUME/2);
      mul_r(optr->prop1, 1./(2*optr->kappa), g_spinor_field[DUM_DERI+1], VOLUME/2);
      mul_r(optr->prop2, 1./(2*optr->kappa), g_spinor_field[DUM_DERI+2], VOLUME/2);
      mul_r(optr->prop3, 1./(2*optr->kappa), g_spinor_field[DUM_DERI+3], VOLUME/2);

      /* mirror source, but not for volume sources */
      if(i == 0 && SourceInfo.no_flavours == 2 && SourceInfo.type != SRC_TYPE_VOL) {
        if (g_cart_id == 0) {
          fprintf(stdout, "# Inversion done in %d iterations, squared residue = %e!\n",
                  optr->iterations, optr->reached_prec);
        }
        mul_one_pm_itau2(g_spinor_field[DUM_DERI], g_spinor_field[DUM_DERI+2], optr->sr0, optr->sr2, -1., VOLUME/2);
        mul_one_pm_itau2(g_spinor_field[DUM_DERI+1], g_spinor_field[DUM_DERI+3], optr->sr1, optr->sr3, -1., VOLUME/2);

        mul_one_pm_itau2(optr->sr0, optr->sr2, g_spinor_field[DUM_DERI+2], g_spinor_field[DUM_DERI], +1., VOLUME/2);
        mul_one_pm_itau2(optr->sr1, optr->sr3, g_spinor_field[DUM_DERI+3], g_spinor_field[DUM_DERI+1], +1., VOLUME/2);
      }
      /* volume sources need only one inversion */
      else if(SourceInfo.type == SRC_TYPE_VOL) i++;
    }
  } else if(optr->type == OVERLAP) {
    g_mu = 0.;
    m_ov=optr->m;
    eigenvalues(&optr->no_ev, 5000, optr->ev_prec, 0, optr->ev_readwrite, nstore, optr->even_odd_flag);
    /*     ov_check_locality(); */
    /*      index_jd(&optr->no_ev_index, 5000, 1.e-12, optr->conf_input, nstore, 4); */
    ov_n_cheby=optr->deg_poly;

    if(use_preconditioning==1)
      g_precWS=(void*)optr->precWS;
    else
      g_precWS=NULL;


    if(g_debug_level > 3) ov_check_ginsparg_wilson_relation_strong(); 

    invert_overlap(op_id, index_start); 

    if(write_prop) optr->write_prop(op_id, index_start, 0);
  }
  etime = gettime();
  if (g_cart_id == 0 && g_debug_level > 0) {
    fprintf(stdout, "# Inversion done in %d iterations, squared residue = %e!\n",
            optr->iterations, optr->reached_prec);
    fprintf(stdout, "# Inversion done in %1.2e sec. \n", etime - atime);
  }
  return;
}


void op_write_prop(const int op_id, const int index_start, const int append_) {
  operator * optr = &operator_list[op_id];
  char filename[100];
  char ending[15];
  WRITER *writer = NULL;
  int append = 0;
  int status = 0;

  paramsSourceFormat *sourceFormat = NULL;
  paramsPropagatorFormat *propagatorFormat = NULL;
  paramsInverterInfo *inverterInfo = NULL;
  if(optr->type == DBTMWILSON || optr->type == DBCLOVER) {
    strcpy(ending, "hinverted");
  }
  else if(optr->type == OVERLAP) {
    strcpy(ending, "ovinverted");
  }
  else {
    strcpy(ending, "inverted");
  }
 
  // timeslice soruces are usually used for smearing/fuzzing and dilution, this is tracked via SourceInfo.ix in the filename 
  if(SourceInfo.type == SRC_TYPE_POINT || SourceInfo.type == SRC_TYPE_TS) {
    if (PropInfo.splitted) {
      if(T_global > 99) sprintf(filename, "%s.%.4d.%.3d.%.2d.%s", PropInfo.basename, SourceInfo.nstore, SourceInfo.t, SourceInfo.ix, ending);
      else sprintf(filename, "%s.%.4d.%.2d.%.2d.%s", PropInfo.basename, SourceInfo.nstore, SourceInfo.t, SourceInfo.ix, ending);
    }
    else {
      if(T_global > 99) sprintf(filename, "%s.%.4d.%.3d.%s", PropInfo.basename, SourceInfo.nstore, SourceInfo.t, ending);
      else sprintf(filename, "%s.%.4d.%.2d.%s", PropInfo.basename, SourceInfo.nstore, SourceInfo.t, ending);
    }
  }
  else if (SourceInfo.type == SRC_TYPE_VOL) {
    sprintf(filename, "%s.%.4d.%.5d.%s", PropInfo.basename, SourceInfo.nstore, SourceInfo.sample, ending);
  }
  else if(SourceInfo.type == SRC_TYPE_PION_TS || SourceInfo.type == SRC_TYPE_GEN_PION_TS) {
    sprintf(filename, "%s.%.4d.%.5d.%.2d.%s", PropInfo.basename, SourceInfo.nstore, SourceInfo.sample, SourceInfo.t, ending);
  }

  if(!PropInfo.splitted || append_)
    append = 1;
  /* the 1 is for appending */
  construct_writer(&writer, filename, append);
  if (PropInfo.splitted || SourceInfo.ix == index_start) {
    write_propagator_type(writer, 0);

    inverterInfo = construct_paramsInverterInfo(optr->reached_prec, optr->iterations, 
                                                optr->solver, optr->no_flavours);
    write_spinor_info(writer, PropInfo.format, inverterInfo, append);
    free(inverterInfo);
  }
  /* write the source depending on format */
  /* to be fixed for 2 fl tmwilson        */
  if (PropInfo.format == 1) {
    sourceFormat = construct_paramsSourceFormat(SourceInfo.precision, optr->no_flavours, 4, 3);
    write_source_format(writer, sourceFormat);
    status = write_spinor(writer, &operator_list[op_id].sr0, &operator_list[op_id].sr1, 1, SourceInfo.precision);
    if(status != 0) {
      fprintf(stderr, "[op_write_prop] Error from write_spinor while writing source, status was %d\n", status);
    }
    if(optr->no_flavours == 2) {
      status = write_spinor(writer, &operator_list[op_id].sr2, &operator_list[op_id].sr3, 1, SourceInfo.precision);
      if(status != 0) {
        fprintf(stderr, "[op_write_prop] Error from write_spinor while writing source for second flavor, status was %d\n", status);
      }
    }
    free(sourceFormat);
  }
  propagatorFormat = construct_paramsPropagatorFormat(optr->prop_precision, optr->no_flavours);
  write_propagator_format(writer, propagatorFormat);
  free(propagatorFormat);

  if(optr->no_flavours == 2) {
    status = write_spinor(writer, &operator_list[op_id].prop2, &operator_list[op_id].prop3, 1, optr->prop_precision);
    if(status != 0) {
      fprintf(stderr, "[op_write_prop] Error from write_spinor while writing propagator , status was %d\n", status);
    }
  }
  status = write_spinor(writer, &operator_list[op_id].prop0, &operator_list[op_id].prop1, 1, optr->prop_precision);
  if(status != 0) {
    fprintf(stderr, "[op_write_prop] Error from write_spinor while writing propagator , status was %d\n", status);
  }

  destruct_writer(writer);
  return;
}
