/* $Id$ */
/*
  Program  to read a 
  tmqcd configuration into memory. - based on ILDG CP-PACS
   (see ToSTDConf.c and web pages)

 Writes float to D_ukqcd

*/

#include<stdlib.h>
#include<stdio.h>
#include<errno.h>
#include"io.h"

int main()
{
  char filename[150] ;
 /* TM    double U[NX][NY][NZ][NT][4][2][3][2]; mu: TXYZ*/
  const int NX = 24 ; 
  const int NY = 24 ; 
  const int NZ = 24 ; 
  const int NT = 48 ; 

  int dim = NX*NY*NZ*NT*2*9*4 ; 
  int dimukqcd = NX*NY*NZ*2*6*4 ;  /*  dimension of write 2 cols case */
  int dimtm = NT*2*6*4 ;  /*  dimension of write 2 cols case */
  int nxyz = NX*NY*NZ ; 
  int iri,irv,idim,icol,irow, ixyz,ix,iy,iz,it, idimtm;
  int iuk, icp, iout;
  double xnr11,xnr12,xnr22,xtemp1,xtemp2;
  float *config=NULL  ;

  FILE *fp;
 /* UKQCD   float U[NT][NZ][NY][NX][4][3][2][2] and separate t-slices */
  float *ukqcd  ;
  char fileout[] = "D_ukqcd"; 
  char fname_t[100];

/* read in file name from stdin - to be checked in ILDF_read with header */
  scanf("%s",filename);


  printf("Read in a tmqcd configuration\n") ; 
  printf("filename = %s\n",filename); 
  fflush(stdout);

  config = (float *) malloc((size_t) dim * sizeof(float) ) ; 
  if(errno == ENOMEM) {
    fprintf(stderr, "Error reserving space for config\n"); 
    return(-1);
  }

  read_lime_gauge_field_doubleprec(config, filename, NT, NX, NY, NZ);

  for ( iri=0; iri<6; iri++) {
    irv=iri*2;
    printf("iri=%d  value %lf",iri,*(config+irv));
    printf(" value %lf\n",*(config+irv+1));
  }

/* test SU(3)   12  34  45  
                67  89 1011  */

  xnr11=0;
  xnr12=0;
  xnr22=0;

  for ( iri=0; iri<6; iri++) {
    xtemp1=*(config+iri);
    xtemp2=*(config+iri+6);
    xnr11+=xtemp1*xtemp1;
    xnr12+=xtemp1*xtemp2;
    xnr22+=xtemp2*xtemp2;
  }
  printf(" xnr11, xnr12, xnr22 =%lf,%lf,%lf\n",xnr11,xnr12,xnr22);


  ukqcd = (float *) malloc((size_t) dimukqcd * sizeof(float) ) ; 
  if(errno == ENOMEM) {
    fprintf(stderr, "Error reserving space for ukqcd\n"); 
    return(-1);
  }

  for ( it=0; it < NT; it++) {

    sprintf(fname_t,"%s_T%02d",fileout,it);

    printf("fname_t is %s\n",fname_t);
    fp= fopen(fname_t, "wb");
    if( fp == NULL ) {
      fprintf(stderr, "Error opening binary file to write\n"); 
      return(-1) ; 
    }

    /* ukqcd order here  mu XYZT */  

    iout=0;

    for (iz = 0; iz < NZ; iz++) {
      for (iy = 0; iy < NY; iy++) {
	for (ix = 0; ix < NX; ix++) {
	  /* 	  ixyz=(ix*NY+iy)*NZ+iz; */
	  /* ILDG has order TZYX */
	  ixyz = (iz*NY+iy)*NX+ix;

	  for (idim = 0; idim < 4; idim++) {
	    /* ILDG has mu XYZT */
	    idimtm = idim;

	    for (icol = 0; icol < 3; icol++) {
	      for (irow = 0; irow < 2; irow++) {
		for (iri = 0; iri < 2; iri++) {
		  icp = iri+icol*2+irow*6+idimtm*12+ixyz*dimtm+it*48 ;
		  *(ukqcd+iout) =*(config+icp);
		  iout++;
		}
	      }
	    }
	  }
	}
      }
    }

    if( fwrite(ukqcd,sizeof(float),dimukqcd,fp) != dimukqcd ) {
      fprintf(stderr, "Error writing binary file\n"); 
      return(-1) ; 
    }
    fclose(fp);
  }
  return(0) ; 
}

