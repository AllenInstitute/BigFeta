/** @file bigfeta_dist_solve.c
  * @brief Main method for distributed solve of Kx=Lm
  *
  * Reads A from file, reads regularizations and constraints, computes K, solves for x.
*/

/**
 * \param help message
 */
static char help[] = "usage:\n"
  "em_dist_solve -input <input_file_path> -output <output_file_path> <ksp options>\n"
  "ksp options:\n"
  "  direct solve with pastix: -ksp_type preonly -pc_type lu\n"
  "see:\n"
  "  https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPPREONLY.html\n"
  "  https://www.mcs.anl.gov/petsc/petsc-3.11/docs/manualpages/Mat/MATSOLVERPASTIX.html#MATSOLVERPASTIX\n";

#include <sys/resource.h>
#include <stdio.h>
#include <petsctime.h>
#include "bigfeta_dist.h"
#include "hw_config.h"

/*! @brief main for BigFeta distributed solve
 * usage : em_dist_solve -input <input_file_path> -output <output_file_path> <ksp options>
 */
int
main (int argc, char **args)
{
  KSP ksp;			//linear solver context
  PetscMPIInt rank, size;	//MPI rank and size
  char fileinarg[PETSC_MAX_PATH_LEN];	//input file name
  char sln_output[PETSC_MAX_PATH_LEN];	//input file name
  char *dir, *sln_input, **csrnames;	//various strings
  int nfiles;			//number of files
  PetscInt **metadata;		//metadata read from index.txt
  PetscInt local_firstfile, local_lastfile;	//local file indices
  PetscInt global_nrow, global_ncol, global_nnz;	//global index info
  PetscInt local_nrow, local_nnz, local_row0;	//local  index info
  PetscInt local_rowN;
  PetscInt *local_indptr, *local_jcol;	//index arrays for local CSR
  PetscScalar *local_data, *local_weights;	//data for local CSR and weights
  PetscScalar **local_rhs;
  PetscBool flg, trunc;		//boolean used in checking command line
  PetscErrorCode ierr;		//error code that gets passed around.
  PetscLogDouble tall0, tall1;	//timers
  int i;
  Mat A, W, ATW, K, L;		//K and the matrices that build it
  Vec ATWRHS[2];
  Vec rhs[2], x0[2], Lm[2], x[2];	//vectors associated with the solve(s)
  PetscLogDouble t0, t1;	//some timers
  PetscReal norm[2], norm2[2];
  PetscLogStage stage;
  int mpisupp;

  /*  Command line handling and setup  */
  MPI_Init_thread (0, 0, MPI_THREAD_MULTIPLE, &mpisupp);

  ierr = PetscInitialize (&argc, &args, (char *) 0, help);
  if (ierr)
    return ierr;
  ierr =
    PetscOptionsGetString (NULL, NULL, "-input", fileinarg,
			   PETSC_MAX_PATH_LEN, &flg);
  CHKERRQ (ierr);
  ierr =
    PetscOptionsGetString (NULL, NULL, "-output", sln_output,
			   PETSC_MAX_PATH_LEN, &flg);
  CHKERRQ (ierr);
  ierr = PetscOptionsHasName (NULL, NULL, "-truncate", &trunc);
  CHKERRQ (ierr);

  PetscTime (&tall0);
  sln_input = strdup (fileinarg);
  dir = strdup (dirname (fileinarg));
  ierr = MPI_Comm_rank (PETSC_COMM_WORLD, &rank);
  CHKERRQ (ierr);
  ierr = MPI_Comm_size (PETSC_COMM_WORLD, &size);
  CHKERRQ (ierr);

  PetscLogStageRegister ("Distributed Read", &stage);
  PetscLogStagePush (stage);
  /*  count the numberof hdf5 CSR files  */
  ierr = CountFiles (PETSC_COMM_WORLD, sln_input, &nfiles);
  CHKERRQ (ierr);
  /*  allocate for file names and metadata  */
  csrnames = (char **) malloc (nfiles * sizeof (char *));
  metadata = (PetscInt **) malloc (nfiles * sizeof (PetscInt *));
  for (i = 0; i < nfiles; i++)
    {
      csrnames[i] = (char *) malloc (PETSC_MAX_PATH_LEN * sizeof (char));
      metadata[i] = (PetscInt *) malloc (4 * sizeof (PetscInt));
    }
  /*  read in the metadata  */
  ierr =
    ReadMetadata (PETSC_COMM_WORLD, sln_input, nfiles, csrnames, metadata);
  CHKERRQ (ierr);

  /*  what does the hardware look like?  */
  node * nodes;
  int count, j, nnodes, thisnode, noderank;
  MPI_Comm MPI_COMM_NODE;
  nodes = hw_config (MPI_COMM_WORLD, &nnodes, &thisnode);
  split_files (nodes, nnodes, nfiles);
  MPI_Comm_split (MPI_COMM_WORLD, thisnode, rank, &MPI_COMM_NODE);
  MPI_Comm_rank (MPI_COMM_NODE, &noderank);
  /*  what files will this rank read  */
  count = 0;
  for (i=0; i<nnodes; i++)
  {
    if (rank == 0)
    {
	    disp_node (nodes[i]);
    }
    for (j=0; j<nodes[i].nrank; j++)
    {
      if ((thisnode == i) && (noderank == j))
      {
	      local_firstfile = count;
	      local_lastfile = count + nodes[i].files[j] - 1;
      }
      count += nodes[i].files[j];
    }
  }

  //ierr =
  //  SetFiles (PETSC_COMM_WORLD, nfiles, &local_firstfile, &local_lastfile);
  //CHKERRQ (ierr);
  /*  how many rows and nnz per worker  */
  GetGlobalLocalCounts (nfiles, metadata, local_firstfile, local_lastfile,
			&global_nrow, &global_ncol, &global_nnz, &local_nrow,
			&local_nnz, &local_row0);

  if (rank == 0)
    {
      printf ("input file: %s\n", sln_input);
      printf ("%d ranks will handle %d files\n", size, nfiles);
    }

  /*  how many solves */
  PetscInt nsolve;
  ierr = CountSolves (PETSC_COMM_WORLD, sln_input, &nsolve);
  CHKERRQ (ierr);

  /*  allocate space for local CSR arrays  */
  local_indptr = (PetscInt *) calloc (local_nrow + 1, sizeof (PetscInt));
  local_jcol = (PetscInt *) calloc (local_nnz, sizeof (PetscInt));
  local_data = (PetscScalar *) calloc (local_nnz, sizeof (PetscScalar));
  local_weights = (PetscScalar *) calloc (local_nrow, sizeof (PetscScalar));
  local_rhs = (PetscScalar **) malloc (nsolve * sizeof (PetscInt *));
  for (i = 0; i < nsolve; i++)
    {
      local_rhs[i] =
	(PetscScalar *) calloc (local_nrow, sizeof (PetscScalar));
    }
  /*  read in local hdf5 files and concatenate into CSR arrays  */
  ierr =
    ReadLocalCSR (PETSC_COMM_SELF, csrnames, local_firstfile, local_lastfile,
		  nsolve, local_indptr, local_jcol, local_data, local_weights,
		  local_rhs);
  CHKERRQ (ierr);
  /*  Create distributed A!  */
  MatCreateMPIAIJWithArrays (PETSC_COMM_WORLD, local_nrow, PETSC_DECIDE,
  			     global_nrow, global_ncol, local_indptr,
  	                     local_jcol, local_data, &A);
  free (local_jcol);
  free (local_data);
  free (local_indptr);
  if (rank == 0)
    {
      printf ("A matrix created\n");
    }
  PetscLogStagePop ();

  /*  Create distributed rhs  */
  PetscScalar *u_local_ptr;
  for (i = 0; i < nsolve; i++)
    {
      VecCreate (PETSC_COMM_WORLD, &rhs[i]);
      VecSetType (rhs[i], VECMPI);
      VecSetSizes (rhs[i], local_nrow, global_nrow);
      VecSet (rhs[i], 0.0);
      VecGetArray (rhs[i], &u_local_ptr);
      for (int j = 0; j < local_nrow; j++)
	{
	  u_local_ptr[j] = local_rhs[i][j];
	}
      ierr = VecRestoreArray (rhs[i], &u_local_ptr);
      VecAssemblyBegin (rhs[i]);
      VecAssemblyEnd (rhs[i]);
    }
  free (u_local_ptr);
  for (i = 0; i < nsolve; i++)
    {
      free (local_rhs[i]);
    }
  free (local_rhs);

  PetscLogStageRegister ("Create K", &stage);
  PetscLogStagePush (stage);

  /*  Create the W matrix  */
  ierr =
    CreateW (PETSC_COMM_WORLD, local_weights, local_nrow, local_row0,
	     global_nrow, &W);
  free (local_weights);

  /*  Start the K matrix with K = AT*W*A */
  ierr = MatTransposeMatMult (A, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &ATW);
  CHKERRQ (ierr);
  ierr = MatMatMult (ATW, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &K);
  CHKERRQ (ierr);

  /*  find out how the rows are distributed   */
  MatGetOwnershipRange (K, &local_row0, &local_rowN);
  MatGetSize (K, &global_nrow, NULL);
  local_nrow = local_rowN - local_row0;

  /*  read in the regularization   */
  ierr =
    CreateL (PETSC_COMM_WORLD, sln_input, local_nrow, global_nrow, trunc, &L);
  if (rank == 0)
    {
      printf ("L created\n");
    }

  /*   K = K+L   */
  MatSetOption (K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  ierr = MatAXPY (K, (PetscScalar) 1.0, L, SUBSET_NONZERO_PATTERN);
  CHKERRQ (ierr);
  MatSetOption (K, MAT_SYMMETRIC, PETSC_TRUE);
  // appropriate for Cholesky
  // ierr = MatConvert (K, MATMPISBAIJ, MAT_INPLACE_MATRIX, &K);
  // CHKERRQ (ierr);

  if (rank == 0)
    {
      printf ("K matrix created\n");
    }
  PetscLogStagePop ();

  PetscLogStageRegister ("Get x0", &stage);
  PetscLogStagePush (stage);

  /*  Read in the x0 vector(s)  */
  ierr =
    Readx0 (PETSC_COMM_WORLD, sln_input, local_nrow, global_nrow, nsolve,
	    trunc, x0);
  CHKERRQ (ierr);

  /*  Create Lm vectors  */
  for (i = 0; i < nsolve; i++)
    {
      ierr = VecDuplicate (x0[i], &Lm[i]);
      ierr = VecDuplicate (x0[i], &ATWRHS[i]);
      CHKERRQ (ierr);
      ierr = MatMult (L, x0[i], Lm[i]);
      CHKERRQ (ierr);
      ierr = MatMult (ATW, rhs[i], ATWRHS[i]);
      CHKERRQ (ierr);
      ierr = VecAXPY (Lm[i], 1.0, ATWRHS[i]);
      CHKERRQ (ierr);
      VecDestroy (&ATWRHS[i]);
    }

  if (rank == 0)
    {
      printf ("Lm(s) created\n");
    }

  PetscLogStagePop ();

  PetscLogStageRegister ("Solve", &stage);
  PetscLogStagePush (stage);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate (PETSC_COMM_WORLD, &ksp);
  CHKERRQ (ierr);
  ierr = KSPSetOperators (ksp, K, K);
  CHKERRQ (ierr);
  ierr = KSPSetFromOptions (ksp);
  CHKERRQ (ierr);
  ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE);
  CHKERRQ (ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  char xname[20];
  PetscViewer viewer;
  if (rank == 0)
    {
      CopyDataSetstoSolutionOut (PETSC_COMM_SELF, sln_input, sln_output);
    }
  ierr =
    PetscViewerHDF5Open (PETSC_COMM_WORLD, sln_output, FILE_MODE_APPEND,
			 &viewer);
  CHKERRQ (ierr);
  for (i = 0; i < nsolve; i++)
    {
      PetscTime (&t0);
      ierr = VecDuplicate (x0[i], &x[i]);
      CHKERRQ (ierr);
      ierr = KSPSolve (ksp, Lm[i], x[i]);
      CHKERRQ (ierr);
      PetscTime (&t1);
      if (rank == 0)
	{
	  printf ("solve %d: %0.1f sec\n", i, t1 - t0);
	}
      sprintf (xname, "x_%d", i);
      ierr = PetscObjectSetName ((PetscObject) x[i], xname);
      CHKERRQ (ierr);
      ierr = VecView (x[i], viewer);
      CHKERRQ (ierr);
    }
  ierr = PetscViewerDestroy (&viewer);
  CHKERRQ (ierr);
  PetscLogStagePop ();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscReal *precision = (PetscReal *) calloc (nsolve, sizeof (PetscReal));
  char results_out[1000], strout[1000], tmp[400];

  sprintf (strout, " precision [norm(Kx-Lm)/norm(Lm)] =");
  sprintf (results_out, "{\"precision\": [");
  for (i = 0; i < nsolve; i++)
    {
      //from here on, x0 is replaced by err
      ierr = VecScale (Lm[i], (PetscScalar) - 1.0);
      CHKERRQ (ierr);
      ierr = MatMultAdd (K, x[i], Lm[i], x0[i]);
      CHKERRQ (ierr);		//err0 = Kx0-Lm0
      ierr = VecNorm (x0[i], NORM_2, &norm[i]);
      CHKERRQ (ierr);		//NORM_2 denotes sqrt(sum_i |x_i|^2)
      ierr = VecNorm (Lm[i], NORM_2, &norm2[i]);
      CHKERRQ (ierr);		//NORM_2 denotes sqrt(sum_i |x_i|^2)
      precision[i] = norm[i] / norm2[i];
      sprintf (tmp, " %0.1e", precision[i]);
      strcat (strout, tmp);
      strcat (results_out, tmp);
      if (i != nsolve - 1)
	{
	  strcat (strout, ",");
	  strcat (results_out, ",");
	}
    }
  strcat (strout, "\n");
  strcat (results_out, "],");

  MatDestroy (&K);

  PetscInt mA, nA, c0, cn;
  ierr = MatGetSize (A, &mA, &nA);
  CHKERRQ (ierr);
  ierr = MatGetOwnershipRange (A, &c0, &cn);
  CHKERRQ (ierr);
  strcat (strout, " error     [norm(Ax-b)] =");
  strcat (results_out, "\"error\": [");
  for (i = 0; i < nsolve; i++)
    {
      ierr = VecCreate (PETSC_COMM_WORLD, &x0[i]);
      CHKERRQ (ierr);
      ierr = VecSetType (x0[i], VECMPI);
      CHKERRQ (ierr);
      ierr = VecSetSizes (x0[i], cn - c0, mA);
      CHKERRQ (ierr);
      ierr = MatMult (A, x[i], x0[i]);
      CHKERRQ (ierr);		//err0 = Ax0
      ierr = VecAXPY (x0[i], -1.0, rhs[i]);
      CHKERRQ (ierr);
      ierr = VecNorm (x0[i], NORM_2, &norm[i]);
      CHKERRQ (ierr);		//NORM_2 denotes sqrt(sum_i |x_i|^2)
      sprintf (tmp, " %0.1f", norm[i]);
      strcat (strout, tmp);
      strcat (results_out, tmp);
      if (i != nsolve - 1)
	{
	  strcat (strout, ",");
	  strcat (results_out, ",");
	}
    }
  strcat (strout, "\n");
  strcat (results_out, "],");

  //calculate the mean and standard deviation
  PetscReal errmean[2], errstd[2];
  PetscInt sz;
  strcat (strout, " [mean(Ax) +/- std(Ax)] =");
  strcat (results_out, " \"err\": [");
  for (i = 0; i < nsolve; i++)
    {
      VecGetSize (x0[i], &sz);
      VecSum (x0[i], &errmean[i]);
      errmean[i] /= sz;
      VecShift (x0[i], -1.0 * errmean[i]);
      VecNorm (x0[i], NORM_2, &errstd[i]);
      errstd[i] /= sqrt (sz);
      sprintf (tmp, " %0.2f +/- %0.2f", errmean[i], errstd[i]);
      strcat (strout, tmp);
      sprintf (tmp, "[%0.2f, %0.2f]", errmean[i], errstd[i]);
      strcat (results_out, tmp);
      if (i != nsolve - 1)
	{
	  strcat (strout, ",");
	  strcat (results_out, ",");
	}
    }
  strcat (strout, "\n");
  strcat (results_out, "]}");
  if (rank == 0)
    {
      printf (strout);
      hid_t fileout = H5Fopen (sln_output, H5F_ACC_RDWR, H5P_DEFAULT);
      hid_t memtype = H5Tcopy (H5T_C_S1);
      hsize_t dims[1] = { 1 };
      hid_t space = H5Screate_simple (1, dims, NULL);
      H5Tset_size (memtype, 1000);
      hid_t dsetout =
	H5Dcreate (fileout, "results", memtype, space, H5P_DEFAULT,
		   H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite (dsetout, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, results_out);
      H5Dclose (dsetout);
      H5Sclose (space);
      H5Tclose (memtype);
      H5Fclose (fileout);
      ierr = PetscViewerDestroy (&viewer);
      CHKERRQ (ierr);
    }

  //cleanup
  for (i = 0; i < nsolve; i++)
    {
      VecDestroy (&Lm[i]);
      VecDestroy (&x[i]);
      VecDestroy (&x0[i]);
      VecDestroy (&rhs[i]);
    }
  MatDestroy (&A);
  MatDestroy (&ATW);
  KSPDestroy (&ksp);
  free (dir);
  free (sln_input);
  for (i = 0; i < nfiles; i++)
    {
      free (csrnames[i]);
      free (metadata[i]);
    }
  free (csrnames);
  free (metadata);

  PetscTime (&tall1);
  if (rank == 0)
    {
      printf ("rank %d total time: %0.1f sec\n", rank, tall1 - tall0);
    }
  ierr = PetscFinalize ();
  CHKERRQ (ierr);
  return ierr;
}
