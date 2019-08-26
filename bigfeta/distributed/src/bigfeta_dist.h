/** @file bigfeta_dist.h
  *
  *
*/
#include <petscsys.h>
#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <petscviewerhdf5.h>
#include <libgen.h>
#include <hdf5.h>

/*! Rank 0 process counts the number of files from index.txt and broadcasts the result 
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] indexname The full path to index.txt, given as command-line argument with -f.
 * @param[out] *nfiles The result of the count, returned to main().
 */
PetscErrorCode
CountFiles (MPI_Comm COMM, char indexname[], int *nfiles)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  ierr = MPI_Comm_rank (COMM, &rank);
  CHKERRQ (ierr);

  if (rank == 0)
    {
      hid_t file, space, dset;
      hsize_t dims[1];

      file = H5Fopen (indexname, H5F_ACC_RDONLY, H5P_DEFAULT);
      dset = H5Dopen (file, "datafile_names", H5P_DEFAULT);
      space = H5Dget_space (dset);
      H5Sget_simple_extent_dims (space, dims, NULL);
      *nfiles = dims[0];
      H5Dclose (dset);
      H5Sclose (space);
      H5Fclose (file);
    }
  MPI_Bcast (nfiles, 1, MPI_INT, 0, COMM);
  return ierr;
}

/*! @brief Rank 0 processor reads metadata and broadcasts
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] indexname The full path to <solution_input>.h5, given as command-line argument with -input.
 * @param[in] nfiles The number of files listed in solution_input.h5 file, read from previous CountFiles() call.
 * @param[out] *csrnames An array of file names from <solution_input>.h5, populated by this function.
 * @param[out] *metadata An array of metadata from index.txt, populated by this function.
 */
PetscErrorCode
ReadMetadata (MPI_Comm COMM, char indexname[], int nfiles, char *csrnames[],
	      PetscInt ** metadata)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  int i;
  ierr = MPI_Comm_rank (COMM, &rank);
  CHKERRQ (ierr);

  if (rank == 0)
    {
      char **rdata;
      hid_t file, filetype, memtype, space, dset;
      hsize_t dims[1];
      char *dir, *tmp;

      tmp = strdup (indexname);
      dir = strdup (dirname (tmp));
      file = H5Fopen (indexname, H5F_ACC_RDONLY, H5P_DEFAULT);
      dset = H5Dopen (file, "datafile_names", H5P_DEFAULT);
      filetype = H5Dget_type (dset);
      space = H5Dget_space (dset);
      H5Sget_simple_extent_dims (space, dims, NULL);
      rdata = (char **) malloc (dims[0] * sizeof (char *));
      memtype = H5Tcopy (H5T_C_S1);
      H5Tset_size (memtype, H5T_VARIABLE);
      H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
      for (i = 0; i < nfiles; i++)
	{
	  sprintf (csrnames[i], "%s/%s", dir, rdata[i]);
	}
      H5Dvlen_reclaim (memtype, space, H5P_DEFAULT, rdata);
      free (rdata);
      H5Dclose (dset);
      H5Sclose (space);

      PetscInt *row, *mxcol, *mncol, *nnz;
      row = (PetscInt *) malloc (nfiles * sizeof (PetscInt));
      mxcol = (PetscInt *) malloc (nfiles * sizeof (PetscInt));
      mncol = (PetscInt *) malloc (nfiles * sizeof (PetscInt));
      nnz = (PetscInt *) malloc (nfiles * sizeof (PetscInt));
      dset = H5Dopen (file, "datafile_nrows", H5P_DEFAULT);
      H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, row);
      dset = H5Dopen (file, "datafile_mincol", H5P_DEFAULT);
      H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, mncol);
      dset = H5Dopen (file, "datafile_maxcol", H5P_DEFAULT);
      H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, mxcol);
      dset = H5Dopen (file, "datafile_nnz", H5P_DEFAULT);
      H5Dread (dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, nnz);
      for (i = 0; i < nfiles; i++)
	{
	  metadata[i][0] = row[i];
	  metadata[i][1] = mncol[i];
	  metadata[i][2] = mxcol[i];
	  metadata[i][3] = nnz[i];
	}
      free (row);
      free (mncol);
      free (mxcol);
      free (nnz);
      H5Dclose (dset);
      H5Tclose (filetype);
      H5Tclose (memtype);
      H5Fclose (file);
      free (dir);
      free (tmp);
    }
  for (i = 0; i < nfiles; i++)
    {
      MPI_Bcast (csrnames[i], PETSC_MAX_PATH_LEN, MPI_CHAR, 0, COMM);
      MPI_Bcast (metadata[i], 4, MPIU_INT, 0, COMM);
    }
  return ierr;
}

/*! @brief output file of a solve is mostly a copy of the input file
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] indexname The full path to <solution_input>.h5, given as command-line argument with -input.
 * @param[in] outputname The full path to <solution_output>.h5, given as command-line argument with -output.
 */
void
CopyDataSetstoSolutionOut (MPI_Comm COMM, char indexname[], char outputname[])
{
  hid_t filein, fileout, filetype, memtype, space, dset, dsetout;
  hsize_t dims[1];
  int nds = 8;
  const char *copyids[8] = { "input_args", "resolved_tiles",
    "datafile_names", "datafile_maxcol", "datafile_mincol",
    "datafile_nnz", "datafile_nrows", "reg", "solve_list"
  };
  char **rdata;
  int i;

  filein = H5Fopen (indexname, H5F_ACC_RDONLY, H5P_DEFAULT);
  fileout = H5Fcreate (outputname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (i = 0; i < nds; i++)
    {
      dset = H5Dopen (filein, copyids[i], H5P_DEFAULT);
      filetype = H5Dget_type (dset);
      space = H5Dget_space (dset);
      H5Sget_simple_extent_dims (space, dims, NULL);
      rdata = (char **) malloc (dims[0] * sizeof (char *));
      memtype = H5Dget_type (dset);
      if (i < 3)
	{
	  memtype = H5Tcopy (H5T_C_S1);
	  H5Tset_size (memtype, H5T_VARIABLE);
	}
      H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
      dsetout =
	H5Dcreate (fileout, copyids[i], filetype, space, H5P_DEFAULT,
		   H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite (dsetout, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
      H5Dclose (dset);
      H5Dclose (dsetout);
      H5Tclose (filetype);
      H5Tclose (memtype);
      H5Dvlen_reclaim (memtype, space, H5P_DEFAULT, rdata);
      free (rdata);
    }
  H5Fclose (filein);
  H5Fclose (fileout);
}

/*! Split the list of files roughly evenly amongst all the workers.
 * @param[in] COMM The MPI communicator, probably PETSC_COMM_WORLD.
 * @param[in] nfiles The number of lines in index.txt, read from previous CountFiles() call.
 * @param[out] *firstfile The index of the first file for this worker.
 * @param[out] *lastfile The index of the first file for this worker.
*/
PetscErrorCode
SetFiles (MPI_Comm COMM, int nfiles, PetscInt * firstfile,
	  PetscInt * lastfile)
{
  PetscErrorCode ierr;
  PetscMPIInt rank, size;
  float avg;
  int i, rem, last;

  ierr = MPI_Comm_rank (COMM, &rank);
  CHKERRQ (ierr);
  ierr = MPI_Comm_size (COMM, &size);
  CHKERRQ (ierr);
  if (nfiles <= size)
    {
      for (i = 0; i < size; i++)
	{
	  if (rank == i)
	    {
	      *firstfile = i;
	      if (i > nfiles - 1)
		{
		  *lastfile = i - 1;
		}
	      else
		{
		  *lastfile = i;
		}
	    }
	}
    }
  else
    {
      avg = nfiles / (float) size;
      rem = nfiles % size;
      last = 0;
      for (i = 0; i < size; i++)
	{
	  if (rank == i)
	    {
	      *firstfile = last;
	    }
	  if (rem-- > 0)
	    {
	      last = last + ceil (avg);
	    }
	  else
	    {
	      last = last + floor (avg);
	    }
	  if (rank == i)
	    {
	      *lastfile = last - 1;
	    }
	}
    }
  return ierr;
}

/*! Read data from a PetscViewer into a Vec (scalar) object. This version good for single-rank reads.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param[in] viewer A PetscViewer instance.
 * @param[in] *varname The new vector will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param[out] *newvec The new vector object.
 * @param[out] *n The number of entries in the new object.
*/
PetscErrorCode
ReadVec (MPI_Comm COMM, PetscViewer viewer, char *varname, Vec * newvec,
	 PetscInt * n)
{
  PetscErrorCode ierr;
  char name[256];
  sprintf (name, "%s", varname);
  ierr = VecCreate (COMM, newvec);
  CHKERRQ (ierr);
  ierr = PetscObjectSetName ((PetscObject) * newvec, name);
  CHKERRQ (ierr);
  ierr = VecLoad (*newvec, viewer);
  CHKERRQ (ierr);
  ierr = VecGetSize (*newvec, n);
  CHKERRQ (ierr);
  return ierr;
}

/*! Read data from a PetscViewer into a Vec (scalar) object. This version good for anticipating allocation across nodes.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param[in] viewer A PetscViewer instance.
 * @param[in] *varname The new vector will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param[out] *newvec The new vector object.
 * @param[out] *n The number of entries in the new object.
 * @param[in] nlocal The number of entries the local rank will own.
 * @param[in] nglobal The total number of expected entries in newvec.
 * @param[in] trunc Boolean whether to truncate the imported data.
*/
PetscErrorCode
ReadVecWithSizes (MPI_Comm COMM, PetscViewer viewer, char *varname,
		  Vec * newvec, PetscInt * n, PetscInt nlocal,
		  PetscInt nglobal, PetscBool trunc)
{
  PetscErrorCode ierr;
  Vec tmpvec;
  PetscInt i, ntmp, low, high, *indx;
  PetscScalar *vtmp;
  char name[256];

  sprintf (name, "%s", varname);
  ierr = VecCreate (COMM, newvec);
  CHKERRQ (ierr);
  ierr = VecSetSizes (*newvec, nlocal, nglobal);
  CHKERRQ (ierr);
  ierr = VecSetType (*newvec, VECMPI);
  CHKERRQ (ierr);
  if (trunc)
    {
      ierr = VecCreate (MPI_COMM_SELF, &tmpvec);
      CHKERRQ (ierr);
      ierr = VecSetType (tmpvec, VECSEQ);
      CHKERRQ (ierr);
      ierr = PetscObjectSetName ((PetscObject) tmpvec, name);
      CHKERRQ (ierr);
      ierr = VecLoad (tmpvec, viewer);
      CHKERRQ (ierr);
      ierr = VecGetOwnershipRange (*newvec, &low, &high);
      CHKERRQ (ierr);
      indx = (PetscInt *) malloc ((high - low) * sizeof (PetscInt));
      vtmp = (PetscScalar *) malloc ((high - low) * sizeof (PetscScalar));
      for (i = low; i < high; i++)
	{
	  indx[i - low] = i;
	}
      ierr = VecGetValues (tmpvec, high - low, indx, vtmp);
      CHKERRQ (ierr);
      ierr = VecSetValues (*newvec, high - low, indx, vtmp, INSERT_VALUES);
      CHKERRQ (ierr);
      free (indx);
      free (vtmp);
      ierr = VecAssemblyBegin (*newvec);
      CHKERRQ (ierr);
      ierr = VecAssemblyEnd (*newvec);
      CHKERRQ (ierr);
      ierr = VecDestroy (&tmpvec);
      CHKERRQ (ierr);
    }
  else
    {
      ierr = PetscObjectSetName ((PetscObject) * newvec, name);
      CHKERRQ (ierr);
      ierr = VecLoad (*newvec, viewer);
      CHKERRQ (ierr);
    }
  ierr = VecGetSize (*newvec, &ntmp);
  CHKERRQ (ierr);
  return ierr;
}

/*! Read data from a PetscViewer into an IS (Index Set, i.e. integer) object.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF or PETSC_COMM_WORLD.
 * @param[in] viewer A PetscViewer instance.
 * @param[in] *varname The new IS will have this name. For reading from hdf5 files, this name must match the dataset name in the file.
 * @param[out] *newIS The new IS object.
 * @param[out] *n The number of entries in the new object.
*/
PetscErrorCode
ReadIndexSet (MPI_Comm COMM, PetscViewer viewer, char *varname, IS * newIS,
	      PetscInt * n)
{
  PetscErrorCode ierr;
  char name[256];
  sprintf (name, "%s", varname);
  ierr = ISCreate (COMM, newIS);
  CHKERRQ (ierr);
  ierr = PetscObjectSetName ((PetscObject) * newIS, name);
  CHKERRQ (ierr);
  ierr = ISLoad (*newIS, viewer);
  CHKERRQ (ierr);
  ierr = ISGetSize (*newIS, n);
  CHKERRQ (ierr);
  return ierr;
}

/*! Print to stdout MatInfo for a Mat object. Caution: hangs on multi-node. Don't use until fixed.
 * @param[in] COMM The MPI communicator PETSC_COMM_WORLD.
 * @param[in] *m The matrix.
 * @param[in] *mesg Name or some other string to prepend the output.
*/
PetscErrorCode
ShowMatInfo (MPI_Comm COMM, Mat * m, const char *mesg)
{
  PetscErrorCode ierr;
  MatInfo info;
  PetscBool isassembled;
  PetscInt rowcheck, colcheck;
  PetscMPIInt rank;

  ierr = MPI_Comm_rank (COMM, &rank);
  CHKERRQ (ierr);
  ierr = MatGetInfo (*m, MAT_GLOBAL_SUM, &info);
  CHKERRQ (ierr);
  if (rank == 0)
    {
      printf ("%s info from rank %d:\n", mesg, rank);
      ierr = MatAssembled (*m, &isassembled);
      CHKERRQ (ierr);
      printf (" is assembled: %d\n", isassembled);
      ierr = MatGetSize (*m, &rowcheck, &colcheck);
      CHKERRQ (ierr);
      printf (" global size %ld x %ld\n", rowcheck, colcheck);
      //ierr = MatGetInfo(*m,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
      printf (" block_size: %f\n", info.block_size);
      printf (" nz_allocated: %f\n", info.nz_allocated);
      printf (" nz_used: %f\n", info.nz_used);
      printf (" nz_unneeded: %f\n", info.nz_unneeded);
      printf (" memory: %f\n", info.memory);
      printf (" assemblies: %f\n", info.assemblies);
      printf (" mallocs: %f\n", info.mallocs);
      printf (" fill_ratio_given: %f\n", info.fill_ratio_given);
      printf (" fill_ratio_needed: %f\n", info.fill_ratio_needed);
    }
  ierr = MatGetInfo (*m, MAT_LOCAL, &info);
  CHKERRQ (ierr);
  if (rank == 0)
    {
      printf ("%s local info from rank %d:\n", mesg, rank);
      ierr = MatAssembled (*m, &isassembled);
      CHKERRQ (ierr);
      printf (" is assembled: %d\n", isassembled);
      ierr = MatGetSize (*m, &rowcheck, &colcheck);
      CHKERRQ (ierr);
      printf (" global size %ld x %ld\n", rowcheck, colcheck);
      //ierr = MatGetInfo(*m,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
      printf (" block_size: %f\n", info.block_size);
      printf (" nz_allocated: %f\n", info.nz_allocated);
      printf (" nz_used: %f\n", info.nz_used);
      printf (" nz_unneeded: %f\n", info.nz_unneeded);
      printf (" memory: %f\n", info.memory);
      printf (" assemblies: %f\n", info.assemblies);
      printf (" mallocs: %f\n", info.mallocs);
      printf (" fill_ratio_given: %f\n", info.fill_ratio_given);
      printf (" fill_ratio_needed: %f\n", info.fill_ratio_needed);
    }
  return ierr;
}

/*! Use metadata to determine global and local sizes and indices.
 * @param[in] nfiles The number of CSR.hdf5 files, from a previous call to CountFiles()
 * @param[in] **metadata Metadata from index.txt from a previous call to ReadIndex()
 * @param[in] local_firstfile first index in the list of files for this rank
 * @param[in] local_lastfile last index in the list of files for this rank
 * @param[out] global_nrow total number of rows from metadata
 * @param[out] global_ncol total number of columns from metadata
 * @param[out] global_nnz total number of non-zero entries from metadata
 * @param[out] local_nrow number of rows for this rank
 * @param[out] local_nnz number of non-zero entries for this rank
 * @param[out] local_row0 index of first row for this rank
*/
void
GetGlobalLocalCounts (int nfiles, PetscInt ** metadata, int local_firstfile,
		      int local_lastfile, PetscInt * global_nrow,
		      PetscInt * global_ncol, PetscInt * global_nnz,
		      PetscInt * local_nrow, PetscInt * local_nnz,
		      PetscInt * local_row0)
{
  PetscInt cmin, cmax;
  int i;

  *global_nrow = 0;
  *global_ncol = 0;
  *global_nnz = 0;
  *local_nrow = 0;
  *local_nnz = 0;
  *local_row0 = 0;
  cmin = 0;
  cmax = 0;
  for (i = 0; i < nfiles; i++)
    {
      *global_nrow += metadata[i][0];
      *global_nnz += metadata[i][3];
      if (i < local_firstfile)
	{
	  *local_row0 += metadata[i][0];
	}
      if (i == 0)
	{
	  cmin = metadata[i][1];
	  cmax = metadata[i][2];
	}
      else
	{
	  if (metadata[i][1] < cmin)
	    {
	      cmin = metadata[i][1];
	    }
	  if (metadata[i][2] > cmax)
	    {
	      cmax = metadata[i][2];
	    }
	}
    }
  *global_ncol = cmax - cmin + 1;
  for (i = local_firstfile; i <= local_lastfile; i++)
    {
      *local_nnz += metadata[i][3];
      *local_nrow += metadata[i][0];
    }
  return;
}

/*! Build local CSR block by sequentially reading in local hdf5 files.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.
 * @param[in] *csrnames[] The names of the CSR.hdf5 files
 * @param[in] local_firstfile,local_lastfile Indices of which files are handled by which rank.
 * @param[in] nsolve how many solves (right-hand sides) to perform
 * @param[out] *local_indptr CSR index ptr for this rank
 * @param[out] *local_jcol CSR column indices for this rank
 * @param[out] *local_data CSR data for this rank
 * @param[out] *local_weights Holds the concatenated weights for this rank.
 * @param[out] *local_rhs Holds the right-hand-side(s) for this rank
*/
PetscErrorCode
ReadLocalCSR (MPI_Comm COMM, char *csrnames[], int local_firstfile,
	      int local_lastfile, int nsolve, PetscInt * local_indptr,
	      PetscInt * local_jcol, PetscScalar * local_data,
	      PetscScalar * local_weights, PetscScalar ** local_rhs)
{

  PetscViewer viewer;		//viewer object for reading files
  IS indices, indptr;
  Vec data, weights, rhs;
  PetscErrorCode ierr;
  int i, j;
  char tmp[200];

  //create the distributed matrix A
  PetscInt vcnt, niptr;
  PetscInt roff = 0, roff2 = 0, zoff = 0;
  const PetscInt *iptr, *jcol;
  PetscInt poff = 0;
  PetscScalar *a, *w;
  //these will concat the multiple files per processor
  for (i = local_firstfile; i <= local_lastfile; i++)
    {
      //open the file
      ierr = PetscViewerHDF5Open (COMM, csrnames[i], FILE_MODE_READ, &viewer);
      CHKERRQ (ierr);
      //indptr
      ierr = ReadIndexSet (COMM, viewer, (char *) "indptr", &indptr, &niptr);
      CHKERRQ (ierr);
      ISGetIndices (indptr, &iptr);
      for (j = 1; j < niptr; j++)
	{
	  local_indptr[j + roff] = iptr[j] + poff;
	}
      ISRestoreIndices (indptr, &iptr);
      poff = local_indptr[niptr - 1 + roff];
      roff += niptr - 1;

      //indices
      ierr = ReadIndexSet (COMM, viewer, (char *) "indices", &indices, &vcnt);
      CHKERRQ (ierr);
      ISGetIndices (indices, &jcol);
      memcpy (&local_jcol[zoff], jcol, vcnt * sizeof (PetscInt));
      ISRestoreIndices (indices, &jcol);

      //data
      ierr = ReadVec (COMM, viewer, (char *) "data", &data, &vcnt);
      CHKERRQ (ierr);
      VecGetArray (data, &a);
      memcpy (&local_data[zoff], a, vcnt * sizeof (PetscScalar));
      VecRestoreArray (data, &a);
      zoff += vcnt;

      //weights
      ierr = ReadVec (COMM, viewer, (char *) "weights", &weights, &vcnt);
      CHKERRQ (ierr);
      VecGetArray (weights, &w);
      memcpy (&local_weights[roff2], w, vcnt * sizeof (PetscScalar));
      VecRestoreArray (weights, &w);

      //rhs
      for (j = 0; j < nsolve; j++)
	{
	  sprintf (tmp, "rhs_%d", j);
	  ierr = ReadVec (COMM, viewer, tmp, &rhs, &vcnt);
	  CHKERRQ (ierr);
	  VecGetArray (rhs, &w);
	  memcpy (&local_rhs[j][roff2], w, vcnt * sizeof (PetscScalar));
	  VecRestoreArray (rhs, &w);
	}
      roff2 += vcnt;

      ierr = PetscViewerDestroy (&viewer);
      CHKERRQ (ierr);
      ISDestroy (&indptr);
      VecDestroy (&data);
      VecDestroy (&weights);
      ISDestroy (&indices);
    }

  return ierr;
}

/*! Creates a diagonal matrix with the weights as the entries.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] *local_weights Passed into this function to get built into W.
 * @param[in] local_nrow Number of rows for this rank
 * @param[in] local_row0 The starting row for this rank.
 * @param[in] global_nrow The total length of the weights vector (N). W is NxN.
 * @param[out] *W The matrix created by this function.
*/
PetscErrorCode
CreateW (MPI_Comm COMM, PetscScalar * local_weights, PetscInt local_nrow,
	 PetscInt local_row0, PetscInt global_nrow, Mat * W)
{
  PetscErrorCode ierr;
  Vec global_weights;
  PetscInt *indx;
  int i;

  ierr = VecCreate (COMM, &global_weights);
  CHKERRQ (ierr);
  ierr = VecSetSizes (global_weights, local_nrow, global_nrow);
  CHKERRQ (ierr);
  ierr = VecSetType (global_weights, VECMPI);
  CHKERRQ (ierr);
  indx = (PetscInt *) malloc (local_nrow * sizeof (PetscInt));
  for (i = 0; i < local_nrow; i++)
    {
      indx[i] = local_row0 + i;
    }
  ierr =
    VecSetValues (global_weights, local_nrow, indx, local_weights,
		  INSERT_VALUES);
  CHKERRQ (ierr);
  free (indx);

  ierr = MatCreate (PETSC_COMM_WORLD, W);
  CHKERRQ (ierr);
  ierr = MatSetSizes (*W, local_nrow, local_nrow, global_nrow, global_nrow);
  CHKERRQ (ierr);
  ierr = MatSetType (*W, MATMPIAIJ);
  CHKERRQ (ierr);
  ierr = MatMPIAIJSetPreallocation (*W, 1, NULL, 0, NULL);
  CHKERRQ (ierr);
  ierr = MatDiagonalSet (*W, global_weights, INSERT_VALUES);
  CHKERRQ (ierr);

  ierr = VecDestroy (&global_weights);
  CHKERRQ (ierr);
  return ierr;
}

/*! Creates a diagonal matrix with the weights as the entries.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] local_nrow Number of rows for this rank
 * @param[in] local_row0 The starting row for this rank.
 * @param[in] global_nrow The total length of the weights vector (N). L is NxN.
 * @param[out] *L The matrix created by this function.
*/
PetscErrorCode
CreateL (MPI_Comm COMM, char indexname[], PetscInt local_nrow,
	 PetscInt global_nrow, PetscBool trunc, Mat * L)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec global_reg;
  PetscMPIInt rank;
  PetscInt junk;

  ierr = MPI_Comm_rank (COMM, &rank);
  CHKERRQ (ierr);

  ierr = PetscViewerHDF5Open (COMM, indexname, FILE_MODE_READ, &viewer);
  CHKERRQ (ierr);
  ierr =
    ReadVecWithSizes (COMM, viewer, (char *) "reg", &global_reg, &junk,
		      local_nrow, global_nrow, trunc);
  CHKERRQ (ierr);

  ierr = MatCreate (PETSC_COMM_WORLD, L);
  CHKERRQ (ierr);
  ierr = MatSetSizes (*L, local_nrow, local_nrow, global_nrow, global_nrow);
  CHKERRQ (ierr);
  ierr = MatSetType (*L, MATMPIAIJ);
  CHKERRQ (ierr);
  ierr = MatMPIAIJSetPreallocation (*L, 1, NULL, 0, NULL);
  CHKERRQ (ierr);
  ierr = MatDiagonalSet (*L, global_reg, INSERT_VALUES);
  CHKERRQ (ierr);
  MatAssemblyBegin (*L, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd (*L, MAT_FINAL_ASSEMBLY);
  ierr = PetscViewerDestroy (&viewer);
  CHKERRQ (ierr);
  VecDestroy (&global_reg);
  return ierr;
}

/*! Counts how many x0 vectors are stored in the regularization file. 
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] dir string containing the directory
 * @param[out] number of solves
*/
PetscErrorCode
CountSolves (MPI_Comm COMM, char indexname[], PetscInt * nsolve)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscInt junk;
  IS test;

  *nsolve = 0;
  ierr = PetscViewerHDF5Open (COMM, indexname, FILE_MODE_READ, &viewer);
  CHKERRQ (ierr);
  ierr = ReadIndexSet (COMM, viewer, (char *) "solve_list", &test, &junk);
  CHKERRQ (ierr);
  *nsolve = junk;
  ierr = PetscViewerDestroy (&viewer);
  CHKERRQ (ierr);
  return ierr;
}

/*! Read the x0 vectors stored in the regularization file.
 * @param[in] COMM The MPI communicator, PETSC_COMM_SELF.A
 * @param[in] dir string containing the directory
 * @param[in] local_nrow Number of local rows this rank will own
 * @param[in] global_nrow Number of global rows.
 * @param[in] nsolve Number of x0 vectors to be read.
 * @param[out] x0[] vectors
*/
PetscErrorCode
Readx0 (MPI_Comm COMM, char indexname[], PetscInt local_nrow,
	PetscInt global_nrow, PetscInt nsolve, PetscBool trunc, Vec x0[])
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscInt junk;
  char tmp[200];
  int i;

  ierr = PetscViewerHDF5Open (COMM, indexname, FILE_MODE_READ, &viewer);
  CHKERRQ (ierr);
  for (i = 0; i < nsolve; i++)
    {
      sprintf (tmp, "x_%d", i);
      ierr =
	ReadVecWithSizes (COMM, viewer, tmp, &x0[i], &junk, local_nrow,
			  global_nrow, trunc);
      CHKERRQ (ierr);
    }
  ierr = PetscViewerDestroy (&viewer);
  CHKERRQ (ierr);
  return ierr;
}

