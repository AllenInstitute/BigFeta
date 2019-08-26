#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/sysinfo.h>

typedef struct
{
  char *name;
  double mem;
  int *ranks;
  int nrank;
  double tmem;
  int nfiles;
  int *files;
} node;

void
freenode (node inode)
{
  free (inode.name);
  free (inode.ranks);
  free (inode.files);
}

void
disp_node (node inode)
{
  int i;
  printf ("name %s mem %f nrank %d ranks", inode.name, inode.mem,
	  inode.nrank);
  for (i = 0; i < inode.nrank; i++)
    {
      printf (" %d", inode.ranks[i]);
    }
  printf (" tmem %f nfile %d nfiles", inode.tmem, inode.nfiles);
  for (i = 0; i < inode.nrank; i++)
  {
      printf(" %d", inode.files[i]);
  }
  printf ("\n");
}

char *
unique_str (char *in, int n, int m, int *nu)
{
  int i, j, c, *unique;
  char *out;

  unique = (int *) malloc (n * sizeof (int));

  // just count first
  for (i = 0; i < n; i++)
    {
      unique[i] = 1;
      for (j = 0; j < i; j++)
	{
	  c = strcmp (&in[i * m], &in[j * m]);
	  if (c == 0)
	    {
	      unique[i] = 0;
	    }
	}
    }

  *nu = 0;
  for (i = 0; i < n; i++)
    {
      if (unique[i] == 1)
	{
	  *nu = *nu + 1;
	}
    }

  // allocate and fill
  out = (char *) malloc (m * (*nu) * sizeof (char));
  j = 0;
  for (i = 0; i < n; i++)
    {
      if (unique[i] == 1)
	{
	  sprintf (&out[j * m], &in[i * m]);
	  j++;
	}
    }

  free (unique);
  return out;
}

node *
index_rank (char *names, double *mem, int n, char *unames, int un, int mx)
{
  int i, j, k;
  node *out;

  out = (node *) malloc (un * sizeof (node));
  for (j = 0; j < un; j++)
    {
      out[j].name = (char *) malloc (mx * sizeof (char));
      sprintf (out[j].name, &unames[j * mx]);

      // count the nodes
      k = 0;
      for (i = 0; i < n; i++)
	{
	  if (strcmp (&names[i * mx], &unames[j * mx]) == 0)
	    {
	      k++;
	    }
	}
      out[j].nrank = k;
      out[j].ranks = (int *) malloc (k * sizeof (int));
      k = 0;


      for (i = 0; i < n; i++)
	{
	  if (strcmp (&names[i * mx], &unames[j * mx]) == 0)
	    {
	      out[j].ranks[k] = i;
	      out[j].mem = mem[i];
	      k++;
	    }
	}
    }
  return out;
}

void
split_files (node * nodes, int nnodes, int nfiles)
{
  int i, j, *f, t, m, k, ncpus;

  f = (int *) malloc (nnodes * sizeof (int));

  t = 0;
  for (i = 0; i < nnodes; i++)
    {
      f[i] = floor (nfiles * nodes[i].mem / nodes[i].tmem);
      t += f[i];
    }

  m = 0;
  while (t < nfiles)
    {
      k = m % nnodes;
      f[k]++;
      t++;
      m++;
    }

  for (i = 0; i < nnodes; i++)
    {
      nodes[i].nfiles = f[i];
      nodes[i].files = (int *) malloc (nodes[i].nrank * sizeof(int));
      t = 0;
      for (j =0; j< nodes[i].nrank; j++){
	      nodes[i].files[j] = floor (f[i] / nodes[i].nrank);
	      t += nodes[i].files[j];
      }
      m = 0;
      while (t < f[i]){
          k = m % nodes[i].nrank;
	  nodes[i].files[k]++;
	  t++;
	  m++;
      }
    }
  
  free (f);
}


node *
hw_config (MPI_Comm COMM, int *nnodes, int *thisnode)
{
  int i, j, k, rank, nrank, len, nu, *inodes;
  char name[MPI_MAX_PROCESSOR_NAME];
  struct sysinfo si;
  const double gigabyte = 1024 * 1024 * 1024;
  char *names, *unames;
  double tmem, rank_mem, *mem;
  node *nodes;

  sysinfo (&si);

  MPI_Comm_size (COMM, &nrank);
  MPI_Comm_rank (COMM, &rank);
  MPI_Get_processor_name (name, &len);
  rank_mem = si.totalram / gigabyte;

  names = (char *) malloc (MPI_MAX_PROCESSOR_NAME * nrank * sizeof (char));
  mem = (double *) malloc (nrank * sizeof (double));

  // gather all the stats to rank 0
  MPI_Gather (&rank_mem, 1, MPI_DOUBLE, mem, 1, MPI_DOUBLE, 0, COMM);
  MPI_Gather (&name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names,
	      MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, COMM);
  // tell everyone
  MPI_Bcast (names, nrank * MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, COMM);
  MPI_Bcast (mem, nrank, MPI_DOUBLE, 0, COMM);

  unames = unique_str (names, nrank, MPI_MAX_PROCESSOR_NAME, &nu);
  nodes = index_rank (names, mem, nrank, unames, nu, MPI_MAX_PROCESSOR_NAME);
  tmem = 0;
  for (i = 0; i < nu; i++)
    {
      tmem += nodes[i].mem;
    }
  for (i = 0; i < nu; i++)
    {
      nodes[i].tmem = tmem;
    }
  inodes = (int *) malloc (nrank * sizeof (int));
  for (i = 0; i < nrank; i++)
    {
      for (j = 0; j < nu; j++)
	{
	  for (k = 0; k < nodes[j].nrank; k++)
	    {
	      if (nodes[j].ranks[k] == i)
		{
		  inodes[i] = j;
		}
	    }
	}
    }

  *nnodes = nu;
  *thisnode = inodes[rank];

  free (names);
  free (mem);
  free (unames);
  free (inodes);

  return nodes;
}

//int
//main (int argc, char *argv[])
//{
//  int i, rank, nnodes, thisnode, noderank;
//  MPI_Comm MPI_COMM_NODE;
//
//  node *nodes;
//  MPI_Init (&argc, &argv);
//  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
//
//  nodes = hw_config (MPI_COMM_WORLD, &nnodes, &thisnode);
//  split_files (nodes, nnodes, 373);
//
//  if (rank == 0)
//    {
//      printf ("main %d nnodes\n", nnodes);
//      for (i = 0; i < nnodes; i++)
//	{
//	  disp_node (nodes[i]);
//	  freenode (nodes[i]);
//	}
//      free (nodes);
//    }
//
//  MPI_Comm_split (MPI_COMM_WORLD, thisnode, rank, &MPI_COMM_NODE);
//  MPI_Comm_rank (MPI_COMM_NODE, &noderank);
//
//  MPI_Finalize ();
//  return 0;
//}
