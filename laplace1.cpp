#include "mpi.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
typedef std::vector<double> Matrix; // alias
void initial_conditions(Matrix &m, int nrows, int ncols, int pid, int np);
void boundary_conditions(Matrix &m, int nrows, int ncols, int pid, int np);
void print_screen(const Matrix &m, int nrows, int ncols, int pid, int np);
void print_slab(const Matrix &m, int nrows, int ncols);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int pid, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  const int N = std::atoi(argv[1]);
  const double L = std::atof(argv[2]);
  const int STEPS = std::atoi(argv[3]);
  const double DELTA = L / N;

  // problem partition
  int NCOLS = N, NROWS = N / np + 2; // include ghosts
  Matrix data(NROWS * NCOLS);        // include ghosts cells
  initial_conditions(data, NROWS, NCOLS, pid, np);
  if (0 == pid) {
    std::cout << " After initial conditions ...\n";
  }
  print_screen(data, NROWS, NCOLS, pid, np);
  boundary_conditions(data, NROWS, NCOLS, pid, np);
  if (0 == pid) {
    std::cout << " After boundary conditions ...\n";
  }
  print_screen(data, NROWS, NCOLS, pid, np);
  MPI_Finalize();
  return 0;
}

void initial_conditions(Matrix &m, int nrows, int ncols, int pid, int np) {
  // same task for all pids, but fill with the pids to distinguish among thems
  for (int ii = 0; ii < nrows; ++ii) {
    for (int jj = 0; jj < ncols; ++jj) {
      m[ii * ncols + jj] = pid;
    }
  }
}

void boundary_conditions(Matrix &m, int nrows, int ncols, int pid, int np) {
  int u = np - 1;
  int ii = 0, jj = 0;
  if (pid == 0) {
    ii = 1;
    for (jj = 0; jj < ncols; ++jj)
      m[ii * ncols + jj] = 100;
    jj = 0;
    for (ii = 1; ii < nrows - 1; ++ii)
      m[ii * ncols + jj] = 0;

    jj = ncols - 1;
    for (ii = 1; ii < nrows - 1; ++ii)
      m[ii * ncols + jj] = 0;

  } else {
    if (pid == u) {
      ii = nrows - 2;
      for (jj = 0; jj < ncols; ++jj)
        m[ii * ncols + jj] = 0;

      jj = 0;
      for (ii = 1; ii < nrows - 1; ++ii)
        m[ii * ncols + jj] = 0;

      jj = ncols - 1;
      for (ii = 1; ii < nrows - 1; ++ii)
        m[ii * ncols + jj] = 0;
    } else {
      jj = 0;
      for (ii = 1; ii < nrows - 1; ++ii)
        m[ii * ncols + jj] = 0;

      jj = ncols - 1;
      for (ii = 1; ii < nrows - 1; ++ii)
        m[ii * ncols + jj] = 0;
    }
  }
}

void print_screen(const Matrix &m, int nrows, int ncols, int pid, int np) {
  MPI_Status status;
  // Master pid prints
  if (0 == pid) {
    // print master data
    print_slab(m, nrows, ncols);
    // now receive in buffer and print other pids data
    Matrix buffer(nrows * ncols);
    for (int k = 1; k < np; k++) {
      MPI_Recv(&buffer[0], nrows * ncols, MPI_DOUBLE, k, 0, MPI_COMM_WORLD,
               &status);
      print_slab(buffer, nrows, ncols);
    }
  } else { // workers send
    MPI_Send(&m[0], nrows * ncols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void print_slab(const Matrix &m, int nrows, int ncols) {
  // ignore ghosts
  for (int ii = 1; ii < nrows - 1; ++ii) {
    for (int jj = 0; jj < ncols; ++jj) {
      std::cout << std::setw(3) << m[ii * ncols + jj] << " ";
    }
    std::cout << "\n";
  }
}
