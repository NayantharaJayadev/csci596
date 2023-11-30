# Parallelization of Singular Value Decomposition
Meivenkatkumar Lakshminarayanan, Nayanthara K. Jayadev

## Singular Value Decomposition
+ Powerful matrix factorization technique that decomposes a matrix into three other matrices, revealing important structural aspects of the original matrix.
+ A wide range of applications, including signal processing, image compression, dimensionality reduction in machine learning, and **quantum chemistry**.
+ Forms the foundational basis for Principal Component Analysis (PCA).
   + And hence involved in various machine learning algorithms' preprocessing step.
+ SVD is beneficial in these applications, because of the following reasons.
   + The need for low-rank (or reduced) representation of a matrix.
   + The need for a set of (orthogonal) bases for the row and column spaces of a matrix.
   + The need for information about the rank of a matrix
<p align="center">
   <img width="468" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/24ba112d-a853-49d0-8af7-3be674a1e620">
</p>

## Quantum Chemistry Applications
 
+ Wave functions provide us with the probability of finding an electron at a certain position in space.
+ In one-electron systems, orbitals are wavefunctions.
+ The wave functions of many-electron systems depend on the coordinates of all electrons.
+ The concept of orbital gets murkier.
+ Introduction of orbitals useful for different processes (Dyson orbitals, Natural transition orbitals)
  
## Auger decay
+ One valence electron fills the core hole, and another gets ejected into the continuum.
<p align="center">
  <img width="396" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/012562ef-2989-48d7-bf78-87da9076d7bf">
</p>

+ The decay width can be calculated as 
<p align="center">
  <img width="496" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/6bad9825-10cd-425c-a188-a94fb557d4db">
</p>

+ Two-body Dyson amplitudes contain information about Auger decay from the bound domain.
<p align="center">
  <img width="200" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/4890e81f-2f2d-4695-b116-e6d9826aae2b">
</p>

+ **The most compact representation of such third-rank tensors can be obtained using a two-step singular value decomposition giving natural Auger orbitals**
<p align="center">
  <img width="468" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/cd6b61ab-acbe-4192-b1c4-c2240ad52b6e">
</p>

+ However, the SVD of bigger matrices is very slow compared to other steps in the algorithm proposed by Krylov and coworkers<sup>1</sup>.
+ The classical SVD of a M x N matrix computation scales as O(MN<sup>2</sup>) and requires O(MN) memory.

## Aim
+ Parallelization of singular value decomposition of third rank tensors relevant to Auger decay.
  + Comparison between different available algorithms and measure the speed up compared to a serial run.
  + Parallelize the SVD code using MPI, and OpenMP and compare it with the existing algorithms for two-body Dyson tensors.

## Preliminary investigation: Existing Algorithms
+ Using PyParSVD: Python Parallel Singular Value Decomposition.
  + A Python library that implements a streaming, distributed and randomized algorithm for the singular value decomposition.
  + PyParSVD is based on algorithms using distribution or partitioned SVD as shown by Wang et al.<sup>2</sup>.
  + The process can be distributed into much smaller tasks over multiple processors in parallel, drastically reducing the computational time.
  + In this paper, a novel partitioned method for generating the SVD basis from given data was introduced.
  + This method preserves the distributed nature of the data and takes advantage of parallelism for computation.
  + Additionally, it greatly reduces subtask communication volume.
    
    https://github.com/Romit-Maulik/PyParSVD#Wang-et-al-2016

+ Parallel SVD using Jacobi’s rotations, implemented in OpenMP.
   + The Jacobi method consists of a sequence of orthogonal similarity transformations.
   + Each transformation (a Jacobi rotation) is just a plane rotation designed to annihilate one of the off-diagonal matrix elements.
   + Successive transformations undo previously set zeros, but the off-diagonal elements nevertheless get smaller and smaller, until the matrix is diagonal to machine precision.
   + Accumulating the product of the transformations as you go gives the matrix of eigenvectors, while the elements of the final diagonal matrix are the eigenvalues.
     
     https://github.com/lixueclaire/Parallel-SVD/blob/master/OMP_SVD.cpp

+ A parallelized implementation of Principal Component Analysis (PCA) using Singular Value Decomposition (SVD) in OpenMP for C.
  + The procedure used is Modified Gram Schmidt algorithm.
    
     https://github.com/arneish/parallel-PCA-openmp

  ## Refererences
  1) J. Phys. Chem. Lett. 2023, 14, 38, 8612–8619
  2) R. Maulik and G. Mengaldo, "PyParSVD: A streaming, distributed and randomized singular-value-decomposition library," 2021 7th International Workshop on Data Analysis and Reduction for Big Scientific Data (DRBSD-7), St. Louis, MO, USA, 2021, pp. 19-25, doi: 10.1109/DRBSD754563.2021.00007.
     










  





















