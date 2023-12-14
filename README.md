# Parallelization of Singular Value Decomposition
Meivenkatkumar Lakshminarayanan, Nayanthara K. Jayadev

## Singular Value Decomposition
+ Powerful matrix factorization technique that decomposes a matrix into three other matrices, revealing important structural aspects of the original matrix. In SVD, a matrix A (M X N) is decomposed into three matrices namely U (M X M), Σ (M X N), and V (N X N).
+ A = UΣV^T
+ SVD has many applications, including signal processing, image compression, dimensionality reduction in machine learning, and **quantum chemistry**.
<p align="center">
   <img width="468" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/24ba112d-a853-49d0-8af7-3be674a1e620">
</p>
+ SVD finds its application in the above-mentioned domains for the following reasons, including,

  + The need for low-rank (or reduced) representation of a matrix.
    
  + The need for a set of (orthogonal) bases for the row and column spaces of a matrix.
    
  + The need for information about the rank of a matrix.

## Fluid Dynamics Applications
+ In the viscous Burgers equation with initial conditions are given as,
<p align="center">
   <img width="161" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/cbe497f7-3b09-420a-bf0f-522ccba5ef35">
</p>
<p align="center">
  <img width="299" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/b3e198af-5b38-4462-82ba-e435decaf76e">
</p>

+ The boundary conditions are given by u(0,t)=0, and u(L,t)=0.

+ The analytic solution is given by,
<p align="center">
  <img width="370" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/39519a50-cc87-4a6a-886f-8fa03e62b042">

</p>,
where Re is the Reynolds number kept fixed at 1000, and t_0 is exp(Re/8).

+ The analytic solution is utilized to create snapshots for the construction of the data matrix.
+ The SVD of the data matrix becomes difficult for larger matrices since the number of snapshots is  and a grid-resolution of the domain x in [0, L] for the assessment is fairly larger (around 800 snapshots and 1600 grid points).

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

## Aim
+ To test the parallelization of the SVD of the data matrix relevant to viscous Burgers equation.
    + To scrutinize the effect of parallelization with respect to the increasing size of the matrices.
    + To measure the speed up compared to a serial run.
+ Parallelization of singular value decomposition of third rank tensors relevant to Auger decay.
  + Parallelize the SVD code using MPI, OpenMP, and CUDA, and compare it with the existing algorithms for two-body Dyson tensors.

## Preliminary investigation: Existing Algorithms
+ Using PyParSVD: Python Parallel Singular Value Decomposition.
  + PyParSVD is based on algorithms using distribution or partitioned SVD as shown by Wang et al.<sup>2</sup>.
  + The process can be distributed into much smaller tasks over multiple processors in parallel, drastically reducing the computational time.
  + In this paper, a novel partitioned method for generating the SVD basis from given data was introduced.
  + This method preserves the distributed nature of the data and takes advantage of parallelism for computation.
  + Additionally, it greatly reduces subtask communication volume.
    
    https://github.com/Romit-Maulik/PyParSVD#Wang-et-al-2016
  + The key components of the PyParSVD algorithm are streaming, distribution, and the randomization of the matrix relevant to SVD.
    + Streaming: The streaming in SVD is done to scrutinize the presence of coherent structures in the data. It functions by extracting solely the first K left singular vectors, which correspond to the
K largest coherent structures. Streaming aids in reducing the cost of the SVD to O(MNK) operations and
the memory footprint to O(MK, compared to the classical SVD which scales as O(MN^2
) and requires O(MN) memory. The streaming aspect of this method involves updating the left singular eigenvectors in a manner similar to processing batches.
    + Distribution: The second foundational element involves the distribution of computations, achieved through the use of the approximate partitioned method of snapshots (APMOS). This method enables the computation of distributed left singular vectors. It's important to note that APMOS, as implemented here, differs from the standard version by not supporting a batch-wise update of singular vectors. Instead, each batch entails its own basis vector calculation, which can be stored on disk. Although this algorithm lacks the capability to build a set of bases for the entire simulation duration, its distributed nature allows for the construction of a global basis, even when dealing with domain decomposition. APMOS depends on computing the left singular vectors locally for the data matrix on each rank within the simulation. To form this data matrix, snapshots of the local data may be gathered across multiple time steps.
    + Randomization: The last foundational component involves randomized linear algebra. In this context, we initially specify the complete singular value decomposition (SVD) of matrix A as UΣV^T. However, the computation of this SVD is expensive, leading to seek a low-rank factorization of A. To achieve this,  an approximate basis for the range of the matrix A is calculated.
     <p align="center">
      <img width="154" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/b26946ea-ad42-4f3b-a59e-29a85b9cbba3">

   </p>
       where Q is generally randomly sampled from a zero-mean unit-variance Gaussian distribution every time a randomized SVD is required
      

+ Parallel SVD using Jacobi’s rotations, implemented in OpenMP.
   + The Jacobi method consists of a sequence of orthogonal similarity transformations.
   + Each transformation (a Jacobi rotation) is just a plane rotation designed to annihilate one of the off-diagonal matrix elements.
   + Successive transformations undo previously set zeros, but the off-diagonal elements nevertheless get smaller and smaller, until the matrix is diagonal to machine precision.
   + Accumulating the product of the transformations as you go gives the matrix of eigenvectors, while the elements of the final diagonal matrix are the eigenvalues.
     
     https://github.com/lixueclaire/Parallel-SVD/blob/master/OMP_SVD.cpp

+ A parallelized implementation of Principal Component Analysis (PCA) using Singular Value Decomposition (SVD) in OpenMP for C.
  + The procedure used is Modified Gram Schmidt algorithm.
    
     https://github.com/arneish/parallel-PCA-openmp

## Results
Here, we compare the SVD timings for the u(x,t) matrices relevant to the viscous Burgers equation by varying dimensions using the PyParSVD algorithm. We test the efficiency of the algorithm by comparing the timings from parallel and serial run. The Table and the plot show how the parallelization reduces the time required to perform SVD. The advantage and use of parallelization of SVD is more pronounced for matrices of larger dimension. It is worth mentioning that the for matrix of dimension 524288 x 800, the parallelization reduces the computational time by half. Consequently, we also observe that the case when the number of threads is 4 takes more time than when threads is 2. Increasing the number of threads to four implies more messages exchanged between threads compared to using only two threads. Given that we are partitioning the matrix into only four equal parts, each message between threads corresponds to one-fourth of the matrix size. The larger the dataset, the longer it takes to transmit messages between threads. This issue may be mitigated by dividing the matrix into smaller matrices with a higher number of rows, but this would make the algorithm more complex.
   
    
   <p align="center">
    <img width="949" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/a99b978b-9420-4944-a0cc-6ece8d2ba210">
   </p>
   
   <p align="center">
     <img width="633" alt="image" src="https://github.com/NayantharaJayadev/csci596/assets/53525004/c27f048e-9e01-422f-acc9-4e137879cc28">
   </p>

## Conclusion

We investigate the effect of parallelization of the singular value decomposition on matrices relevant to fluid dynamics, especially in the domains of the viscous Burgers equation. We observe a speed up especially for larger matrices in the case of parallel run in comparison with serial case. We also scrutinize the effect of parallelization by varying the number of threads.

## Future directions

The given algorithm is based on MPI parallization. We are in the process of developing an Open-MP variant of the same. We also work towards modifying the algorithm for two-body Dyson matrices relevant to Auger decay.

 ## Refererences
  1) J. Phys. Chem. Lett. 2023, 14, 38, 8612–8619
  2) R. Maulik and G. Mengaldo, "PyParSVD: A streaming, distributed and randomized singular-value-decomposition library," 2021 7th International Workshop on Data Analysis and Reduction for Big Scientific Data (DRBSD-7), St. Louis, MO, USA, 2021, pp. 19-25, doi: 10.1109/DRBSD754563.2021.00007.
     

    
   









  














 










  





















