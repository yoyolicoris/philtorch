//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "helpers.h"
#include "rnn_cell_impl.h"

template< const int chunk_size,
          typename idx_t >
__forceinline__ __device__ int countMyEqs(
    idx_t seqLength,
    idx_t hiddenDim,
    idx_t batchSize,
    int shiftBlock = 0,
    int shiftIdx = 0
){
    const bool isInside = ( ( threadIdx.x + (blockIdx.x+shiftBlock) * blockDim.x ) * chunk_size + shiftIdx < seqLength )
                      &&( ( threadIdx.y + blockIdx.y * blockDim.y ) < hiddenDim )
                      &&( ( threadIdx.z + blockIdx.z * blockDim.z ) < batchSize);
    if( isInside ){
        return min( chunk_size, seqLength - shiftIdx - ( threadIdx.x + (blockIdx.x+shiftBlock) * blockDim.x ) * chunk_size );
    }

    return 0;
}


template< const int chunk_size,
          typename idx_t >
__forceinline__ __device__ idx_t getMyDataIdx(
    idx_t seqLength,
    idx_t hiddenDim,
    int shiftBlock = 0
){
    return ( ( threadIdx.x + (blockIdx.x+shiftBlock) * blockDim.x ) * chunk_size
           + ( threadIdx.y + blockIdx.y * blockDim.y ) * seqLength
           + ( threadIdx.z + blockIdx.z * blockDim.z ) * seqLength * hiddenDim
           );
}


template< const int chunk_size,
          const int threads_per_warp,
          typename idx_t >
__forceinline__ __device__ int countMyWarpEqs(
    idx_t seqLength,
    idx_t hiddenDim,
    idx_t batchSize,
    int shiftBlock = 0
){
    const int wIdx = threadIdx.x / threads_per_warp;      // warp index
    return max( 0, min( chunk_size * threads_per_warp, seqLength - ( wIdx * threads_per_warp + (blockIdx.x+shiftBlock) * blockDim.x ) * chunk_size ) )
                                                                            * (( threadIdx.y + blockIdx.y * blockDim.y ) < hiddenDim
                                                                            && ( threadIdx.z + blockIdx.z * blockDim.z ) < batchSize);
}


template< const int chunk_size,
          const int threads_per_warp,
          typename idx_t >
__forceinline__ __device__ idx_t getWarpDataIdx(
    idx_t seqLength,
    idx_t hiddenDim,
    int shiftBlock = 0
){
    const int wIdx = threadIdx.x / threads_per_warp;      // warp index
    return ( ( wIdx * threads_per_warp + (blockIdx.x+shiftBlock) * blockDim.x ) * chunk_size
           + ( threadIdx.y + blockIdx.y * blockDim.y ) * seqLength
           + ( threadIdx.z + blockIdx.z * blockDim.z ) * seqLength * hiddenDim
           );
}



// Thomas reduction ****************************************************************************************************
template< typename pimpl_t >
__forceinline__ __device__ void thomasReduction(
    typename pimpl_t::jac_t (&jac)[pimpl_t::chunk_size],
    typename pimpl_t::rhs_t (&rhs)[pimpl_t::chunk_size]
){
    /*
        Applies Thomas algorithm (ie, sequential forward substitution) to a matrix (chunk) in the form:
        j[0]| 1              |   |r[0]|      j[0]    | 1             |   |r[0]         |
            |j[1] 1          |   |r[1]|     -j[0]j[1]|     1         |   |r[1]-j[1]r[0]|
            |    j[2]  1     | = |r[2]| -->          |   j[2]  1     | = |r[3]         | --> ...
            |         j[3]   |   |r[3]|              |        j[3] 1 |   |r[4]         |
            |             ...|   | ...|              |           ... |   | ...         |
    */
    static constexpr int chunk_size = pimpl_t::chunk_size;

    for( int t = 1; t < chunk_size; ++t ){
        pimpl_t::reduceEqs( jac[t-1], jac[t], rhs[t-1], rhs[t] );
    }
}




// Parallel reduction within warp **************************************************************************************
template< typename pimpl_t >
__forceinline__ __device__ void parallelReductionWarp(
    typename pimpl_t::jac_t (&jac),
    typename pimpl_t::rhs_t (&rhs),
    int lIdx
){
    /*
        Applies PCR algorithm (ie, a parallel version of forward substitution) to a matrix in the form:
        j[0]| 1              |   |r[0]|           j[0]|    1                        |   |r[0]         |
            |j[1] 1          |   |r[1]|      -j[0]j[1]|    0       1                |   |r[1]-j[1]r[0]|
            |    j[2]  1     | = |r[2]| -->           |-j[1]j[2]   0       1        | = |r[2]-j[2]r[1]|  --> ...
            |         j[3]   |   |r[3]|               |        -j[2]j[3]   0     1  |   |r[3]-j[3]r[2]|
            |             ...|   | ...|               |                         ... |   | ...         |
        At each step of the alg, eqs are combined and reduced to split a system of N eqs into two independent systems of
         N/2 eqs each. The alg terminates after log2(N) steps, when we're left with N systems of 1 eq each.

        NB: this is a warp-wide-implementation! Meaning that it should *only* be called on a system whose coefficients
            are sitting in a warp! This allows to use warp-specific directives to read their registers, as well as to
            ignore syncing directives (as warp-wide instructions are automatically synced)
    */

    static constexpr int threads_per_warp = pimpl_t::threads_per_warp;

    for( int po2 = 1; po2 < threads_per_warp; po2 <<= 1) {
        pimpl_t::reduceEqsWarp( jac, rhs, po2, lIdx );
        // No need for syncing: warps are automatically synced
    }
}



// Parallel reduction within block *************************************************************************************
// Assumptions:
// - Time is the leading dimension (t=x), of size T
// - each thread along the t (=x) dimension handles chunk_size data
// - grid dimension y is the inner state dimension (so one thread per y)
// - grid dimension z is the batch size (so one thread per z)
template< typename pimpl_t >
__device__ void parallelReductionSharedInner(
    typename pimpl_t::jac_t (&myRegJac)[pimpl_t::chunk_size],
    typename pimpl_t::rhs_t (&myRegRhs)[pimpl_t::chunk_size],
    typename pimpl_t::jac_t *sharedJac,
    typename pimpl_t::rhs_t *sharedRhs,
    int numMyEqs,
    typename pimpl_t::idx_t seqLength
){
    /*
        This function applies parallel reduction to a bi-diagonal matrix, in the form
         0=j[0]| 1               |   |r[0]|
               |j[1] 1           |   |r[1]|
               |    j[2]  1      | = |r[2]|
               |         j[3]  1 |   |r[3]|
               |              ...|   | ...|
        To do so, it proceeds hierarchically, in steps.

        Step 1) Chunk-wise Thomas reduction
                The matrix is split into chunks (of size chunk_size), where classical sequential forward substitution is
                 performed to reduce each eq in the chunk in terms of the last variable of the previous chunk.
                Each thread is responsible for performing reduction on its chunk.
                After this operation, the matrix structure is reduced as such (here, chunk_size=3):
                |j[0]| 1     |             |  -
                |j[1]|   1   |             |  | Chunk0
                |j[2]|     1 |             |  -
                |        j[3]| 1     |     |  -
                |        j[4]|   1   |     |  | Chunk1
                |        j[5]|     1 |     |  -
                |                      ... |  ...

        Step 2) Warp-wise parallel reduction
                Threads belonging to the same warp perform parallel reduction onto the system composed of the last
                variables of their chunks. This is the lowest level of the parallel reduction hierarchy, and allows to
                use synced warp-wise directives to speed-up the operations.
                After this operation, the matrix structure is reduced as such (here, chunk_size=3, threads_per_warp=3):
                |j[0]| 1     |                                         |  -
                |j[1]|   1   |                                         |  |
                |j[2]|     1 |                                         |  |
                |        j[3]| 1     |                                 |  |
                |        j[4]|   1   |                                 |  | Warp 0
                |j[5]        |     1 |                                 |  |
                |                j[6]| 1     |                         |  |
                |                j[7]|   1   |                         |  |
                |j[8]                |     1 |                         |  -
                |                        j[9]| 1     |                 |  -
                |                       j[10]|   1   |                 |  |
                |                       j[11]|     1 |                 |  |
                |                               j[12]| 1     |         |  |
                |                               j[13]|   1   |         |  | Warp 1
                |                       j[14]        |     1 |         |  |
                |                                       j[15]| 1     | |  |
                |                                       j[16]|   1   | |  |
                |                       j[17]                |     1 | |  -
                |                                                  ... |  ...

        Step 3) Block-wise parallel reduction
                The last thread of each warp continues performing parallel reduction onto the system composed of their
                 last variables (ie, the last variable of the last chunk of each warp). This is the block-level of the
                 parallel reduction hierarchy: as it involves threads in a same block, shared memory can be used, but
                 explicit syncing is required.
                After this operation, the matrix structure is reduced as such:
                |j[0]| 1     |                                         |  -
                |j[1]|   1   |                                         |  |
                |j[2]|     1 |                                         |  |
                |        j[3]| 1     |                                 |  |
                |        j[4]|   1   |                                 |  | Warp 0
                |j[5]        |     1 |                                 |  |
                |                j[6]| 1     |                         |  |
                |                j[7]|   1   |                         |  |
                |j[8]                |     1 |                         |  -
                |                        j[9]| 1     |                 |  -
                |                       j[10]|   1   |                 |  |
                |                       j[11]|     1 |                 |  |
                |                               j[12]| 1     |         |  |
                |                               j[13]|   1   |         |  | Warp 1
                |                       j[14]        |     1 |         |  |
                |                                       j[15]| 1     | |  |
                |                                       j[16]|   1   | |  |
                |j[17]                                       |     1 | |  -
                |                                                  ... |  ...
                Assuming the whole matrix can sit in a block, at this stage the last variable of the last chunk of each
                 warp have been solved for.

        Step 4) Warp-wise forward substitution
                The last thread of each warp broadcasts its solution to the threads in the next warp, so that they can
                 solve for the last variable in each chunk
                After this operation, the matrix structure is reduced as such:
                |j[0]| 1     |                                         |  -
                |j[1]|   1   |                                         |  |
                |j[2]|     1 |                                         |  |
                |        j[3]| 1     |                                 |  |
                |        j[4]|   1   |                                 |  | Warp 0
                |j[5]        |     1 |                                 |  |
                |                j[6]| 1     |                         |  |
                |                j[7]|   1   |                         |  |
                |j[8]                |     1 |                         |  -
                |                        j[9]| 1     |                 |  -
                |                       j[10]|   1   |                 |  |
                |j[11]                       |     1 |                 |  |
                |                               j[12]| 1     |         |  |
                |                               j[13]|   1   |         |  | Warp 1
                |j[14]                               |     1 |         |  |
                |                                       j[15]| 1     | |  |
                |                                       j[16]|   1   | |  |
                |j[17]                                       |     1 | |  -
                |                                                  ... |  ...
                At this stage, the last variables in each chunk are solved for.

        Step 5) Chunk-wise forward substitution
                Every thread sends the solution of the last variable in its chunk to the next thread. This can use it to
                 substitute forward and recover the solution of the whole chunk.
                After this operation, the matrix structure is reduced as such:
                |j[0]| 1     |                                         |  -
                |j[1]|   1   |                                         |  |
                |j[2]|     1 |                                         |  |
                |j[3]        | 1     |                                 |  |
                |j[4]        |   1   |                                 |  | Warp 0
                |j[5]        |     1 |                                 |  |
                |j[6]                | 1     |                         |  |
                |j[7]                |   1   |                         |  |
                |j[8]                |     1 |                         |  -
                |j[9]                        | 1     |                 |  -
                |j[10]                       |   1   |                 |  |
                |j[11]                       |     1 |                 |  |
                |j[12]                               | 1     |         |  |
                |j[13]                               |   1   |         |  | Warp 1
                |j[14]                               |     1 |         |  |
                |j[15]                                       | 1     | |  |
                |j[16]                                       |   1   | |  |
                |j[17]                                       |     1 | |  -
                |                                                  ... |  ...
                The system is now completely reduced, and the solution can be copied back to memory.
    */

    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    // - handy aliases for data of the initial / final eq in a chunk
    jac_t& myJacI = myRegJac[0];
    jac_t& myJacF = myRegJac[chunk_size-1];
    rhs_t& myRhsI = myRegRhs[0];
    rhs_t& myRhsF = myRegRhs[chunk_size-1];

    // - handy extra indices for identifying thread position
    const int lIdx = threadIdx.x % threads_per_warp;
    const int wIdx = threadIdx.x / threads_per_warp;
    const bool lastOfWarp = (lIdx == (threads_per_warp-1));


    // Step 1) Chunk-wise Thomas reduction *****************************************************************************
    thomasReduction< pimpl_t >( myRegJac, myRegRhs );
    // At this point, all variables in a chunk depend on the last variable of the previous chunk

    // Step 2) Warp-wise parallel reduction ****************************************************************************
    parallelReductionWarp< pimpl_t >( myJacF, myRhsF, lIdx );
    // At this stage, all the last variables in a chunk depend on the last variable of the previous warp

    // Time to move on to shared memory stuff...

    // save to shared to allow access by other threads
    if( lastOfWarp && numMyEqs>0 ){
        pimpl_t::copyJac( sharedJac[wIdx], myJacF );
        pimpl_t::copyRhs( sharedRhs[wIdx], myRhsF );
    }

    // Step 3) Block-wise parallel reduction ***************************************************************************
    const idx_t maxDist = blockIdx.x < ( seqLength / (blockDim.x*chunk_size) ) ?
                            (blockDim.x * chunk_size)
                          : ( (seqLength % (blockDim.x*chunk_size)) ? (seqLength % (blockDim.x*chunk_size)) : seqLength );
    for( int po2 = 1; (po2*chunk_size*threads_per_warp) < maxDist; po2<<=1 ){
        const int prevEqIdx = wIdx - po2;

        // sync to make sure all threads have written
        __syncthreads();

        // reduce variable i using equation po2 places back
        if( prevEqIdx >= 0 && lastOfWarp && numMyEqs>0 ){ // this should make sure that only the correct eqs are updated
            pimpl_t::reduceEqs( sharedJac[prevEqIdx], myJacF, sharedRhs[prevEqIdx], myRhsF );
        }

        // sync to make sure all threads have read
        __syncthreads();
        if( lastOfWarp && numMyEqs>0 ){
            pimpl_t::copyJac( sharedJac[wIdx], myJacF );
            pimpl_t::copyRhs( sharedRhs[wIdx], myRhsF );
        }
    }
    // sync to make sure all threads have written
    __syncthreads();
    // At this stage, all last variables in the last chunk of each warp are solved for


    // Step 4) Warp-wise forward substitution **************************************************************************
    // All threads in warp bar the last read solution from previous warp
    jac_t jacPrev;
    rhs_t solPrev;
    if( wIdx>0 && !lastOfWarp ){
        pimpl_t::copyJac( jacPrev, sharedJac[wIdx-1] );
        pimpl_t::copyRhs( solPrev, sharedRhs[wIdx-1] );
    }else{
        pimpl_t::setToNeutralJac(jacPrev);
        pimpl_t::setToNeutralRhs(solPrev);
    }

    // All threads in warp bar the last solve for last eq in chunk and send sol to next thread
    pimpl_t::reduceEqs( jacPrev, myJacF, solPrev, myRhsF );
    pimpl_t::sendSolNextInWarp( jacPrev, myJacF, solPrev, myRhsF, lIdx );

    // Step 5) Chunk-wise forward substitution *************************************************************************
    #pragma unroll
    for( int t=0; t<chunk_size-1; ++t ){
        pimpl_t::reduceEqs( jacPrev, myRegJac[t], solPrev, myRegRhs[t] );
    }

    return;
}


template< typename pimpl_t >
__global__ void __launch_bounds__(1024, 1) parallelReductionShared( // prescribing some bounds to limit register usage
    typename pimpl_t::scalar_t *jac,
    typename pimpl_t::scalar_t *rhs,
    typename pimpl_t::scalar_t *jacTemp,
    typename pimpl_t::scalar_t *rhsTemp,
    typename pimpl_t::idx_t seqLength,
    typename pimpl_t::idx_t hiddenDim,
    typename pimpl_t::idx_t batchSize,
    bool updateJac = false,
    int numLoops = 1    // If I choose to solve sequentially over blocks, this tells me how many blocks I need to cover the whole sequence
){
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    // Data about a given chunk will sit on registers memory - Hopefully!
    jac_t myRegJac[chunk_size];
    rhs_t myRegRhs[chunk_size];
    // - shared memory for easy access
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)blockDim.x/threads_per_warp)
    //extern  __shared__ scalar_t temp[];                                           // this fails complaining that it can't identify the right temp for the various templates
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];          // this ensures unique naming for different templates instantiations
    scalar_t *temp = reinterpret_cast<scalar_t *>(my_smem);                         // might be __align__ is not needed: see https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    //    const size_t sharedMemSize =  ( sizeof(typename pimpl_t::jac_t) + sizeof(typename pimpl_t::rhs_t) ) * warpsPerBlock; // otherwise we can always allocate it statically... -> but this doesnt work, for some reason..
    //    alignas(scalar_t) __shared__ char temp[sharedMemSize];
    jac_t *sharedJac = (jac_t*) (temp);
    rhs_t *sharedRhs = (rhs_t*) (temp + warpsPerBlock * sizeof(jac_t) / sizeof(scalar_t) );


    for( int loop = 0; loop < numLoops; ++loop ){
        const int numMyEqs = countMyEqs< chunk_size, idx_t >( seqLength, hiddenDim, batchSize, loop );
        const idx_t myGlbDataIdx = getMyDataIdx< chunk_size, idx_t >( seqLength, hiddenDim, loop );

        if( loop>0 ){
            // If this block doesn't cover the very beginning of the seq, I need to store the last element of the sol
            __syncthreads();
            if( threadIdx.x == threads_per_block-1 && blockIdx.x == 0 ){
                pimpl_t::copyRhs( sharedRhs[0], myRegRhs[chunk_size-1] );
                pimpl_t::setToNeutralJac( sharedJac[0] );                   // TODO: useless?
            }
        }

        pimpl_t::readFromGlobal(
            jac, rhs,
            myRegJac, myRegRhs,
            numMyEqs, myGlbDataIdx
        );

        if( loop>0 ){
            // If this block doesn't cover the very beginning of the seq, I need to use the previous last sol to solve for the first current one
            __syncthreads();
            if( threadIdx.x == 0 && blockIdx.x == 0 ){
                pimpl_t::reduceEqs( sharedJac[0], myRegJac[0], sharedRhs[0], myRegRhs[0] );
            }
        }

        // TODO: set very first Jacobian to 0 - this could be handled outside to not impact function's behaviour
        if( threadIdx.x == 0 && blockIdx.x == 0 ){
            pimpl_t::setToZeroJac( myRegJac[0] );
        }

        //Solve!
        parallelReductionSharedInner< pimpl_t >(
            myRegJac, myRegRhs, sharedJac, sharedRhs, numMyEqs, seqLength
        );

        pimpl_t::writeToGlobal(
            jacTemp,
            rhsTemp,
            myRegJac, myRegRhs,
            numMyEqs, myGlbDataIdx,
            updateJac
        );

    }

    return;
}




// PCR reduction across blocks *****************************************************************************************
template< typename pimpl_t >
__global__ void __launch_bounds__(1024, 1) parallelReductionGlobal( // prescribing some bounds to limit register usage
    typename pimpl_t::scalar_t *jac,
    typename pimpl_t::scalar_t *rhs,
    typename pimpl_t::idx_t seqLength,
    typename pimpl_t::idx_t hiddenDim,
    typename pimpl_t::idx_t batchSize
){
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    const bool isInside = ( ( (threadIdx.x+1) * (threads_per_block * chunk_size) - 1) < seqLength )
                       && ( ( threadIdx.y + blockIdx.y * blockDim.y ) < hiddenDim )
                       && ( ( threadIdx.z + blockIdx.z * blockDim.z ) < batchSize ); // thread holds (some) matrix data?

    const idx_t myDataIdx = ( ( (threadIdx.x+1) * (threads_per_block * chunk_size) - 1 )
                            + ( threadIdx.y + blockIdx.y * blockDim.y ) * seqLength
                            + ( threadIdx.z + blockIdx.z * blockDim.z ) * seqLength * hiddenDim
                            ) * sizeof(rhs_t) / sizeof(scalar_t); // first idx of data relevant to thread

    const int lIdx = threadIdx.x % threads_per_warp;
    const int wIdx = threadIdx.x / threads_per_warp;
    const bool  lastOfWarp = (lIdx == (threads_per_warp-1));

    // Data about a given chunk will sit on registers memory - Hopefully!
    jac_t myRegJac;
    rhs_t myRegRhs;

    jac_t* myJacData = isInside? (jac_t*) (jac + myDataIdx * sizeof(jac_t) / sizeof(rhs_t)) : 0;
    rhs_t* myRhsData = isInside? (rhs_t*) (rhs + myDataIdx) : 0;

    // - read from global memory
    if( isInside ){
        pimpl_t::copyJac( myRegJac, myJacData[0] );
        pimpl_t::copyRhs( myRegRhs, myRhsData[0] );
    }else{
        pimpl_t::setToNeutralJac(myRegJac);
        pimpl_t::setToNeutralRhs(myRegRhs);
    }

    // Warp-wise parallel reduction ************************************************************************************
    parallelReductionWarp< pimpl_t >( myRegJac, myRegRhs, lIdx);

    // Populate shared memory for easy access
    const int warpsPerBlock = 1 + ((blockDim.x - 1) / (threads_per_warp)); // = ceil((float)blockDim.x/threads_per_warp)
    //extern  __shared__ scalar_t temp[];                                           // this fails complaining that it can't identify the right temp for the various templates
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];          // this ensures unique naming for different templates instantiations. Might be __align__ is not needed: see https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    scalar_t *temp = reinterpret_cast<scalar_t *>(my_smem);                         // notice this CANNOT be instantiated statically... :(

    jac_t *sharedJac = (jac_t*) (temp);
    rhs_t *sharedRhs = (rhs_t*) (temp + warpsPerBlock * sizeof(jac_t) / sizeof(scalar_t) );

    // - save to shared to allow access by other threads
    if( lastOfWarp && isInside ){
        pimpl_t::copyJac( sharedJac[wIdx], myRegJac );
        pimpl_t::copyRhs( sharedRhs[wIdx], myRegRhs );
    }


    // Block-wise parallel reduction ***********************************************************************************
    for( int po2 = 1; po2 < warpsPerBlock; po2<<=1 ){
        const int prevEqIdx = wIdx - po2;

        // sync to make sure all threads have written
        __syncthreads();

        // reduce variable i using equation po2 places back
        if( prevEqIdx >= 0 && lastOfWarp && isInside ){ // this should make sure that only the correct eqs are updated
            pimpl_t::reduceEqs( sharedJac[prevEqIdx], myRegJac, sharedRhs[prevEqIdx], myRegRhs );
        }

        // sync to make sure all threads have read
        __syncthreads();
        if( lastOfWarp && isInside){
            pimpl_t::copyJac( sharedJac[wIdx], myRegJac );
            pimpl_t::copyRhs( sharedRhs[wIdx], myRegRhs );
        }
    }
    // sync to make sure all threads have written
    __syncthreads();


    // At this stage, all last variables in the last chunk of each warp are solved for

    // Warp-wise forward substitution **********************************************************************************
    // All threads in warp read solution from previous warp
    jac_t jacPrev;
    rhs_t solPrev;
    if( wIdx>0 && !lastOfWarp ){
        pimpl_t::copyJac( jacPrev, sharedJac[wIdx-1] );
        pimpl_t::copyRhs( solPrev, sharedRhs[wIdx-1] );
    }else{
        pimpl_t::setToNeutralJac(jacPrev);
        pimpl_t::setToNeutralRhs(solPrev);
    }

    // All threads in warp bar the last solve for last eq in chunk and send sol to next thread
    pimpl_t::reduceEqs( jacPrev, myRegJac, solPrev, myRegRhs );
    pimpl_t::sendSolNextInWarp( jacPrev, myRegJac, solPrev, myRegRhs, lIdx );

    // Chunk-wise forward substitution *********************************************************************************
    pimpl_t::reduceEqs( jacPrev, myRegJac, solPrev, myRegRhs );


    // Write back to output
    if(isInside){
        pimpl_t::copyJac( myJacData[0], myRegJac );      // TODO: useless
        pimpl_t::copyRhs( myRhsData[0], myRegRhs );
    }

    return;
}




// Post-parallel-reduction cleanup *************************************************************************************
template< typename pimpl_t >
__global__ void __launch_bounds__(1024, 1) finalReduction(  // prescribing some bounds to limit register usage
    typename pimpl_t::scalar_t *jac,
    typename pimpl_t::scalar_t *rhs,
    typename pimpl_t::idx_t seqLength,
    typename pimpl_t::idx_t hiddenDim,
    typename pimpl_t::idx_t batchSize
){
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    const int numMyEqs = countMyEqs< chunk_size, idx_t >( seqLength, hiddenDim, batchSize, 1 );
    const idx_t myGlbDataIdx = getMyDataIdx< chunk_size, idx_t >( seqLength, hiddenDim, 1 );

    const idx_t prevDataIdx = ( ( (blockIdx.x+1) * blockDim.x ) * chunk_size - 1
                            + ( threadIdx.y + blockIdx.y * blockDim.y ) * seqLength
                            + ( threadIdx.z + blockIdx.z * blockDim.z ) * seqLength * hiddenDim
                            ) * sizeof(rhs_t) / sizeof(scalar_t);  // data with previous solution

    // Data about a given chunk will sit on registers memory - Hopefully!
    jac_t myRegJac[chunk_size];
    rhs_t myRegRhs[chunk_size];

    pimpl_t::readFromGlobal(
        jac, rhs,
        myRegJac, myRegRhs,
        numMyEqs, myGlbDataIdx
    );

    jac_t jacPrev;
    rhs_t solPrev;
    if( numMyEqs>0 ){
        pimpl_t::copyJac( jacPrev, *((jac_t(*)) &jac[prevDataIdx * sizeof(jac_t) / sizeof(rhs_t)]));
        pimpl_t::copyRhs( solPrev, *((rhs_t(*)) &rhs[prevDataIdx]                                ));
    }else{
        pimpl_t::setToNeutralJac(jacPrev);
        pimpl_t::setToNeutralRhs(solPrev);
    }

    #pragma unroll
    for( int t=0; t<chunk_size; ++t ){
        pimpl_t::reduceEqs( jacPrev, myRegJac[t], solPrev, myRegRhs[t] );
    }

    // Write back to output
    pimpl_t::writeToGlobal(
        jac, rhs,
        myRegJac, myRegRhs,
        numMyEqs, myGlbDataIdx,
        false
    );

    return;

}





// *********************************************************************************************************************
// BACKWARD SUBSTITUTION
// *********************************************************************************************************************

// Thomas reduction - BACKWARD *****************************************************************************************
template< typename pimpl_t >
__forceinline__ __device__ void thomasBwdReduction(
    typename pimpl_t::jac_t (&jac)[pimpl_t::chunk_size],
    typename pimpl_t::rhs_t (&rhs)[pimpl_t::chunk_size]
){
    static constexpr int chunk_size = pimpl_t::chunk_size;

    for( int t = chunk_size-2; t >= 0; --t ){
        pimpl_t::reduceEqs( jac[t+1], jac[t], rhs[t+1], rhs[t] );
    }
}


// parallel reduction within warp - BACKWARD ***************************************************************************
template< typename pimpl_t >
__forceinline__ __device__ void parallelBwdReductionWarp(
    typename pimpl_t::jac_t (&jac),
    typename pimpl_t::rhs_t (&rhs),
    int lIdx
){
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;

    for( int po2 = 1; po2 < threads_per_warp; po2 <<= 1) {
        pimpl_t::bwdReduceEqsWarp( jac, rhs, po2, lIdx );
    }
}


// Parallel reduction within block - BACKWARD **************************************************************************
// Assumptions:
// - Time is the leading dimension (t=x), of size T
// - each thread along the t (=x) dimension handles chunk_size data
// - grid dimension y is the inner state dimension (so one thread per y)
// - grid dimension z is the batch size (so one thread per z)
template< typename pimpl_t >
__device__ void parallelBwdReductionSharedInner(
    typename pimpl_t::jac_t (&myRegJac)[pimpl_t::chunk_size],
    typename pimpl_t::rhs_t (&myRegRhs)[pimpl_t::chunk_size],
    typename pimpl_t::jac_t *sharedJac,
    typename pimpl_t::rhs_t *sharedRhs,
    int numMyEqs,
    typename pimpl_t::idx_t seqLength
){

    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    // - handy aliases for data of the initial / final eq in a chunk
    jac_t& myJacI = myRegJac[0];
    jac_t& myJacF = myRegJac[chunk_size-1];
    rhs_t& myRhsI = myRegRhs[0];
    rhs_t& myRhsF = myRegRhs[chunk_size-1];

    // - handy extra indices for identifying thread position
    const int lIdx = threadIdx.x % threads_per_warp;
    const int wIdx = threadIdx.x / threads_per_warp;
    const bool firstOfWarp = (lIdx == 0);
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)THREADS_PER_BLOCK/THREADS_PER_WARP)


    // Step 1) Chunk-wise Thomas reduction *****************************************************************************
    thomasBwdReduction< pimpl_t >( myRegJac, myRegRhs );
    // At this point, all variables in a chunk depend on the first variable of the next chunk

    // Step 2) Warp-wise PCR reduction *********************************************************************************
    parallelBwdReductionWarp< pimpl_t >( myJacI, myRhsI, lIdx );
    // At this stage, all the first variables in a chunk depend on the first variable of the next warp

    // Time to move on to shared memory stuff...

    // save to shared to allow access by other threads
    if( firstOfWarp && numMyEqs>0 ){
        pimpl_t::copyJac( sharedJac[wIdx], myJacI );
        pimpl_t::copyRhs( sharedRhs[wIdx], myRhsI );
    }

    // Step 3) Block-wise parallel reduction ***************************************************************************
    const idx_t maxDist = blockIdx.x < ( seqLength / (blockDim.x*chunk_size) ) ?
                            (blockDim.x * chunk_size)
                          : ( (seqLength % (blockDim.x*chunk_size)) ? (seqLength % (blockDim.x*chunk_size)) : seqLength );
    for( int po2 = 1; (po2*chunk_size*threads_per_warp) < maxDist; po2<<=1 ){
        const int nextEqIdx = wIdx + po2;

        // sync to make sure all threads have written
        __syncthreads();

        // reduce variable i using equation po2 places ahead
        if( nextEqIdx < warpsPerBlock && nextEqIdx * threads_per_warp * chunk_size < seqLength && firstOfWarp && numMyEqs>0 ){ // this should make sure that only the correct eqs are updated
            pimpl_t::reduceEqs( sharedJac[nextEqIdx], myJacI, sharedRhs[nextEqIdx], myRhsI );
        }

        // sync to make sure all threads have read
        __syncthreads();
        if( firstOfWarp && numMyEqs>0 ){
            pimpl_t::copyJac( sharedJac[wIdx], myJacI );
            pimpl_t::copyRhs( sharedRhs[wIdx], myRhsI );
        }
    }
    // sync to make sure all threads have written
    __syncthreads();
    // At this stage, all first variables in the first chunk of each warp are solved for

    // Step 4) Warp-wise backward substitution *************************************************************************
    // All threads in warp bar the first read solution from next warp
    jac_t jacNext;
    rhs_t solNext;

    if( wIdx<warpsPerBlock-1 && !firstOfWarp ){
        pimpl_t::copyJac( jacNext, sharedJac[wIdx+1] );
        pimpl_t::copyRhs( solNext, sharedRhs[wIdx+1] );
    }else{
        pimpl_t::setToNeutralJac(jacNext);
        pimpl_t::setToNeutralRhs(solNext);
    }

    // All threads in warp bar the first solve for first eq in chunk and send sol to prev thread
    pimpl_t::reduceEqs( jacNext, myJacI, solNext, myRhsI );
    pimpl_t::sendSolPrevInWarp( jacNext, myJacI, solNext, myRhsI, lIdx );

    // Step 5) Chunk-wise backward substitution ************************************************************************
    #pragma unroll
    for( int t=1; t<chunk_size; ++t ){
        pimpl_t::reduceEqs( jacNext, myRegJac[t], solNext, myRegRhs[t] );
    }

    // */

    return;

}











// Main Kernel *********************************************************************************************************
template< typename pimpl_t,
          int max_sequential_steps >
void parallelReduceLauncher(
        typename pimpl_t::scalar_t *jac,
        typename pimpl_t::scalar_t *rhs,
        typename pimpl_t::scalar_t *jacTemp,
        typename pimpl_t::scalar_t *rhsTemp,
        typename pimpl_t::idx_t seqLength,
        typename pimpl_t::idx_t hiddenDim,
        typename pimpl_t::idx_t batchSize
    ) {
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    // Define grid and block dimensions
    const int blocksPerSeq = 1 + ((seqLength - 1) / (threads_per_block * chunk_size)); // = ceil((float)seqLength/(THREADS_PER_BLOCK * chunk_size))
    dim3 blockSize( threads_per_block, 1, 1 ); // One-dimensional block

    // Define amount of shared memory necessary per block
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)THREADS_PER_BLOCK/THREADS_PER_WARP)
    unsigned int sharedMemSize =  ( sizeof(typename pimpl_t::jac_t) + sizeof(typename pimpl_t::rhs_t) ) * warpsPerBlock;

//    std::cout<<"Launching kernel on gridsize  ("<< gridSize.x  <<","<< gridSize.y  <<","<< gridSize.z  <<",)"<<std::endl
//             <<"with blocks of size blockSize ("<< blockSize.x <<","<< blockSize.y <<","<< blockSize.z <<",)"<<std::endl
//             <<"for tensors ot size (B,N,T):  ("<< batchSize   <<","<< hiddenDim   <<","<< seqLength   <<",)"<<std::endl
//             <<"Total memory allocated: "       << sharedMemSize <<" for "<<warpsPerBlock<< " warps."<<std::endl;

    if ( blocksPerSeq <= max_sequential_steps ){
        dim3 gridSize( 1, hiddenDim, batchSize );
        parallelReductionShared< pimpl_t >
                          <<<gridSize, blockSize, sharedMemSize>>>(
                          jac, rhs, jacTemp, rhsTemp,
                          seqLength, hiddenDim, batchSize,
                          false, blocksPerSeq );
    } else {
        dim3 gridSize( blocksPerSeq, hiddenDim, batchSize );
        parallelReductionShared< pimpl_t >
                          <<<gridSize, blockSize, sharedMemSize>>>(
                          jac, rhs, jacTemp, rhsTemp,
                          seqLength, hiddenDim, batchSize,
                          true, 1 );

        cudaDeviceSynchronize();                          // wait for previous reduction to complete
        const dim3 gridSize2( 1, hiddenDim, batchSize );
        const dim3 blockSize2(blocksPerSeq, 1, 1);        // TODO: must put a check to ensure whole T is covered
        const int warpsPerBlock2 = 1 + ((blocksPerSeq - 1) / (threads_per_warp)); // = ceil((float)blocksPerSeq/THREADS_PER_WARP)
        const int sharedMemSize2 = ( sizeof(typename pimpl_t::jac_t) + sizeof(typename pimpl_t::rhs_t) ) * warpsPerBlock2;

//        std::cout<<"Multi-block reduction needed for chunk_size "<<chunk_size<<std::endl;
//        std::cout<<"Launching kernel on gridsize  ("<< gridSize2.x  <<","<< gridSize2.y  <<","<< gridSize2.z  <<",)"<<std::endl
//                 <<"with blocks of size blockSize ("<< blockSize2.x <<","<< blockSize2.y <<","<< blockSize2.z <<",)"<<std::endl
//                 <<"for tensors ot size (B,N,T):  ("<< batchSize    <<","<< hiddenDim    <<","<< seqLength    <<",)"<<std::endl
//                 <<"Total memory allocated: "       << sharedMemSize2 <<" for "<<warpsPerBlock2<< " warps."<<std::endl;

        parallelReductionGlobal< pimpl_t >
                          <<<gridSize2, blockSize2, sharedMemSize2>>>(
                          jacTemp, rhsTemp, seqLength, hiddenDim, batchSize
                          );
        cudaDeviceSynchronize();  // TODO: same here. seqLength must be massive to need multiple blocks to reduce

        const dim3 gridSize3( blocksPerSeq-1, hiddenDim, batchSize );
        const dim3 blockSize3(threads_per_block, 1, 1);

//        std::cout<<"Launching kernel on gridsize  ("<< gridSize3.x  <<","<< gridSize3.y  <<","<< gridSize3.z  <<",)"<<std::endl
//                 <<"with blocks of size blockSize ("<< blockSize3.x <<","<< blockSize3.y <<","<< blockSize3.z <<",)"<<std::endl
//                 <<"for tensors ot size (B,N,T):  ("<< batchSize    <<","<< hiddenDim    <<","<< seqLength    <<",)"<<std::endl;

        finalReduction< pimpl_t >
                      <<<gridSize3, blockSize3, 0>>>(
                      jacTemp, rhsTemp, seqLength, hiddenDim, batchSize);
    }
}






