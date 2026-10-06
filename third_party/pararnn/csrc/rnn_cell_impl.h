//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>          // Required for __half
#include <cuda_bf16.h>          // Required for __nv_bfloat16

#include "helpers.h"
#include "nonlinearities.h"


// *********************************************************************************************************************
// NB: The big assumption here is that a single thread can hold data about (multiple) equations!
//     For dense systems (but even unstructured sparse ones), this is very much *not* the case!
// *********************************************************************************************************************

// Extra support for mixed half/bool arithmetic
__forceinline__ __device__ __half operator*(const __half a, const bool b) { return b? a: __float2half(0.0f); }  // Equivalent to a * (b ? 1 : 0)
__forceinline__ __device__ __half operator+(const __half a, const bool b) { return a + __float2half(b ? 1.0f : 0.0f);}

__forceinline__ __device__ __nv_bfloat16 operator*(const __nv_bfloat16 a, const bool b) { return b? a: __float2bfloat16(0.0f); }  // Equivalent to a * (b ? 1 : 0)
__forceinline__ __device__ __nv_bfloat16 operator+(const __nv_bfloat16 a, const bool b) { return a + __float2bfloat16(b ? 1.0f : 0.0f);}

// And for constructors from float
template <typename scalar_t>
__device__ scalar_t zero(){ return static_cast<scalar_t>(0.0f); };

template <typename scalar_t>
__device__ scalar_t minusOne(){ return static_cast<scalar_t>(-1.0f); };

// If __CUDA_NO_HALF_OPERATORS__ or __CUDA_NO_HALF_CONVERSIONS__ are set, these are handy
//__forceinline__ __device__ __half operator-(const __half a) {return __hneg(a);}
//__forceinline__ __device__ __half operator*(const __half a, const __half b) { return __hmul(a, b); }
//__forceinline__ __device__ __half& operator*=(__half& a, const __half& b){ a=__hmul(a,b); return a; }
//__forceinline__ __device__ __half& operator+=(__half& a, const __half& b){ a=__hadd(a,b); return a; }
//__forceinline__ __device__ __half& operator-=(__half& a, const __half& b){ a=__hsub(a,b); return a; }
//__forceinline__ __device__ __half operator+(const __half a, const __half b) { return __hadd(a, b); }
//template<> __device__ __half minusOne(){ return __float2half(-1.0f); };
//template<> __device__ __nv_bfloat16 minusOne(){ return __float2bfloat16(-1.0f); };
//template<> __device__ __half zero(){ return __float2half(0.0f); };
//template<> __device__ __nv_bfloat16 zero(){ return __float2bfloat16(0.0f); };


// Forward defs of template classes
template < typename Derived >
class RNNCellBaseImpl;

template < typename scalar_t, typename Derived >
class RNNCellDiagImpl;

template < typename scalar_t, int N, typename Derived >
class RNNCellBlockDiagImpl;



// *********************************************************************************************************************
// Traits for RNNCellImpl
// *********************************************************************************************************************
class RNNCellBaseTraits{
public:
    static constexpr int threads_per_warp  = THREADS_PER_WARP;
    static constexpr int threads_per_block = THREADS_PER_BLOCK;
    static constexpr int max_newton_its = MAX_NEWTON_ITS;

    using idx_t = int;
};

template< typename RNNCellImpl_t >
class RNNCellTraits: public RNNCellBaseTraits{};


template< typename scalar_type, typename Derived >
class RNNCellTraits< RNNCellDiagImpl< scalar_type, Derived > >: public RNNCellBaseTraits{
public:
    using scalar_t = scalar_type;
    using rhs_t = scalar_t;
    using jac_t = scalar_t;
    static constexpr int chunk_size      = dtype2chunkSizeDiag<scalar_t>;
    static constexpr int num_hidden_vars = 1;
};


template< typename scalar_type, int N, typename Derived >
class RNNCellTraits< RNNCellBlockDiagImpl< scalar_type, N, Derived > >: public RNNCellBaseTraits{
public:
    using scalar_t = scalar_type;
    using rhs_t = scalar_t[N];
    using jac_t = scalar_t[N][N];
    static constexpr int chunk_size      = dtype2chunkSizeBlockDiag<N,scalar_t>;
    static constexpr int num_hidden_vars = N;
};




// *********************************************************************************************************************
// Base class for RNNCellImpl
// *********************************************************************************************************************

template < typename Derived >
class RNNCellBaseImpl{
public:

    using Traits = RNNCellTraits< Derived >;
    using idx_t = typename Traits::idx_t;
    static constexpr int chunk_size      = Traits::chunk_size;
    static constexpr int num_hidden_vars = Traits::num_hidden_vars;
    using scalar_t = typename Traits::scalar_t;
    using rhs_t = typename Traits::rhs_t;
    using jac_t = typename Traits::jac_t;


    // Overloading of setters to "Neutral/Zero" values for Jacobian / RHS for full chunk_size **************************
    // - Actually, overloading is not a good idea, as it confuses the compiler - so, different name
    __forceinline__ __device__ static void setToNeutralJacChunk( jac_t (&jac)[chunk_size] ){
        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            Derived::setToNeutralJac( jac[t] );
        }
    }
    __forceinline__ __device__ static void setToZeroJacChunk( jac_t (&jac)[chunk_size] ){
        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            Derived::setToZeroJac( jac[t] );
        }
    }
    __forceinline__ __device__ static void setToNeutralRhsChunk( rhs_t (&rhs)[chunk_size] ){
        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            Derived::setToNeutralRhs( rhs[t] );
        }
    }
    __forceinline__ __device__ static void setToZeroRhsChunk( rhs_t (&rhs)[chunk_size] ){
        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            Derived::setToZeroRhs( rhs[t] );
        }
    }
    __forceinline__ __device__ static void copyRhsChunk( rhs_t (&dst)[chunk_size], const rhs_t (&src)[chunk_size] ){
        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            Derived::copyRhs( dst[t], src[t] );
        }
    }
    __forceinline__ __device__ static void copyJacChunk( jac_t (&dst)[chunk_size], const jac_t (&src)[chunk_size] ){
        #pragma unroll
        for ( int t=0; t<chunk_size; ++t ){
            Derived::copyJac( dst[t], src[t] );
        }
    }
    __forceinline__ __device__ static void updateSolChunk( rhs_t (&h)[chunk_size], const rhs_t (&update)[chunk_size] ){
        #pragma unroll
        for (int t=0; t<chunk_size; ++t ){
            Derived::updateSol( h[t], update[t] );
        }
    }



    // Reading/writing jac/rhs values from global memory ***************************************************************
    __forceinline__ __device__ static void readFromGlobal(
        const scalar_t* jac, const scalar_t* rhs,
        jac_t (&myJac)[chunk_size],
        rhs_t (&myRhs)[chunk_size],
        int numMyEqs, idx_t myGlbDataIdx
    ){
        const jac_t* myJacData = (numMyEqs>0)? (jac_t*) (jac + myGlbDataIdx * num_hidden_vars * num_hidden_vars) : 0;
        const rhs_t* myRhsData = (numMyEqs>0)? (rhs_t*) (rhs + myGlbDataIdx * num_hidden_vars ) : 0;
        // - read from global memory
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            if( t < numMyEqs ){
                Derived::copyJac( myJac[t], myJacData[t] );
                Derived::copyRhs( myRhs[t], myRhsData[t] );
            }else{
                Derived::setToNeutralJac( myJac[t] );
                Derived::setToNeutralRhs( myRhs[t] );
            }
        }
    }

    __forceinline__ __device__ static void writeToGlobal(
        scalar_t* jac, scalar_t* rhs,
        const jac_t (&myJac)[chunk_size],
        const rhs_t (&myRhs)[chunk_size],
        int numMyEqs, idx_t myGlbDataIdx,
        bool updateJac=false
    ){
        jac_t* myJacData = (numMyEqs>0)? (jac_t*) ( jac + myGlbDataIdx * num_hidden_vars * num_hidden_vars ) : 0;
        rhs_t* myRhsData = (numMyEqs>0)? (rhs_t*) ( rhs + myGlbDataIdx * num_hidden_vars ) : 0;
        // - write to global memory
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            if( t < numMyEqs ){
                if( updateJac ){
                    Derived::copyJac( myJacData[t], myJac[t] );
                }
                Derived::copyRhs( myRhsData[t], myRhs[t] );
            }
        }
    }

    __forceinline__ __device__ static void writeToGlobal(       // Only rhs (solution)
        int numMyEqs, idx_t myGlbDataIdx,
        const rhs_t (&myRhs)[chunk_size],
        scalar_t* rhs
    ){
        rhs_t* myRhsData = (numMyEqs>0)? (rhs_t *) (rhs + myGlbDataIdx * num_hidden_vars) : 0;
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            if( t < numMyEqs ){
                Derived::copyRhs(myRhsData[t], myRhs[t]);
            }
        }
    }

};



// *********************************************************************************************************************
// Specialisation of RNNCellImpl with diagonal Jacobians (independent systems)
// *********************************************************************************************************************

template < typename scalar_t, typename Derived = void >
class RNNCellDiagImpl : public RNNCellBaseImpl<
    std::conditional_t< std::is_void_v< Derived >, RNNCellDiagImpl< scalar_t >,
    Derived >
> {
public:

    using Traits = RNNCellTraits< std::conditional_t< std::is_void_v< Derived >, RNNCellDiagImpl< scalar_t >, Derived > >;
    static constexpr int chunk_size      = Traits::chunk_size;
    static constexpr int num_hidden_vars = Traits::num_hidden_vars;
    static constexpr int threads_per_warp  = Traits::threads_per_warp;
    static constexpr int threads_per_block = Traits::threads_per_block;
    static_assert(std::is_same_v<scalar_t, typename Traits::scalar_t>,
                  "ERROR: scalar_t template parameter doesn't match Traits::scalar_t! "
                  "Check RNNCellImplTraits specialization.");
    using rhs_t = typename Traits::rhs_t;
    using jac_t = typename Traits::jac_t;
    using idx_t = typename Traits::idx_t;



    // Setters for Jacobian / Right-Hand Sides *************************************************************************
    // - "Neutral" are defined so that reducing equations using these values should leave them unchanged
    __forceinline__ __device__ static void setToNeutralJac( jac_t& jac ){
        jac = minusOne< scalar_t >();
    }
    __forceinline__ __device__ static void setToZeroJac( jac_t& jac ){
        jac = zero< scalar_t >();
    }
    __forceinline__ __device__ static void setToNeutralRhs( rhs_t& rhs ){
        rhs = zero< scalar_t >();
    }
    __forceinline__ __device__ static void setToZeroRhs( rhs_t& rhs ){
        rhs = zero< scalar_t >();
    }
    __forceinline__ __device__ static void copyRhs( rhs_t& dst, const rhs_t& src ){
        dst = src;
    }
    __forceinline__ __device__ static void copyJac( jac_t& dst, const jac_t& src ){
        dst = src;
    }
    __forceinline__ __device__ static void updateSol( rhs_t& h, const rhs_t& update ){
        h += update;
    }


    // Equation reductions *********************************************************************************************
    __forceinline__ __device__ static void reduceEqs(        // Vanilla reduction of current eq using other
        const jac_t& jacOther, jac_t& jac,
        const rhs_t& rhsOther, rhs_t& rhs
    ){
        rhs -= jac * rhsOther;
        jac *=     - jacOther;
    }
    __forceinline__ __device__ static void reduceEqsWarp(    // Reduction communicating within threads in a warp
        jac_t& jac, rhs_t& rhs,
        int po2, int lIdx
    ){
        // Only reduce if I'm fetching values from within the warp, otherwise leave untouched
        const bool reduce = ( lIdx>=po2 );
        rhs -= jac * __shfl_up_sync( 0xffffffff, rhs, po2 ) * reduce;
        jac *=     - __shfl_up_sync( 0xffffffff, jac, po2 ) * reduce + (!reduce);
    }
    __forceinline__ __device__ static void bwdReduceEqsWarp(  // Same as above, but flipping sides (backward)
        jac_t& jac, rhs_t& rhs,
        int po2, int lIdx
    ){
        const bool reduce = ( lIdx < ( threads_per_warp - po2 ) );
        rhs -= jac * __shfl_down_sync( 0xffffffff, rhs, po2 ) * reduce;
        jac *=     - __shfl_down_sync( 0xffffffff, jac, po2 ) * reduce + (!reduce);
    }

    // In-warp communications ******************************************************************************************
    __forceinline__ __device__ static void sendSolNextInWarp(
        jac_t& jacPrev, const jac_t& jac,
        rhs_t& rhsPrev, const rhs_t& rhs,
        int lIdx, int po2=1
    ){
        const bool firstOfWarp = ( lIdx==0 );
        rhsPrev = __shfl_up_sync( 0xffffffff, rhs, po2 ) * (!firstOfWarp) + rhsPrev * firstOfWarp;
        jacPrev = __shfl_up_sync( 0xffffffff, jac, po2 ) * (!firstOfWarp) + jacPrev * firstOfWarp;
    }
    __forceinline__ __device__ static void sendSolNextInWarp(
        rhs_t& rhsPrev, const rhs_t& rhs,
        bool firstOfWarp
    ){
        rhsPrev = __shfl_up_sync( 0xffffffff, rhs, 1 ) * (!firstOfWarp) + rhsPrev * firstOfWarp;
    }
    __forceinline__ __device__ static void sendJacNextInWarp(
        jac_t& jacPrev, const jac_t& jac,
        bool firstOfWarp
    ){
        jacPrev = __shfl_up_sync( 0xffffffff, jac, 1 ) * (!firstOfWarp) + jacPrev * firstOfWarp;
    }
    // Same as above, but flipping sides (backward)
    __forceinline__ __device__ static void sendSolPrevInWarp(
        jac_t& jacNext, const jac_t& jac,
        rhs_t& rhsNext, const rhs_t& rhs,
        int lIdx, int po2=1
    ){
        const bool lastOfWarp = ( lIdx == threads_per_warp-1 );
        rhsNext = __shfl_down_sync( 0xffffffff, rhs, po2 ) * (!lastOfWarp) + rhsNext * lastOfWarp;
        jacNext = __shfl_down_sync( 0xffffffff, jac, po2 ) * (!lastOfWarp) + jacNext * lastOfWarp;
    }
    __forceinline__ __device__ static void sendSolPrevInWarp(
        rhs_t& rhsNext, const rhs_t& rhs,
        bool lastOfWarp
    ){
        rhsNext = __shfl_down_sync( 0xffffffff, rhs, 1 ) * ( !lastOfWarp ) + rhsNext * lastOfWarp;
    }
    __forceinline__ __device__ static void sendJacPrevInWarp(
        jac_t& jacNext, const jac_t& jac,
        bool lastOfWarp
    ){
        jacNext = __shfl_down_sync( 0xffffffff, jac, 1 ) * (!lastOfWarp) + jacNext * lastOfWarp;
    }


    // Extended interface when deriving from this class ****************************************************************
    // SFINAE: Only available when inherited from!
    template< typename TupleType, bool Enable = !std::is_void_v< Derived > >
    __forceinline__ __device__ static std::enable_if_t< Enable, void > computeNegativeResidual(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const TupleType& lclVars,
        rhs_t (&res)[chunk_size]
    ){
        Derived::recurrenceStep( hm1, lclVars, h );
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            res[t] -= h[t];
        }
        return;
    }


    template< typename TupleType, bool Enable = !std::is_void_v< Derived > >
    __forceinline__ __device__ static std::enable_if_t< Enable, void > assembleInitialGuess(
        rhs_t (&h)[chunk_size],
        const TupleType& lclVars
    ){
        rhs_t hm1[chunk_size];
        setToZeroRhsChunk( hm1 );
        Derived::recurrenceStep( hm1, lclVars, h );
        return;
    }


    template< typename TupleType, bool Enable = !std::is_void_v< Derived > >
    __forceinline__ __device__ static std::enable_if_t< Enable, void > computeJacobiansTranspose(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const TupleType& lclVars,
        jac_t (&jac)[chunk_size]
    ){
        Derived::computeJacobians( h, hm1, lclVars, jac );
        // Transpose! (do nothing)
        return;
    }



};







// *********************************************************************************************************************
// Specialisation of RNNCellImpl with Block-diagonal Jacobians (independent systems with multiple hidden variables)
// *********************************************************************************************************************

template < typename scalar_t, int N, typename Derived = void >
class RNNCellBlockDiagImpl : public RNNCellBaseImpl<
    std::conditional_t< std::is_void_v< Derived >, RNNCellBlockDiagImpl< scalar_t, N, void >,
    Derived >
> {
public:

    using Traits = RNNCellTraits< std::conditional_t< std::is_void_v< Derived >, RNNCellBlockDiagImpl< scalar_t, N, void >, Derived > >;
    static constexpr int chunk_size      = Traits::chunk_size;
    static constexpr int num_hidden_vars = Traits::num_hidden_vars;
    static constexpr int threads_per_warp  = Traits::threads_per_warp;
    static constexpr int threads_per_block = Traits::threads_per_block;
    static_assert(std::is_same_v<scalar_t, typename Traits::scalar_t>,
                  "ERROR: scalar_t template parameter doesn't match Traits::scalar_t! "
                  "Check RNNCellImplTraits specialization.");
    using rhs_t = typename Traits::rhs_t;
    using jac_t = typename Traits::jac_t;
    using idx_t = typename Traits::idx_t;


    // Setters for Jacobian / Right-Hand Sides *************************************************************************
    // - "Neutral" are defined so that reducing equations using these values should leave them unchanged
    __forceinline__ __device__ static void setToNeutralJac( jac_t& jac ){   // -I
        setToZeroJac( jac );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            jac[i][i] = minusOne< scalar_t >();
        }
    }

    __forceinline__ __device__ static void setToNeutralRhs( rhs_t& rhs ){
        setToZeroRhs( rhs );
    }

    __forceinline__ __device__ static void setToZeroJac( jac_t& jac ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jac[i][j] = zero< scalar_t >();
            }
        }
    }

    __forceinline__ __device__ static void setToZeroRhs( rhs_t& rhs ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhs[i] = zero< scalar_t >();
        }
    }

    __forceinline__ __device__ static void copyJac( jac_t& dst, const jac_t& src ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                dst[i][j] = src[i][j];
            }
        }
    }

    __forceinline__ __device__ static void copyRhs( rhs_t& dst, const rhs_t& src ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            dst[i] = src[i];
        }
    }

    __forceinline__ __device__ static void updateSol( rhs_t& h, const rhs_t& update ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            h[i] += update[i];
        }
    }


    // Equation reductions *********************************************************************************************
    __forceinline__ __device__ static void reduceEqs(        // Vanilla reduction of current eq using other
        const jac_t& jacOther, jac_t& jac,
        const rhs_t& rhsOther, rhs_t& rhs
    ){
        // Update rhs
        rhs_t rhsUpdate;
        matVecMulReg( jac, rhsOther, rhsUpdate );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhs[i] -= rhsUpdate[i];
        }
        // Update jac
        jac_t jacUpdate;
        matMatMulReg( jac, jacOther, jacUpdate );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jac[i][j] = -jacUpdate[i][j];
            }
        }
    }

    __forceinline__ __device__ static void reduceEqsWarp(    // Reduction communicating within threads in a warp
        jac_t& jac, rhs_t& rhs,
        int po2, int lIdx
    ){
        // Initialise updates to 0
        rhs_t rhsUpdate; setToZeroRhs(rhsUpdate);
        jac_t jacUpdate; setToZeroJac(jacUpdate);
        // Compute updates
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            // MVM(J[t],rhs[t-1])
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                rhsUpdate[i] += jac[i][j] * __shfl_up_sync( 0xffffffff, rhs[j], po2 );
            }
            // MMM(J[t],J[t-1])
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                #pragma unroll
                for( int k = 0; k < N; ++k ){
                    jacUpdate[i][j] += jac[i][k] * __shfl_up_sync( 0xffffffff, jac[k][j], po2 );
                }
            }
        }
        // Only reduce if I'm fetching values from within the warp, otherwise leave untouched
        const bool reduce = ( lIdx >= po2 );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhs[i] -= rhsUpdate[i] * reduce;
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jac[i][j] = -jacUpdate[i][j] * reduce + jac[i][j] * (!reduce);
            }
        }
    }

    __forceinline__ __device__ static void bwdReduceEqsWarp(  // Same as above, but flipping sides (backward)
        jac_t& jac, rhs_t& rhs,
        int po2, int lIdx
    ){
        // Initialise updates to 0
        rhs_t rhsUpdate; setToZeroRhs(rhsUpdate);
        jac_t jacUpdate; setToZeroJac(jacUpdate);
        // Compute updates
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            // MVM(J[t],rhs[t+1])
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                rhsUpdate[i] += jac[i][j] * __shfl_down_sync( 0xffffffff, rhs[j], po2 );
            }
            // MMM(J[t],J[t+1])
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                #pragma unroll
                for( int k = 0; k < N; ++k ){
                    jacUpdate[i][j] += jac[i][k] * __shfl_down_sync( 0xffffffff, jac[k][j], po2 );
                }
            }
        }
        // Only reduce if I'm fetching values from within the warp, otherwise leave untouched
        const bool reduce = ( lIdx < ( threads_per_warp - po2 ) );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhs[i] -= rhsUpdate[i] * reduce;
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jac[i][j] = -jacUpdate[i][j] * reduce + jac[i][j] * (!reduce);
            }
        }
    }

    // In-warp communications ******************************************************************************************
    __forceinline__ __device__ static void sendSolNextInWarp(
        jac_t& jacPrev, const jac_t& jac,
        rhs_t& rhsPrev, const rhs_t& rhs,
        int lIdx, int po2=1
    ){
        const bool firstOfWarp = ( lIdx==0 );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhsPrev[i] = __shfl_up_sync( 0xffffffff, rhs[i], po2 ) * (!firstOfWarp) + rhsPrev[i] * firstOfWarp;
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jacPrev[i][j] = __shfl_up_sync( 0xffffffff, jac[i][j], po2 ) * (!firstOfWarp) + jacPrev[i][j] * firstOfWarp;
            }
        }
    }

    __forceinline__ __device__ static void sendSolNextInWarp(
        rhs_t& rhsPrev, const rhs_t& rhs,
        bool firstOfWarp
    ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhsPrev[i] = __shfl_up_sync( 0xffffffff, rhs[i], 1 ) * (!firstOfWarp) + rhsPrev[i] * firstOfWarp;
        }
    }

    __forceinline__ __device__ static void sendJacNextInWarp(
        jac_t& jacPrev, const jac_t& jac,
        bool firstOfWarp
    ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jacPrev[i][j] = __shfl_up_sync( 0xffffffff, jac[i][j], 1 ) * (!firstOfWarp) + jacPrev[i][j] * firstOfWarp;
            }
        }
    }

    // Same as above, but flipping sides (backward)
    __forceinline__ __device__ static void sendSolPrevInWarp(
        jac_t& jacNext, const jac_t& jac,
        rhs_t& rhsNext, const rhs_t& rhs,
        int lIdx, int po2=1
    ){
        const bool lastOfWarp = ( lIdx == threads_per_warp-1 );
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhsNext[i] = __shfl_down_sync( 0xffffffff, rhs[i], po2 ) * (!lastOfWarp) + rhsNext[i] * lastOfWarp;
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jacNext[i][j] = __shfl_down_sync( 0xffffffff, jac[i][j], po2 ) * (!lastOfWarp) + jacNext[i][j] * lastOfWarp;
            }
        }
    }

    __forceinline__ __device__ static void sendSolPrevInWarp(
        rhs_t& rhsNext, const rhs_t& rhs,
        bool lastOfWarp
    ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            rhsNext[i] = __shfl_down_sync( 0xffffffff, rhs[i], 1 ) * (!lastOfWarp) + rhsNext[i] * lastOfWarp;
        }
    }

    __forceinline__ __device__ static void sendJacPrevInWarp(
        jac_t& jacNext, const jac_t& jac,
        bool lastOfWarp
    ){
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                jacNext[i][j] = __shfl_down_sync(0xffffffff, jac[i][j], 1) * (!lastOfWarp) + jacNext[i][j] * lastOfWarp;
            }
        }
    }



    // Extended interface when deriving from this class ****************************************************************
    // SFINAE: Only available when inherited from!
    template< typename TupleType, bool Enable = !std::is_void_v< Derived > >
    __forceinline__ __device__ static std::enable_if_t< Enable, void > computeNegativeResidual(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const TupleType& lclVars,
        rhs_t (&res)[chunk_size]
    ){
        Derived::recurrenceStep( hm1, lclVars, h );
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            #pragma unroll
            for( int i = 0; i < N; ++i ){
                res[t][i] -= h[t][i];
            }
        }
        return;
    }


    template< typename TupleType, bool Enable = !std::is_void_v< Derived > >
    __forceinline__ __device__ static std::enable_if_t< Enable, void > assembleInitialGuess(
        rhs_t (&h)[chunk_size],
        const TupleType& lclVars
    ){
        rhs_t hm1[chunk_size];
        setToZeroRhsChunk( hm1 );
        Derived::recurrenceStep( hm1, lclVars, h );
        return;
    }


    template< typename TupleType, bool Enable = !std::is_void_v< Derived > >
    __forceinline__ __device__ static std::enable_if_t< Enable, void > computeJacobiansTranspose(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const TupleType& lclVars,
        jac_t (&jac)[chunk_size]
    ){
        Derived::computeJacobians( h, hm1, lclVars, jac );
        // Transpose!
        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            #pragma unroll
            for( int i = 0; i < N; ++i ){
                #pragma unroll
                for( int j = i+1; j < N; ++j ){
                    const scalar_t temp = jac[t][i][j];
                    jac[t][i][j] = jac[t][j][i];
                    jac[t][j][i] = temp;
                }
            }
        }
        return;
    }





private:

    // Auxiliary funcs for register-wise operations ********************************************************************
    // - MMM
    __forceinline__ __device__ static void matMatMulReg(
        const jac_t& A, const jac_t& B, jac_t& C
    ){
        setToZeroJac(C);
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                #pragma unroll
                for( int k = 0; k < N; ++k ){
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    // - MVM
    __forceinline__ __device__ static void matVecMulReg(
        const jac_t& A, const rhs_t& b, rhs_t& c
    ){
        setToZeroRhs(c);
        #pragma unroll
        for( int i = 0; i < N; ++i ){
            #pragma unroll
            for( int j = 0; j < N; ++j ){
                c[i] += A[i][j] * b[j];
            }
        }
    }

};


