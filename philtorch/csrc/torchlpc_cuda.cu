#include "stable_support.h"

#include "recur_checks.h"

#define torchlpc philtorch

#include "../../third_party/torchlpc/cuda/lpc.cu"

#undef torchlpc
