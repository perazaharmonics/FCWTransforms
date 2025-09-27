/* 
1. The first part of the code is to initialize the matrix that will be used to store the values of the Ambiguity Function (AF) for each combination of tau and f.
2. The next part of the code is to calculate the AF for each combination of tau and f. This part is parallelized using OpenMP.
3. The final part of the code is to print the result for each combination of tau and f. This part is not parallelized.
4. The code then stops the timer and prints the duration of the calculation. */


#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono> 

using namespace std;
using namespace std::chrono;

typedef vector<double> Signal;
typedef vector<vector<double>> AFMatrix;
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Parallelize the 2-D AF surface loop nest over (m,n)
// Each thread owns distinct AF[m][n] cells (race-free)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
void ComputeAF_ParallelCollapse(AFMatrix& AF,const Signal& x,
                                const vector<double>& tau_values,
                                const vector<double>& f_values)
{
    int tau_length=tau_values.size(); // Number of propagation delay measurements
    int f_length=f_values.size();     // Number of observed Doppler Shifts
    // Start timer (optional and just for measurement)
    auto t0=std::chrono::system_clock::now();
    // Parallelize both loops by collapsing them into one iteration space
#pragma omp parallel for collapse(2)
    for(int m=0;m<tau_length;m++)
    {
      for(int n=0;n<f_length;n++)
      {
        double tau=tau_values[m];
        double f=f_values[n]; // kept for symmetry; use if needed
        double result=0.0;
        // Compute cell value (can call SIMD helper below)
        for(size_t t=0;t<x.size();t++)
        {
          double time_val=t+tau;
          if(time_val>=0&&time_val<x.size())
            result+=x[t]*x[(int)time_val];
        }
        AF[m][n]=result;
        // WARN: Avoid std::cout here inside parallel region; it serializes/garbles output.
      }
    }
    // Stop timer (optional)
    auto t1=std::chrono::system_clock::now();
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    std::cout<<"ComputeAF_ParallelCollapse took "<<ms<<" ms\n";
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// SIMD-vectorize the inner accumulation over t
// Uses a reduction to safely combine lane partial sums
// Hooks after ComputeAF_ParallelCollapse to accumulate && sum AF surface
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
double SimdCorrelateLaneSum(const Signal& x,double tau)
{
    double acc=0.0;
    // Tell the compiler to vectorize with a sum reduction on acc
    // Optionally add: simdlen(8) or safelen(8), aligned(x:64), linear(t:1)
#pragma omp simd reduction(+:acc)
    for(size_t t=0;t<x.size();t++)
    {
      double time_val=t+tau;
      if(time_val>=0&&time_val<x.size())
        acc+=x[t]*x[(int)time_val];
    }

    return acc;
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Compute the AF surface with 2-D parallelism over (m,n)
// and SIMD inside the inner correlation via SimdCorrelateLaneSum
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
AFMatrix computeAF(const Signal& x,
                   const vector<double>& tau_values,
                   const vector<double>& f_values)
{
    int tau_length=tau_values.size();   // number of propagation delay measurements
    int f_length=f_values.size();       // number of observed Doppler shifts
    AFMatrix AF(tau_length,vector<double>(f_length,0));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Start the timer and announce (serial)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto start=std::chrono::system_clock::now();
    std::time_t start_time=std::chrono::system_clock::to_time_t(start);
    char str[26];
    ctime_s(str,sizeof str,&start_time);
    std::cout<<"Starting the timer at: "<<str; // \n included by ctime_s buffer
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Parallelize the (m,n) grid; each thread writes AF[m][n]
    // Inner accumulation is SIMD via SimdCorrelateLaneSum
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
#pragma omp parallel for collapse(2)
    for(int m=0;m<tau_length;m++)
    {
      for(int n=0;n<f_length;n++)
      {
        double tau=tau_values[m];
        double f=f_values[n]; // kept for future Doppler use
        // SIMD-accelerated inner accumulation
        double result=SimdCorrelateLaneSum(x,tau);
        AF[m][n]=result;
        // Optional sparse progress print (avoid garble with a critical)
        if(m%10000==0 && n%10000==0)
        {
#pragma omp critical
          std::cout<<"For Time Delay "<<tau<<", Doppler Shift "<<f
            <<", AF = "<<result<<std::endl;
        }
      }
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Stop the timer and report duration (serial)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    auto stop=std::chrono::system_clock::now();
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    std::cout<<"Time taken by function: "<<ms<<" milliseconds"<<std::endl;
    return AF;   // Target locked in
}