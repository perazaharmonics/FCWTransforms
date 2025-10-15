/*
* *
* * Filename: FCWTranforms.h
* *
* * Description:
* *  This is a really old file I made through my Master's degree where I just
* *  a bunch of transforms I picked up through my different studies during
* *  years. It contains, of course, the Fast Fourier Transforms and MANY
* *  permutations. It also contains the Wavelet Transforms, and the Discrete
* *  Cosine Transforms, among others. It also contains helper transform based
* *  algorithms.
* *
* * NOTE: 
* *  Maybe in the future I will add SIMD but this is so old that I will
* *  probalbly contain the spectral class and extend it to use SIMD
* *
* *  Author:
* *   JEP, J.Enrique Peraza
* *
* *
*/
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <complex>
#include <thread>
#include <future>
#include <chrono>
#include <stdexcept>
#include <optional>
#include "DSPWindows.h"

namespace sig::spectral
{
  using namespace std;
 
// Class template for spectral operations
template<typename T>
class WaveletOps
{
  public:
  // The wavelets we know.
  enum class WaveletType
  {
    Haar,                            // The Haar Wavelet type.
    Db1,                             // Daubechies wavelet type 1.
    Db6,                             // Daubechies wavelet type 6.
    Sym5,                            // Symlet type 5
    Sym8,                            // Symlet type 8.
    Coif5,                           // Coiflet type 5
    Morlet,                          // Morlet wavelet.
    MexicanHat,                      // Mexican Hat wavelet.
    Meyer,                           // Meyer wavelet.
    Gaussian                         // Gaussian wavelet. 
  };
  enum class ThresholdType
  {
    Hard,                           // Hard thresholding.
    Soft                            // Soft thresholding.
  };
  public:
    // Constructors
  explicit WaveletOps(          // Constructor
      WaveletType wt=WaveletType::Haar, // Our default wavelet.
      size_t levels=1,            // Decomposition level.
      double threshold=0.0f,      // Denoising threshold.
      ThresholdType tt=ThresholdType::Hard)// The denoising type.
   : wType{wt},tType{tt},levels{levels},threshold{threshold} {}
    // ---------------------------- //
    // Denoise: Apply multi-level DWT denoising and reconstruct signal.
    // ---------------------------- //
    std::vector<double> Denoise(const std::vector<double>& signal)
    {                               // --------- Denoise --------------- //
      // -------------------------- //
      // 1. Zero-Pad the signal to make sure the operation is possible.
      // -------------------------- //   
      auto padded=pad_to_pow2(signal);// Zero-pad the signal.
      // -------------------------- //
      // 2. Now forward DWT + threshold.
      // -------------------------- //
      auto coeffs=dwt_multilevel(padded,selectForward(),
        this->levels,this->threshold,tTypeToString());
      // -------------------------- //
      // 3. Now perform the inverse DWT.
      // ------------------------- //
      auto recon=idwt_multilevel(coeffs,selectInverse());
      // ------------------------- //
      // 4. Remove the zero padding from the signal.
      // ------------------------- //
      return remove_padding(recon,signal.size());
    }                              // ---------- Denoise ----------- 
    // ---------- SplitTransientTonal: wavelet-based decomposition into transient/tonal layers ---------- //
// We run a multi-level DWT. The sum of details across levels behaves like "transient/noisy"
// content (fast changes), while the final approximation behaves like "tonal/sustained".
// You can process each independently (e.g., saturate transients, chorus tonals) and mix back.
inline pair<vector<double>,vector<double>> SplitTransientTonal(
  const vector<double>& signal_in)
{
  vector<double> x=signal_in;
  auto coeffs=dwt_multilevel(x,selectForward(),this->levels,this->threshold,tTypeToString());
  // Reconstruct tonal from approximation only
  vector<pair<vector<double>,vector<double>>> tonal_coeffs=coeffs;
  for(auto& p:tonal_coeffs)p.second=vector<double>(p.second.size(),0.0);
  vector<double> tonal=idwt_multilevel(tonal_coeffs,selectInverse());
  // Transient layer is residual detail sum
  vector<double> trans(signal_in.size(),0.0);
  for(size_t i=0;i<signal_in.size();++i)trans[i]=signal_in[i]-tonal[i];
  return {trans,tonal};
}
public:
  inline vector<double> pad_to_pow2(const vector<double>& signal) 
  {
    size_t original_length = signal.size();
    size_t padded_length = static_cast<size_t>(next_power_of_2(original_length));
    vector<double> padded_signal(padded_length);

    copy(signal.begin(), signal.end(), padded_signal.begin());
    fill(padded_signal.begin() + original_length, padded_signal.end(), 0);

    return padded_signal;
  }                                                     
/// Remove padding back to the original length
  inline vector<double> remove_padding(const vector<double>& signal, size_t original_length) 
  {
    return vector<double>(signal.begin(), signal.begin() + original_length);
  }
  
// Normalization.
inline vector<double> normalize_minmax(const vector<double>& data) 
{
    double min_val = *min_element(data.begin(), data.end());
    double max_val = *max_element(data.begin(), data.end());
    vector<double> normalized_data(data.size());

    transform(data.begin(), data.end(), normalized_data.begin(),
        [min_val, max_val](double x) { return (x - min_val) / (max_val - min_val); });

    return normalized_data;
}

inline vector<double> normalize_zscore(const vector<double>& data) 
{
    double mean_val = accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double std_val = sqrt(sq_sum / data.size() - mean_val * mean_val);
    vector<double> normalized_data(data.size());

    transform(data.begin(), data.end(), normalized_data.begin(),
        [mean_val, std_val](double x) { return (x - mean_val) / std_val; });

    return normalized_data;
}

inline vector<double> awgn(const vector<double>& signal, double desired_snr_db) 
{
    double signal_power = accumulate(signal.begin(), signal.end(), 0.0,
        [](double sum, double val) { return sum + val * val; }) / signal.size();
    double desired_noise_power = signal_power / pow(10, desired_snr_db / 10);
    vector<double> noise(signal.size());

    generate(noise.begin(), noise.end(), [desired_noise_power]() {
        return sqrt(desired_noise_power) * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    });

    vector<double> noisy_signal(signal.size());
    transform(signal.begin(), signal.end(), noise.begin(), noisy_signal.begin(), plus<double>());

    return noisy_signal;
}
/// Multi-level discrete wavelet transform with hard/soft thresholding
inline vector<pair<vector<double>, vector<double>>> 
  dwt_multilevel (
    vector<double>& signal, 
    function<pair<vector<double>,
    vector<double>>(const vector<double>&)> wavelet_func, 
    size_t levels,
    double threshold,
    const string& threshold_type = "hard") 
    {

      size_t n = signal.size();
      size_t n_pad = static_cast<size_t>(next_power_of_2(n));
      if (n_pad != n)
        signal.resize(n_pad, 0.0);
      vector<pair<vector<double>, vector<double>>> coeffs;
      vector<double> current_signal = signal;
      for (size_t i = 0; i < levels; ++i) 
      {
         auto [approx, detail] = wavelet_func(current_signal);
        // Apply thresholding to the detail coefficients
        if (threshold_type == "hard") 
          detail = hard_threshold(detail, threshold);
        else if (threshold_type == "soft")
          detail = soft_threshold(detail, threshold);
        coeffs.emplace_back(approx, detail);
        current_signal = approx;

        if (current_signal.size() < 2)
          break;
    }
    return coeffs;
}                                                                                                                         
/// Inverse multi-level DWT
inline vector<double> 
idwt_multilevel(vector<pair<vector<double>, vector<double>>>& coeffs, function<vector<double>(const vector<double>&, const vector<double>&)> wavelet_func) 
{                                       // ~~~~~~~~~~~ idwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~
    vector<double> signal = coeffs[0].first;
    for (int i=coeffs.size()-1;i>=0;--i)// For each level in decomposition algorithm... 
    {                                   // Get the produced approx (LP response) and detail (HP response)...
      auto [approx,detail]=coeffs[i];   // The output of the wavelet multilevel Filter bank.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Recompose the original signal, now that it has been compressed
      // and maybe denoised...
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      signal=wavelet_func(approx,detail);// Perfect Reconstruction (PR) Filter Bank output.
    }                                   // Done reconstructing signal from approx and details.
    return signal;                      // Return Perfectly Reconstructed signal.
}                                       // ~~~~~~~~~~~ idwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~


/// Forward wavelet mother waveforms
inline pair<vector<double>, vector<double>> haar (
  const vector<double>& signal) const
{                                       // ~~~~~~~~~~~~~~~~~ Haar ~~~~~~~~~~~~~~~~~~~~~~~~
    const vector<double> h={ 1.0/sqrt(2.0),1.0/sqrt(2.0)};// The LP coeffs of the Haar wavelet PR Filter Bank
    const vector<double> g={1.0/sqrt(2.0),-1.0 / sqrt(2.0)};// The HP coeffs of the Haar wavelet PR Filter Bank
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Our integration limit to decompose the signal into two complementary parts
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    size_t n=signal.size()/2;           // Integration limit (see Wavelet equation).
    vector<double> approx(n),detail(n); // Where to store LP & HP output samples.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Perform the convolution for each pair of input samples
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (size_t i=0;i<n;++i)            // For half the length of the signal....
    {                                   // Convolute with the filter bank.
        approx[i] = h[0]*signal[2*i]+h[1]*signal[2*i+1];// LP
        detail[i] = g[0]*signal[2*i]+g[1]*signal[2*i+1];// HP
    }                                   // Done producing approx and detail coeffs.
    return make_pair(approx,detail);    // Return both produced signals.
}                                       // ~~~~~~~~~~~~~~~~~ Haar ~~~~~~~~~~~~~~~~~~~~~~~~
// Daubechies' Mother Wavelet Type 1
inline pair<vector<double>, vector<double>> db1(const vector<double>& signal) const
{                                        // ~~~~~~~~~~~~~~ Db1 ~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Daubechies' Mother Wavelet Type 1 filters H is LP and G is HP
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    const vector<double> h={(1.0+sqrt(3.0))/4.0,(3.0+sqrt(3.0))/4.0,(3.0-sqrt(3.0))/4.0,(1.0-sqrt(3.0))/4.0};
    const vector<double> g={h[3],-h[2],h[1],-h[0]};
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Our integration limit to decompose the signal into two complementary parts
    // We actually downsample in the Forward wavelet algorithm, and then
    // upsample in the Inverse wavelet algorithm.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    size_t n=signal.size()/2;           // Integration limit (see Wavelet equation).
    vector<double> approx(n),detail(n); // Resize our output buffers.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Perform the convolution for each pair of input samples
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //    
    for (size_t i=0;i<n;++i)            // For the integration limit.... 
    {                                   // Apply the Daubechies PR Type 1 filters
      approx[i]=0.0;                    // Init Lowpass accumulator to zero
      detail[i]=0.0;                    // Init Highpass accumulator to zero
      for (size_t k=0;k<h.size();++k)   // For the number of coefficients in the filters...
      {                                 // Wavelet decompose....
        size_t index=2*i+k-h.size()/2+1;// The index for the current sample
        if (index<signal.size())        // Is our index within the integration limit?
        {                               // Yes
          approx[i]+=h[k]*signal[index];// Convolute and accumulate with LP
          detail[i]+=g[k]*signal[index];// Convolute and accumulate with HP
        }                               // Done applying forward wavelet filters once
      }                                 // Done applying traversing through filter coefficients
    }                                   // Done applying forward wavelet algorithm
    return make_pair(approx, detail);   // Return both produced signals
}                                       // ~~~~~~~~~~~~~~ Db1 ~~~~~~~~~~~~~~~~~~~~~~~~~~
inline pair<vector<double>, vector<double>> db6(const vector<double>& signal) const
{
    const vector<double> h = {
        -0.001077301085308,
        0.0047772575109455,
        0.0005538422011614,
        -0.031582039318486,
        0.027522865530305,
        0.097501605587322,
        -0.129766867567262,
        -0.226264693965440,
        0.315250351709198,
        0.751133908021095,
        0.494623890398453,
        0.111540743350109
    };
    const vector<double> g = {
        h[11], -h[10], h[9], -h[8], h[7], -h[6], h[5], -h[4], h[3], -h[2], h[1], -h[0]
    };

    size_t n = signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i = 0; i < n; ++i) {
        approx[i] = 0;
        detail[i] = 0;
        for (size_t k = 0; k < h.size(); ++k) {
            size_t index = 2 * i + k - h.size() / 2 + 1;
            if (index < signal.size()) {
                approx[i] += h[k] * signal[index];
                detail[i] += g[k] * signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}                                     
inline pair<vector<double>, vector<double>> sym5(const vector<double>& signal) const
{
    const vector<double> h = {
        0.027333068345078, 0.029519490925774, -0.039134249302383,
        0.199397533977394, 0.723407690402421, 0.633978963458212,
        0.016602105764522, -0.175328089908450, -0.021101834024759,
        0.019538882735287
    };
    const vector<double> g = {
        h[9], -h[8], h[7], -h[6], h[5], -h[4], h[3], -h[2], h[1], -h[0]
    };

    size_t n = signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i = 0; i < n; ++i) {
        approx[i] = 0;
        detail[i] = 0;
        for (size_t k = 0; k < h.size(); ++k) {
            size_t index = 2 * i + k - h.size() / 2 + 1;
            if (index < signal.size()) {
                approx[i] += h[k] * signal[index];
                detail[i] += g[k] * signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}
                                    
inline pair<vector<double>, vector<double>> sym8(const vector<double>& signal) const
{
    const vector<double> h = {
        -0.003382415951359, -0.000542132331635, 0.031695087811492,
        0.007607487325284, -0.143294238350809, -0.061273359067908,
        0.481359651258372, 0.777185751700523, 0.364441894835331,
        -0.051945838107709, -0.027219029917056, 0.049137179673476,
        0.003808752013890, -0.014952258336792, -0.000302920514551,
        0.001889950332900
    };
    const vector<double> g = {
        h[15], -h[14], h[13], -h[12], h[11], -h[10], h[9], -h[8],
        h[7], -h[6], h[5], -h[4], h[3], -h[2], h[1], -h[0]
    };

    size_t n = signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i = 0; i < n; ++i) {
        approx[i] = 0;
        detail[i] = 0;
        for (size_t k = 0; k < h.size(); ++k) {
            size_t index = 2 * i + k - h.size() / 2 + 1;
            if (index < signal.size()) {
                approx[i] += h[k] * signal[index];
                detail[i] += g[k] * signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}

inline pair<vector<double>, vector<double>> coif5(const vector<double>& signal) const
{
    const vector<double> h = {
        -0.000720549445364, -0.001823208870703, 0.005611434819394,
        0.023680171946334, -0.059434418646456, -0.076488599078311,
        0.417005184421393, 0.812723635445542, 0.386110066821162,
        -0.067372554721963, -0.041464936781959, 0.016387336463522
    };
    const vector<double> g = {
        h[11], -h[10], h[9], -h[8], h[7], -h[6], h[5], -h[4],
        h[3], -h[2], h[1], -h[0]
    };

    size_t n = signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i = 0; i < n; ++i) {
        approx[i] = 0;
        detail[i] = 0;
        for (size_t k = 0; k < h.size(); ++k) {
            size_t index = 2 * i + k - h.size() / 2 + 1;
            if (index < signal.size()) {
                approx[i] += h[k] * signal[index];
                detail[i] += g[k] * signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}

// Inverse wavelet reconstruction
inline vector<double> inverse_haar(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv = { 0.7071067811865476, 0.7071067811865476 };
    const vector<double> g_inv = { -0.7071067811865476, 0.7071067811865476 };

    vector<double> reconstructed_signal;
    for (size_t i = 0; i < approx.size(); ++i) {
        reconstructed_signal.push_back(approx[i] * h_inv[0] + detail[i] * g_inv[0]);
        reconstructed_signal.push_back(approx[i] * h_inv[1] + detail[i] * g_inv[1]);
    }

    return reconstructed_signal;
}
inline vector<double> inverse_db1(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv = {(1.0 +sqrt(3.0))/4.0,(3.0+sqrt(3.0))/4.0,(3.0-sqrt(3.0))/4.0,(1.0-sqrt(3.0))/4.0};
    const vector<double> g_inv = { h_inv[3],-h_inv[2],h_inv[1],-h_inv[0] };

    vector<double> reconstructed_signal(2 * approx.size(), 0.0);
    for (size_t i = 0; i < approx.size(); ++i) 
    {
        for (size_t k = 0; k < h_inv.size(); ++k)
        {
          size_t index = (2*i+k-h_inv.size()/2+1);
          if (index < reconstructed_signal.size())
            reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k];
        }
    }
    return reconstructed_signal;
}
inline vector<double> inverse_db6(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv = {
        0.111540743350109, 0.494623890398453, 0.751133908021095,
        0.315250351709198, -0.226264693965440, -0.129766867567262,
        0.097501605587322, 0.027522865530305, -0.031582039318486,
        0.0005538422011614, 0.0047772575109455, -0.001077301085308
    };
    const vector<double> g_inv = {
        h_inv[11], -h_inv[10], h_inv[9], -h_inv[8], h_inv[7], -h_inv[6],
        h_inv[5], -h_inv[4], h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2 * approx.size(), 0.0);
    for (size_t i = 0; i < approx.size(); ++i) {
        for (size_t k = 0; k < h_inv.size(); ++k) {
            size_t index = (2 * i + k - h_inv.size() / 2 + 1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

inline vector<double> inverse_sym5(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv = {
        0.019538882735287, -0.021101834024759, -0.175328089908450,
        0.016602105764522, 0.633978963458212, 0.723407690402421,
        0.199397533977394, -0.039134249302383, 0.029519490925774,
        0.027333068345078
    };
    const vector<double> g_inv = {
        h_inv[9], -h_inv[8], h_inv[7], -h_inv[6], h_inv[5], -h_inv[4],
        h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2 * approx.size(), 0.0);
    for (size_t i = 0; i < approx.size(); ++i) {
        for (size_t k = 0; k < h_inv.size(); ++k) {
            size_t index = (2 * i + k - h_inv.size() / 2 + 1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

inline vector<double> inverse_sym8(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv = {
        0.001889950332900, -0.000302920514551, -0.014952258336792,
        0.003808752013890, 0.049137179673476, -0.027219029917056,
        -0.051945838107709, 0.364441894835331, 0.777185751700523,
        0.481359651258372, -0.061273359067908, -0.143294238350809,
        0.007607487325284, 0.031695087811492, -0.000542132331635,
        -0.003382415951359
    };
    const vector<double> g_inv = {
        h_inv[15], -h_inv[14], h_inv[13], -h_inv[12], h_inv[11], -h_inv[10],
        h_inv[9], -h_inv[8], h_inv[7], -h_inv[6], h_inv[5], -h_inv[4],
        h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2 * approx.size(), 0.0);
    for (size_t i = 0; i < approx.size(); ++i) {
        for (size_t k = 0; k < h_inv.size(); ++k) {
            size_t index = (2 * i + k - h_inv.size() / 2 + 1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

inline vector<double> inverse_coif5(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv = {
        0.016387336463522, -0.041464936781959, -0.067372554721963,
        0.386110066821162, 0.812723635445542, 0.417005184421393,
        -0.076488599078311, -0.059434418646456, 0.023680171946334,
        0.005611434819394, -0.001823208870703, -0.000720549445364
    };
    const vector<double> g_inv = {
        h_inv[11], -h_inv[10], h_inv[9], -h_inv[8], h_inv[7], -h_inv[6],
        h_inv[5], -h_inv[4], h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2 * approx.size(), 0.0);
    for (size_t i = 0; i < approx.size(); ++i) {
        for (size_t k = 0; k < h_inv.size(); ++k) {
            size_t index = (2 * i + k - h_inv.size() / 2 + 1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Continuous Wavelet Transforms (CWT)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// CWT multilevel with hard/soft thresholding
inline vector<vector<double>> cwt_multilevel (
  const vector<double>& signal, 
  const vector<double>& scales,
  function<vector<double>(const vector<double>&, double)> wavelet_func,
  double threshold,
  const string& threshold_type = "hard") 
{                                       // ~~~~~~~~~~~ cwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~
    size_t n=signal.size();             // Original signal length
    size_t n_pad=static_cast<size_t>(next_power_of_2(n));// Next power of 2
    vector<double> padded_signal=signal;// Copy original signal
    if (n_pad!=n)                       // Need to pad?
      padded_signal.resize(n_pad,0.0);  // Yes, zero-pad
    vector<vector<double>> coeffs;      // CWT coefficients for each scale
    for (const auto& scale:scales)      // For each scale 
    {                                   // Compute CWT coefficients
      auto cwt_coeffs=wavelet_func(padded_signal,scale);
      // Apply thresholding to the CWT coefficients
      if (threshold_type == "hard") 
        cwt_coeffs = hard_threshold(cwt_coeffs, threshold);
      else if (threshold_type == "soft")
        cwt_coeffs = soft_threshold(cwt_coeffs, threshold);
      coeffs.push_back(cwt_coeffs);
    }
    return coeffs;
}                                       // ~~~~~~~~~~~ cwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~

// ICWT multilevel reconstruction
inline vector<double> icwt_multilevel (
  const vector<vector<double>>& coeffs, // CWT coefficients for each scale
  function<vector<double>(const vector<double>&, double)> wavelet_func, // Wavelet function
  const vector<double>& scales)         // Corresponding scales
{                                       // ~~~~~~~~~~~ icwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~
    if (coeffs.size()!=scales.size())   // Mismatched sizes?
      throw invalid_argument("Coeffs and scales size mismatch");
    size_t n=coeffs[0].size();          // Length of each coeff vector
    vector<double> signal(n,0.0);       // Reconstructed signal
    for (size_t i=0;i<coeffs.size();++i)// For each scale
    {                                   // Reconstruct signal
      auto recon=wavelet_func(coeffs[i],scales[i]);
      transform(signal.begin(),signal.end(),recon.begin(),signal.begin(),plus<double>());
    }                                   // Done reconstructing signal
    return signal;                      // Return reconstructed signal
}                                       // ~~~~~~~~~~~ icwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~

// Scalogram: CWT magnitude squared
inline vector<vector<double>> Scalogram (
  const vector<vector<double>>& cwt_coeffs) // CWT coefficients
{                                       // ~~~~~~~~~~~ Scalogram ~~~~~~~~~~~~~~~~~~~~~~~~
    vector<vector<double>> scalogram;   // Scalogram output
    for (const auto& coeffs:cwt_coeffs) // For each scale's coefficients
    {                                   // Compute magnitude squared
      vector<double> mag_sq(coeffs.size());// Magnitude squared
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Square of each coefficient (magnitude squared)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      transform(coeffs.begin(),coeffs.end(),mag_sq.begin(),
        [](double x){return x*x;});
      scalogram.push_back(mag_sq);      // Add to scalogram
    }                                   // Done all scales
    return scalogram;                   // Return scalogram
}                                       // ~~~~~~~~~~~ Scalogram ~~~~~~~~~~~~~~~~~~~~~~~~

// Select the forward continuous wavelet function
inline function<vector<double>(const vector<double>&, double)> selectCWTForward (void) const
{
  return [this](const vector<double>& signal, double scale)
  {
    return this->cwt_forward(signal, scale);
  };
}

// --- Scalar mother-wavelet kernels for CWT (psi) --- //
inline double MorletPsi(double x, double w0) const
{
  // Standard real Morlet (unnormalized amplitude as in comments above)
  // psi(x) = exp(-x^2/2) * cos(w0*x)
  return std::exp(-0.5 * x * x) * std::cos(w0 * x);
}

inline double MexicanHatPsi(double x, double /*a*/) const
{
  // Ricker (Mexican hat): (1 - x^2) * exp(-x^2/2)
  const double x2 = x * x;
  return (1.0 - x2) * std::exp(-0.5 * x2);
}

inline double MeyerPsi(double x) const
{
  // Piecewise Meyer per comment (real part only)
  const double ax = std::fabs(x);
  if (ax < 1.0/3.0)
    return 1.0;
  if (ax <= 2.0/3.0)
    return std::sin(M_PI/2.0 * MeyersVx(3.0 * ax - 1.0)) * std::cos(M_PI * x);
  return 0.0;
}

inline double GaussianPsi(double x, double a) const
{
  // A Gaussian-derivative-like real wavelet variant from comment
  // psi(x) ~= (1/sqrt(a))*pi^(-1/4) * exp(-x^2/(2a^2)) * (cos(sqrt(2pi/a)*x) - exp(-a/2))
  if (a <= 0.0) a = 1.0;
  const double a2 = a * a;
  const double norm = 1.0 / (std::sqrt(a) * std::pow(M_PI, 0.25));
  return norm * std::exp(-(x * x) / (2.0 * a2)) * (std::cos(std::sqrt(2.0 * M_PI / a) * x) - std::exp(-a / 2.0));
}

// cwt_forward: Continuous Wavelet Transform (CWT) using the selected wavelet
inline vector<double> cwt_forward (
  const vector<double>& s,
  double scale,
  double w0=5.0,
  double t0=1.0,
  double a=1.0) const
{                                       // ~~~~~~~~~~~ cwt_forward ~~~~~~~~~~~~~~~~~~~~~~~~
    size_t n=s.size();             // Signal length
    vector<double> coeffs(n,0.0);       // CWT coefficients
    if (scale <= 0.0) scale = 1.0;     // Guard against invalid scale
    // Choose wavelet kernel psi(x) based on wType
    auto psi = [this, w0, t0, a](double x) -> double
    {
      switch(this->wType)
      {
        case WaveletType::Morlet:       return this->MorletPsi(x, w0);
        case WaveletType::MexicanHat:   return this->MexicanHatPsi(x, t0);
        case WaveletType::Meyer:        return this->MeyerPsi(x);
        case WaveletType::Gaussian:     return this->GaussianPsi(x, a);
        default:                        return 0.0; // Unsupported for CWT here
      }
    };
    // Compute CWT coefficients
    for (size_t i=0;i<n;++i)            // For each time point
    {                                   // Compute CWT coefficient
      double t=i;                       // Time index
      double sum=0.0;                   // Accumulator
      for (size_t j=0;j<n;++j)          // For each signal sample
      {                                 // Compute contribution
        double tau=j;                   // Sample index
        double wt=psi((t-tau)/scale);   // Wavelet value at scaled time
        sum+=s[j]*wt;                   // Accumulate contribution
      }                                 // Done signal samples
      coeffs[i]=sum/sqrt(scale);        // Normalize by sqrt(scale)
    }                                   // Done time points
    return coeffs;                      // Return CWT coefficients
}                                       // ~~~~~~~~~~~ cwt_forward ~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Morlet wavelet decomposition
// PSI(x) = e^-x/2*cos(5*x)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline pair<vector<double>,vector<double>> Morlet (
  const vector<double>& s,
  double w0=5.0)
{                                       // ~~~~~~~~~~~~~ Morlet ~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  double A=1.0/sqrt(2.0*M_PI);          // Normalization constant.
  double s_inv=1.0/w0;                  // Scale inverse.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with Morlet wavelet.
    double t=i-n/2;                     // Center time around zero.
    double morlet=A*exp(-t*t/(2*s_inv*s_inv))*cos(w0*t/s_inv);// Morlet wavelet
    approx[i]=s[i]*morlet;              // Lowpass output (approx)
    detail[i]=s[i]*(1.0-morlet);        // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~~~~ Morlet ~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Mexican hat wavelet decomposition
// PSI(x) = (1-x^2)*e^-x/2
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline pair<vector<double>,vector<double>> MexicanHat (
  const vector<double>& s,
  double a=1.0)
{                                       // ~~~~~~~~~~ MexicanHat ~~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  double A=2.0/(sqrt(3.0*a)*pow(M_PI,0.25));// Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with Mexican Hat wavelet.
    double t=i-n/2;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double mexhat=A*(1.0-t2/a2)*exp(-t2/(2*a2));// Mexican Hat wavelet
    approx[i]=s[i]*mexhat;              // Lowpass output (approx)
    detail[i]=s[i]*(1.0-mexhat);        // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~ MexicanHat ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Meyer wavelet decomposition
// PSI(x) = sin(pi/2*v(3|x|-1))*e^(j*pi*x) for 1/3<=|x|<=2/3
//        = 1 for |x|<1/3
//        = 0 for |x|>2/3
// v(x)   = 0 for x<0
//        = x^3(35/32 - 35/16*x + 21/16*x^2 - 5/8*x^3) for 0<=x<1
//        = 1 for x>=1
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline pair<vector<double>,vector<double>> Meyer (
  const vector<double>& s)
{                                       // ~~~~~~~~~~~~~ Meyer ~~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  for (size_t i=0;i<n;++i)              // For each input sample
  {                                     // Convolute with Meyer wavelet.
    double t=i-n/2;                     // Center time around zero.
    double t_abs=fabs(t);               // |t|
    double meyer=0.0;                   // Meyer wavelet
    if (t_abs<1.0/3.0)                  // |t|<1/3
      meyer=1.0;                        // Psi=1
    else if (t_abs>=1.0/3.0&&t_abs<=2.0/3.0)// 1/3<=|t|<=2/3
      meyer=sin(M_PI/2.0*MeyersVx(3.0*t_abs-1.0))*cos(M_PI*t);// Psi
    else                                // |t|>2/3
      meyer=0.0;                        // Psi=0
    approx[i]=s[i]*meyer;               // Lowpass output (approx)
    detail[i]=s[i]*(1.0-meyer);         // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~~~~ Meyer ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// v(x) function used in Meyer wavelet
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline double MeyersVx (
  double x) const                       // v(x) function
{                                       // ~~~~~~~~~~~~~ MeyersVx ~~~~~~~~~~~~~~~~~~ //
  if (x<0.0)                            // Outside support
    return 0.0;                         // v(x)=0
  else if (x>=0.0&&x<1.0)               // Within compact support grid?
    return x*x*x*(35.0/32.0-35.0/16.0*x+21.0/16.0*x*x-5.0/8.0*x*x*x);
  else                                  // Outside support
    return 1.0;                         // v(x)=1
}                                       // ~~~~~~~~~~~~~ MeyersVx ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Gaussian wavelet decomposition
// PSI(x) = (1/sqrt(a))*pi^(-1/4)*e^(-x^2/2a^2)*(e^(j*sqrt(2pi/a)x)-e^(-a/2))
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
 inline pair<vector<double>,vector<double>> GaussianWavelet (
  const vector<double>& s,              // Input signal
  double a=1.0)                         // Scale parameter
{                                       // ~~~~~~~~~~ GaussianWavelet ~~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  double A=1.0/(sqrt(a)*pow(M_PI,0.25)); // Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with Gaussian wavelet.
    double t=i-n/2;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double gauss=A*exp(-t2/(2*a2))*(cos(sqrt(2.0*M_PI/a)*t)-exp(-a/2.0));// Gaussian wavelet
    approx[i]=s[i]*gauss;               // Lowpass output (approx)
    detail[i]=s[i]*(1.0-gauss);         // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~ GaussianWavelet ~~~~~~~~~~~~~~~~~~ //
// Inverse Morlet wavelet reconstruction
inline vector<double> InverseMorlet (
  const vector<double>& approx,
  const vector<double>& detail,
  double w0=6.0)
{                                       // ~~~~~~~~~~ InverseMorlet ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  double A=1.0/sqrt(2.0*M_PI);          // Normalization constant.
  double s_inv=1.0/w0;                  // Scale inverse.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Morlet wavelet.
    double t=2*i-n;                     // Center time around zero.
    double morlet=A*exp(-t*t/(2*s_inv*s_inv))*cos(w0*t/s_inv);// Morlet wavelet
    reconstructed_signal[2*i]+=approx[i]*morlet;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-morlet);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~~~~ InverseMorlet ~~~~~~~~~~~~~~~~~~ //
// Inverse Mexican Hat wavelet reconstruction
inline vector<double> InverseMexicanHat (
  const vector<double>& approx,         // Low frequency coefficients
  const vector<double>& detail,         // High frequency coefficients 
  double a=1.0)                         // Scale parameter
{                                       // ~~~~~~~~ InverseMexicanHat ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  double A=2.0/(sqrt(3.0*a)*pow(M_PI,0.25));// Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Mexican Hat wavelet.
    double t=2*i-n;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double mexhat=A*(1.0-t2/a2)*exp(-t2/(2*a2));// Mexican Hat wavelet
    reconstructed_signal[2*i]+=approx[i]*mexhat;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-mexhat);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~~ InverseMexicanHat ~~~~~~~~~~~~~~~~~~ 
// Inverse Meyer wavelet Perfect Reconstruction Filter Bank
inline vector<double> InverseMeyer (
  const vector<double>& approx,         // Low frequency coefficients
  const vector<double>& detail)         // High frequency coefficients
{                                       // ~~~~~~~~~~ InverseMeyer ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Meyer wavelet.
    double t=2*i-n;                     // Center time around zero.
    double t_abs=fabs(t);               // |t|
    double meyer=0.0;                   // Meyer wavelet
    if (t_abs<1.0/3.0)                  // |t|<1/3
      meyer=1.0;                        // Psi=1
    else if (t_abs>=1.0/3.0&&t_abs<=2.0/3.0)// 1/3<=|t|<=2/3
      meyer=sin(M_PI/2.0*MeyersVx(3.0*t_abs-1.0))*cos(M_PI*t);// Psi
    else                                // |t|>2/3
      meyer=0.0;                        // Psi=0
    reconstructed_signal[2*i]+=approx[i]*meyer;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-meyer);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~~~~ InverseMeyer ~~~~~~~~~~~~~~~~~~ //

// Inverse Gaussian wavelet reconstruction
inline vector<double> InverseGaussianWavelet (
  const vector<double>& approx,         // Low frequency coefficients
  const vector<double>& detail,         // High frequency coefficients
  double a=1.0)                         // Scale parameter
{                                       // ~~~~~~~ InverseGaussianWavelet ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  double A=1.0/(sqrt(a)*pow(M_PI,0.25)); // Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Gaussian wavelet.
    double t=2*i-n;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double gauss=A*exp(-t2/(2*a2))*(cos(sqrt(2.0*M_PI/a)*t)-exp(-a/2.0));// Gaussian wavelet
    reconstructed_signal[2*i]+=approx[i]*gauss;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-gauss);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~ InverseGaussianWavelet ~~~~~~~~~~~~~~~~~~ //


inline vector<double> hard_threshold(const vector<double>& detail, double threshold) 
{
    vector<double> result(detail.size());
    transform(detail.begin(), detail.end(), result.begin(), [threshold](double coeff) {
        return abs(coeff) < threshold ? 0.0 : coeff;
    });
    return result;
}

inline vector<double> soft_threshold(const vector<double>& detail, double threshold) 
{
    vector<double> result(detail.size());
    transform(detail.begin(), detail.end(), result.begin(), [threshold](double coeff) {
        return signbit(coeff) ? -max(0.0, abs(coeff) - threshold) : max(0.0, abs(coeff) - threshold);
    });
    return result;
}  
  
private:
  WaveletType wType{WaveletType::Haar};       // The wavelet of choice.
  ThresholdType tType{ThresholdType::Hard};     // The type of thresholding.
  size_t levels{1};              // The wavelet decomposition level.
  double threshold{0.0f};        // The threshold of when to cancel.
// Method to choose the correct forward wavelet function.
inline std::function<std::pair<std::vector<double>, std::vector<double>>(const std::vector<double>&)> selectForward(void) const
  {                              // -------- selectForward --------
    switch (wType)               // Act according to the type.
    {                            //
      case WaveletType::Haar:  return [this](const std::vector<double>& v){ return this->haar(v); };
      case WaveletType::Db1:   return [this](const std::vector<double>& v){ return this->db1(v); };
      case WaveletType::Db6:   return [this](const std::vector<double>& v){ return this->db6(v); };
      case WaveletType::Sym5:  return [this](const std::vector<double>& v){ return this->sym5(v); };
      case WaveletType::Sym8:  return [this](const std::vector<double>& v){ return this->sym8(v); };
      case WaveletType::Coif5: return [this](const std::vector<double>& v){ return this->coif5(v); };
    }                           // Done acting according to wlet typ.
    return [this](const std::vector<double>& v){ return this->haar(v); };
  }                             // -------- selectForward --------
// Chooses the correct inverse wavelet reconstruction
  inline std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>
  selectInverse(void) const   // Select the correct reconstruct wave.
  {                           // -------- selectInverse --------
    switch(wType)             // Act according to the wave type
    {                         // Select the recon wavelet.
      case WaveletType::Haar:  return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_haar(a,d); };
      case WaveletType::Db1:   return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_db1(a,d); };
      case WaveletType::Db6:   return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_db6(a,d); };
      case WaveletType::Sym5:  return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_sym5(a,d); };
      case WaveletType::Sym8:  return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_sym8(a,d); };
      case WaveletType::Coif5: return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_coif5(a,d); };
    }                          // Done acting according to wtype.
    return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_haar(a,d); };
  }                            // -------- selectInverse --------
  // Convert enum to the string the denoiser expects.
  inline std::string tTypeToString(void) const
  {
    return (this->tType==ThresholdType::Hard?"hard":"soft");
  }
protected:
/// Pad a signal up to the next power of two
  inline double next_power_of_2(double x) 
  {
    return x == 0 ? 1 : pow(2, ceil(log2(x)));
  }
  
};
// ----------------------------------------------------------------------------
// The discrete Cosine Transform (DCT) part of FCWTransforms.h
// Discrete Cosine / Modified Discrete Cosine Transform. 
// Supports DCT-I, DCT-II, DCT-III, and DCT-IV (1-D, real-valued).
// Supports MDCT/IMDCT (Princen-Bradley sine window pair).
// Uses the existing SpectralOps<T>::FFTStride/IFFTStride for O(N*log(N))
// No heap allocations inside the hot path (caller passes working buffers).
// --------------------------------------------------------------------------- 
template<typename T=float>
class DCT
{
  public:
    enum class Type { I, II, III, IV,MDCT,IMDCT };

    // ----------- Forward 1-D transforms ----------------------------------- 
    static vector<T> Transform (
      const vector<T>& x,               // The input signal to transform.
      Type t,                           // The transform type.
      SpectralOps<T>& engine)           // Our spectral engine.
    {                                   // ----------- Transform -------------
      switch (t)                        // Act according to transform type...
      {                                 // 
        case Type::I:     return DCTI(x,engine);  // DCT-I
        case Type::II:    return DCTII(x,engine); // DCT-II
        case Type::III:   return DCTIII(x,engine);// DCT-III
        case Type::IV:    return DCTIV(x,engine); // DCT-IV
        case Type::MDCT:  {
      // Default to PB sine window for MDCT if none is provided
      sig::spectral::Window<T> W; 
      using WT = typename sig::spectral::Window<T>::WindowType;
      W.SetWindowType(WT::MLTSine, x.size());
      return MDCT(x, W, engine);  // MDCT (length N/2 coefficients)
    }
    case Type::IMDCT: {
      // Default to PB sine window for IMDCT (output length = 2*X.size())
      const size_t wlen = 2 * x.size();
      sig::spectral::Window<T> W;
      using WT = typename sig::spectral::Window<T>::WindowType;
      W.SetWindowType(WT::MLTSine, wlen);
      return IMDCT(x, W, engine); // IMDCT (returns 2N block for OLA)
    }
    default:          return DCTIV(x,engine); // Default to DCT-IV
      }                                 // Done dispatching transform          
    }                                   // ----------- Transform -------------
    static vector<T> Forward (
      const vector<T>& timeBlock,       // The time-domain block to transform.
      const Window<T>& analysisW,       // The analysis window to apply.
       SpectralOps<T>& engine)          // Our spectral engine.
    {                                   // ----------- Forward --------------- //
      return MDCT(timeBlock,analysisW,engine); // Use MDCT for forward transform.
    }                                   // ----------- Forward --------------- //
    // ----------- Inverse 1-D transforms -----------------------------------
    static vector<T> Inverse(
      const vector<T>& X,               // The frequency-domain block to transform.
      Type t,                           // The transform type.
    SpectralOps<T>& engine)             // Our spectral engine.
    {                                   // ----------- Inverse ----------------
      switch(t)                         // Dispatch according to type.
      {                                 //
        case Type::I:     return DCTI(X,engine,/*inverse=*/true); // DCT-I
        case Type::II:    return DCTIII(X,engine); // -1 dual
        case Type::III:   return DCTII(X,engine); // -1 dual
        case Type::IV:    return DCTIV(X,engine); // Self-inverse
        default:          return DCTIV(X,engine); // Default to DCT-II
      }                                 // Done dispatching inverse transform
    }                                   // ----------- Inverse ----------------
    static vector<T> Inverse(
      const vector<T>& coeffs,          // The frequency-domain coefficients to transform.
      const Window<T>& synthesisW,      // The synthesis window to apply.
      SpectralOps<T>& engine)           // Our spectral engine.      
    {                                   // ----------- Inverse ----------------
      return IMDCT(coeffs, synthesisW, engine); // Use IMDCT for inverse transform.
    }                                   // ----------- Inverse ----------------
    // ---------------- Discrete Cosine Transforms ----------------
    // All four classical DCTs are computed through a single length-2N complex FFT.
    // following the well-known even/odd embeddings (see Britanak & Rao, ch. 6).
    // We use the existing SpectralOps<T>::FFTStride/IFFTStride for O(N*log(N))
    // So no extra radix-2 code is duplicated here.
    // 
    static vector<T> DCTII(
      const vector<T>& x,               // The input signal to transform.
      SpectralOps<T>& engine,           // Our spectral engine.
      const bool inverse=false)         // If it is the inverse.           
    {                                   // ----------- DCT-II ----------------
      const size_t N=x.size();          // Get the size of the input signal.
      if (N==0) return {};              // Guard: empty
      vector<std::complex<T>> w(2*N);   // Create a complex vector of size 2*N.
      // Even-symmetrical extensions (x, reversed(x))
      for (size_t n=0;n<N;++n)          // For all samples.
      {                                 // Set the even-symmetrical extensions.
        w[n]={x[n],0};                  // Fill the first N points with x[n].
        w[2*N-1-n]={x[n],0};            // Fill the last N points with x[n].
      }                                 // Done symmetrical extensions.
      // Compute the FFT of the complex vector w.
      auto W=engine.FFTStride(w);       // Compute the FFT of the complex vector w.
      // Post-twiddle & packing.
      vector<T> X(N);                   // Create a vector of size N for the output.
      const T scale=inverse?T(2)/T(N):T(1); // Orthonormal pair (II <--> III).
      const T factor=T(M_PI)/(T(2)*T(N));   // The factor to apply to the output.
      for (size_t k=0;k<N;++k)          // For all frequency bins...
      {                                 // Apply the post-twiddle and scale/pack.
        std::complex<T> c=std::exp(std::complex<T>(0,-factor*T(k)));
        X[k]=T(2)*(W[k]*c).real()*scale;// Apply the post-twiddle and scale.
      }                                 // Done post-twiddle and packing.
      return X;                         // Return the DCT-II coefficients.
    }                                   // ----------- DCT-II ----------------
    static std::vector<T> DCTIII(
      const std::vector<T>& x,          // The input signal to transform.
      SpectralOps<T>& engine)           // Our spectral engine.
    {                                   // ----------- DCT-III ---------------
      // DCT-III is the inverse of DCT-II. Call DCTII with the inverse flag
      return DCTII(x,engine,true);      // Call DCT-II with the inverse flag.
    }                                   // ----------- DCT-III ---------------
    static vector<T> DCTI(
      const std::vector<T>& x,          // The input signal to transform.
      SpectralOps<T>& engine,           // Our spectral engine.
      const bool inverse=false)         // If it is the inverse.
    {                                   // ----------- DCT-I -----------------
      // Even-even extensions --> length-2(N-1) FFT.
  const size_t N=x.size();          // Get the size of the input signal.
      if (N<2) return x;                // If the input signal is too short, return it.
      vector<std::complex<T>> w(2*(N-1)); // Create a complex vector of size 2*(N-1).
      // Fill the complex vector with the even-even extensions.
      for (size_t n=0;n<N-1;++n)        // For all samples.
      {                                 // Set the even-even extensions.
        w[n]={x[n],0};                  // Fill the first N-1 points with x[n].
        w[2*(N-1)-1-n]={x[N-2-n],0};    // Fill the last N-1 points with x[N-2-n].
      }                                 // Done even-even extensions.
      w[N-1]={x.back(),0};              // Fill the middle point with x[N-1].
      auto W=engine.FFTStride(w);       // Compute the FFT of the complex vector w.
      vector<T> X(N);                   // Create a vector of size N for the output.
      const T scale=inverse?T(1)/T(N-1):T(1); // Orthonormal pair (I <--> IV).
      for (size_t k=0;k<N;++k)          // For all frequency bins...
        X[k]=W[k].real()*scale;         // Apply the post-twiddle and scale.
      return X;                         // Return the DCT-I coefficients.
    }                                   // ----------- DCT-I -----------------
    static vector<T> DCTIV(
      const vector<T>& x,               // The input signal to transform.
      SpectralOps<T>& engine)           // Our spectral engine.
    {                                   // ----------- DCT-IV ----------------
  const size_t N=x.size();          // Get the size of the input signal.
  if (N==0) return {};              // Guard: empty
  vector<std::complex<T>> w(2*N);   // Create a complex vector of size 2*N.
      // Fill the complex vector with the even-even extensions.
      for (size_t n=0;n<N;++n)          // For all samples...
      {
        w[n]={x[n],0};                  // Fill the first N points with x[n].
        w[2*N-1-n]={-x[n],0};           // Fill the last N points with -x[n].
      }                                 // Done symmetrical extensions.
      auto W=engine.FFTStride(w);       // Compute the FFT of the complex vector w.
      vector<T> X(N);                   // Create a vector of size N for the output.
      const T factor=T(M_PI)/(T(4)*T(N)); // The factor to apply to the output.
      for (size_t k=0;k<N;++k)          // For all frequency bins....
      {                                 
        std::complex<T> c=std::exp(std::complex<T>(0,-factor*(T(2)*T(k)+T(1))));
        X[k]=T(2)*(W[k].real()*c.real()-W[k].imag()*c.imag());// Apply the post-twiddle and scale.
      }                                 // Done post-twiddle and packing.
      return X;                         // Self-inverse (orthogonal, no extra scaling).
    }                                   // ----------- DCT-IV ----------------
  // --------------------- MDCT/IMDCT ---------------------
  // MDCT length is N (produces N/2 coefficients from a 2N-sample block).
  // IMDCT returns a 2N block that the caller overlap-adds by N samples.
  // -----------------------------------------------------------
  static vector<T> MDCT(
    const vector<T>& timeBlock,         // The time-domain block to transform.
    const Window<T>& win,               // The window to apply before MDCT.
    SpectralOps<T>& engine)             // Our spectral engine.
  {                                     // ------------ MDCT ---------------
    // Preconditions:
  const size_t n2=timeBlock.size();   // Get the size of the input block.
    if (n2%2)                           // If a remainder appears after division by 2....
      return vector<T>();               // Return an empty vector.
    const size_t N=n2/2;                // N is the number of MDCT coefficients.
    // -------------------------------- //
    // Windowing + pre-twiddle (+ phase shift n+0.5).
    // -------------------------------- //
    vector<std::complex<T>> xwin(n2);   // Create a complex vector of size n2.
    for (size_t n=0;n<n2;++n)           // Fpr all samples...
      xwin[n]={timeBlock[n]*win[n],0}; // Apply the window to the time block.
    // -------------------------------- //
    // Rearrange into even-odd groups for FFT of length 2N.
    // -------------------------------- //
    vector<std::complex<T>> v(n2);      //  Create a complex vector of size n2.
    for (size_t n=0;n<N;++n)            // For all samples....
    {                                   // Fill the complex vector with the even-odd groups.
      v[n]=xwin[n]+xwin[n2-1-n];        // Even part: x[n]+w[n2-1-n].
      v[n+N]=(xwin[n]-xwin[n2-1-n])*std::complex<T>{0,-1};     // Odd part: x[n]-w[n2-1-n].
    }                                   //
    auto V=engine.FFTStride(v);         // Compute the FFT of the complex vector v.
    // Post-twiddle: take real part, multiply by exp(-j(pi*k+0.5)/N)                                                
    vector<T> X(N);                    // Create a vector of size N for the output.
    const T factor=T(M_PI)/(T(2)*T(N)); // The factor to apply to the output.
    for (size_t k=0;k<N;++k)            // For all frequency bins...
    {                                   // Apply the post-twiddle and scale.
      std::complex<T> c=std::exp(std::complex<T>(0,-factor*(T(k)+T(0.5))));
      X[k]=(V[k]*c).real();             // Take the real part and apply the post-twiddle.
    }                                   // Done post-twiddle and packing.
    return X;                           // Return the MDCT coefficients.
  }                                     // ------------ MDCT ---------------
  static vector<T> IMDCT(
    const vector<T>& X,                 // The frequency-domain block to transform.
    const Window<T>& win,               // The window to apply after IMDCT.
    SpectralOps<T>& engine)             // Our spectral engine.
  {                                     // ------------ IMDCT ---------------
    const size_t N=X.size();            // Get the size of the input block.
    const size_t n2=N*2;                //
    // Pre-twiddle
    vector<std::complex<T>> V(n2);      // Create a complex vector of size n2.
    const T factor=T(M_PI)/(T(2)*T(N)); // The factor to apply to the output.
    for (size_t k=0;k<N;++k)            // For all frequency bins....
    {
      std::complex<T> c=std::exp(std::complex<T>(0,factor*(T(k)+T(0.5))));
      V[k]=X[k]*c;                      // Apply the pre-twiddle.
      V[k+N]=-X[k]*c;                   // odd symmetry: V[k+N]=-X[k]*c.
    }                                   // 
    // IFFT but actually FFT of size 2N, then scale/flip.
    auto v=engine.IFFTStride(V);        // Compute the IFFT of the complex vector V.
    // Post-twiddle + windowing.
    vector<T> y(n2);                     // Output signal.
    for (size_t n=0;n<n2;++n)           // For all samples...
      y[n]=T(2)*(v[(n+N/2)%n2].real())*win[n]; // Apply the post-twiddle and windowing.
    return y;           // Caller must perform 50% OLA with previous frame (block).
  }                                     // ------------ IMDCT ---------------
};  

// ============================================================================
// MDCT Beat Granulator (free function)
// Depends on: Window<T>, DCT<T>::MDCT/IMDCT and a SpectralOps<T>& engine.
// ============================================================================
// ===================== MDCT Beat Granulator (with Pitch & Stutter) =====================
template<typename T=float>
class MDCTBeatGranulator {
public:
  struct Config {
    double bpm        {120.0};   // musical tempo
    double stutterBeats {0.0};   // 0 = off; else duration in beats for a frozen grain
    double pitchRatio {1.0};     // 1.0=no shift; 2.0=+12 st; 0.5=-12 st
    size_t windowLen  {2048};    // MDCT operates on 2N samples; windowLen == 2N
    size_t hop        {0};       // if 0, defaults to 50% overlap (N)
  };

  MDCTBeatGranulator(const Config& cfg,
                     typename Window<T>::WindowType wtype = Window<T>::WindowType::MLTSine)
  : cfg_(cfg)
  {
    if(cfg_.windowLen % 2) cfg_.windowLen++;            // force even 2N
    N_  = cfg_.windowLen/2;                             // MDCT length
    hop_= (cfg_.hop==0) ? N_ : cfg_.hop;                // default 50% OLA
    W_.SetWindowType(wtype, cfg_.windowLen);            // PB-compatible sine by default
    lastFreezeStart_ = 0;
    framesPerStutter_ = 0; // computed at prepare()
  }

  // Call this once you know sample rate and before processing
  void prepare(double sampleRate) {
    fs_ = sampleRate;
    // how many *output* samples per beat:
    const double samplesPerBeat = (60.0 / cfg_.bpm) * fs_;
    // how many analysis hops fit in a stutter chunk:
    if (cfg_.stutterBeats > 0.0) {
      const double stutterSamples = cfg_.stutterBeats * samplesPerBeat;
      framesPerStutter_ = std::max<size_t>(1, static_cast<size_t>(std::round(stutterSamples / hop_)));
    } else {
      framesPerStutter_ = 0;
    }
    writePtr_ = 0;
    frozen_   = false;
    frozenX_.clear();
    xFrame_.assign(cfg_.windowLen, T(0));
    ola_.assign(cfg_.windowLen, T(0));
    ring_.clear();
    ring_.reserve(8); // tiny cache of recent frames? MDCT coeffs
  }

  // Process a block of audio. This consumes input with hop-sized increments and returns OLA?d output.
  // You can feed this in a pull model: push hop samples at a time and collect hop samples back.
  std::vector<T> processHop(const std::vector<T>& inHop,
                            SpectralOps<T>& engine)
  {
    if (inHop.size()!=hop_) throw std::runtime_error("MDCTBeatGranulator: inHop size must equal hop.");
    // Slide the analysis window buffer
    std::rotate(xFrame_.begin(), xFrame_.begin()+hop_, xFrame_.end());
    std::copy(inHop.begin(), inHop.end(), xFrame_.end()-hop_);

    // 1) MDCT
    auto X = DCT<T>::MDCT(xFrame_, W_, engine);           // length N_

    // 2) optionally enter/maintain stutter
    if (framesPerStutter_>0) {
      if (!frozen_) {
        // start a freeze chunk at interval
        if ((frameCount_ - lastFreezeStart_) >= framesPerStutter_) {
          lastFreezeStart_ = frameCount_;
          frozen_   = true;
          frozenX_  = X;      // capture the grain
          freezeIdx_= 0;
        }
      }
    }

    // 3) choose source spectrum (live or frozen)
    const std::vector<T>& src = (frozen_ ? frozenX_ : X);

    // 4) apply pitch shift in MDCT domain (bin resample)
    std::vector<T> Xp = (cfg_.pitchRatio==1.0) ? src : pitchShiftBins(src, cfg_.pitchRatio);

    // 5) IMDCT ? overlap-add
    auto y2N = DCT<T>::IMDCT(Xp, W_, engine);             // returns 2N samples
    // standard 50% OLA: emit last hop from the middle
    // keep an OLA buffer so windows sum to 1
    for (size_t n=0;n<cfg_.windowLen;++n) {
      ola_[n] += y2N[n];
    }
    std::vector<T> out(hop_);
    std::copy(ola_.begin()+N_/2, ola_.begin()+N_/2+hop_, out.begin());
    // slide down OLA buffer
    std::rotate(ola_.begin(), ola_.begin()+hop_, ola_.end());
    std::fill(ola_.end()-hop_, ola_.end(), T(0));

    // 6) manage stutter lifetime
    if (frozen_) {
      ++freezeIdx_;
      if (freezeIdx_ >= framesPerStutter_) {
        frozen_ = false; // release freeze, resume live updates
      }
    }

    // save a small history if you want to audition different frames
    if (ring_.size()<ring_.capacity()) ring_.push_back(X);
    else { ring_[ringHead_] = X; ringHead_ = (ringHead_+1)%ring_.capacity(); }

    ++frameCount_;
    return out;
  }

  // Runtime setters
  void setPitchRatio(double r){ cfg_.pitchRatio = std::max(0.01, r); }
  void setStutterBeats(double b) {
    cfg_.stutterBeats = std::max(0.0, b);
    if (fs_>0) prepare(fs_); // recompute framesPerStutter_
  }
  void setBPM(double bpm) {
    cfg_.bpm = std::max(1.0, bpm);
    if (fs_>0) prepare(fs_);
  }

  const Config& config() const { return cfg_; }

private:
  // Lightweight MDCT-bin resampler for pitch shifting.
  // Reindexes magnitudes in the real MDCT domain with linear interpolation.
  // (For higher quality, add phase-aware methods in STFT or PQMF banks.)
  std::vector<T> pitchShiftBins(const std::vector<T>& X, double ratio)
  {
    const size_t N = X.size();
    std::vector<T> Y(N, T(0));
    for (size_t k=0;k<N;++k) {
      double srcIdx = static_cast<double>(k)/ratio;
      if (srcIdx<0 || srcIdx>static_cast<double>(N-1)) continue;
      size_t i0 = static_cast<size_t>(std::floor(srcIdx));
      size_t i1 = std::min(N-1, i0+1);
      double t  = srcIdx - static_cast<double>(i0);
      Y[k] = static_cast<T>((1.0-t)*X[i0] + t*X[i1]);
    }
    return Y;
  }

private:
  Config cfg_;
  size_t N_{0};
  size_t hop_{0};
  double fs_{0.0};

  Window<T> W_;
  std::vector<T> xFrame_;      // rolling 2N analysis buffer
  std::vector<T> ola_;         // 2N synthesis OLA buffer

  // stutter state
  size_t framesPerStutter_{0};
  size_t lastFreezeStart_{0};
  bool   frozen_{false};
  size_t freezeIdx_{0};
  std::vector<T> frozenX_;

  // tiny ring of recent MDCT frames (optional, handy for UI scrubbing)
  std::vector<std::vector<T>> ring_;
  size_t ringHead_{0};

  // counters
  size_t writePtr_{0};
  size_t frameCount_{0};
};

// The Fourier part of FCWTransforms.h
template<typename T>
class SpectralOps
{
    using WindowType = typename Window<T>::WindowType; // Alias for WindowType
public:
    // Constructors


    // Accessors
  inline vector<T> GetSignal (void) const { return signal; }
    inline void SetSignal (const vector<complex<T>>& s) {signal=s;}
  inline int GetSamples (void) const { return length; }
    inline void SetSamples (const int N) {length=N;}
  inline double GetSampleRate (void) const { return sRate; }
    inline void SetSampleRate (const double fs) {sRate=fs;}
  inline vector<complex<T>> GetTwiddles (void) const { return twiddles; }
  inline void SetSubCarrier(const vector<complex<T>> &s) { subCarrier = s; }
  inline vector<complex<T>> GetSubCarrier (void) { return subCarrier; } 
    
// ------------------------------------ //
// Constructors and Destructors
// ------------------------------------ //

SpectralOps(void)
{
  this->length = 0; // Set the length to 0.
  this->sRate = 0.0; // Set the sample rate to 0.
  this->window = WindowType::Hanning; // Default window type is Hamming.
  this->windowSize = 1024; // Default window size is 1024.
  this->overlap = 0.5; // Default overlap is 50%.
  this->signal.clear(); // Clear the signal vector.
  this->twiddles.clear(); // Clear the twiddles vector.
  this->subCarrier.clear(); // Clear the subcarrier vector.

}

SpectralOps(const vector<T> &s, const WindowType &w, const int windowSize) : signal{s}, length{0}, sRate{0.0}, window{w}, windowSize{windowSize}, overlap{0.5}
{
    // Generate the appropriate window for the given window size
    this->window = w.SetWindowType(window, windowSize);
}


SpectralOps(const vector<T> &s, const WindowType &w, const int windowSize, const float overlap) : signal{s}, length{0}, sRate{0.0}, window{w}, windowSize{windowSize}, overlap{overlap}
{
    // Generate the appropriate window for the given window size
    this->window=w.SetWindowType(window, windowSize);
}


~SpectralOps(void)
{
    signal.clear();
    twiddles.clear();
}


// ======================== Utility Methods ================================= //
// Utility Methods to precompute operations needed for Spectral Manipulations.
// ========================================================================== //
// The twiddle factor is a little trick which tunes the odd indexed samples.
// we need this because we need to tune the instantenous frequency of the 
// frequency component in the odd indexed samples of that frequency bin to the right position. 
// It rotates the phase to tune.
inline vector<complex<T>> TwiddleFactor(int N)
{
  if (twiddles.size() != static_cast<size_t>(N / 2))    // Did we precompute N/2 twiddles before?
    {                                                    // No, so we..
    twiddles.resize(static_cast<size_t>(N / 2));     // Resize the twiddles factor vector.
    for (int i = 0; i < N / 2; ++i)                  //  loop for the N/2 points and
      twiddles[static_cast<size_t>(i)] = polar(1.0, -2 * M_PI * i / N); //  compute the twiddles factors.
    }
    return twiddles;
}
// Get the smallest power of 2 that is greater than or equal to N
// that can hold the input sequence for the Cooley-Tukey FFT,
// which splits the input sequence into even and odd halves.

inline int UpperLog2(const int N)
{
    for (int i = 0; i < 30; ++i) // For the first 30 powers of 2
    {                            // Compute the power of 2 as 2^i
      const int mask = 1 << i;   // Compute the value of 2^i
      if (mask >= N)             // If the power of 2 is >= N
        return i;                // Return the smallest power of 2 (i).
    }                            //
    return 30;                   // Else return 30 as the upper bound.
}


inline vector<int>ToInt(const vector<complex<T>> &s)
{
    vector<int> sInt(s.size());
    for (size_t i = 0; i < s.size(); ++i)
        sInt[i] = static_cast<int>(s[i].real());
    return sInt;
}


inline vector<double> ToReal(const vector<complex<T>> &s)
{
    vector<double> sReal(s.size());
    for (size_t i = 0; i < s.size(); ++i)
        sReal[i] = s[i].real();
    return sReal;
}
// Determine the amount of frequency bins to analyze per second of data.

inline vector<T> SetRBW(double rbw, double fs)
{
  const int wSiz=static_cast<int>(fs/rbw);
  // Window is assumed to have been defined by the caller before calling this
  // method.
  return GetWindow(window,wSiz);
}
// ---------- Modular arithmetic helpers for index-mapping FFTs (Rader/Good-Thomas) ---------- //
// Fast exponentiation modulo m: computes (a^e) mod m with O(log e) multiplications.
inline int ModPow(int a,int e,int m)const
{
  long long res=1,base=(a%m+m)%m;
  while(e>0){if(e&1)res=(res*base)%m;base=(base*base)%m;e>>=1;}
  return static_cast<int>(res);
}
// Extended Euclid inverse: returns a^{-1} mod m (assuming gcd(a,m)==1).
inline int ModInv(int a,int m)const
{
  long long t=0,newt=1,r=m,newr=a;
  while(newr!=0){long long q=r/newr;long long tmp=t;t=newt;newt=tmp-q*newt;tmp=r;r=newr;newr=tmp-q*newr;}
  if(r>1)throw std::invalid_argument{"ModInv: non-invertible"};
  if(t<0)t+=m;
  return static_cast<int>(t);
}
// Find a primitive root g modulo prime p (tiny search is ok for FFT sizes we?ll use).
inline int PrimitiveRootPrime(int p)const
{
  // Factor p-1 into its prime factors (trial division is fine here).
  int phi=p-1;vector<int> factors;
  int n=phi;
  for(int f=2;f*f<=n;++f){if(n%f==0){factors.push_back(f);while(n%f==0)n/=f;}}
  if(n>1)factors.push_back(n);
  for(int g=2;g<p;++g)
  {
    bool ok=true;
    for(int q:factors){if(ModPow(g,phi/q,p)==1){ok=false;break;}}
    if(ok)return g;
  }
  throw std::runtime_error{"PrimitiveRootPrime: not found"};
}
// ---------- Tiny structure helpers for size classification & CRT splits ---------- //
inline bool IsPowerOfTwo(size_t n)const{return n&&((n&(n-1))==0);}
inline bool IsPrime(int n)const{if(n<2)return false;for(int d=2;d*d<=n;++d)if(n%d==0)return false;return true;}
inline int GCD(int a,int b)const{while(b){int t=a%b;a=b;b=t;}return a;}

// Try to factor N into co-prime pair (n1,n2) with n1*n2==N and gcd(n1,n2)==1.
// Returns the first small factorization it finds (prefers small n1). If none, nullopt.
inline std::optional<std::pair<int,int>> FindCoprimeFactorization(int N)const
{
  for(int d=2;d*d<=N;++d)
    if(N%d==0){int n1=d,n2=N/d;if(GCD(n1,n2)==1)return std::make_pair(n1,n2);}
  return std::nullopt;
}
// ---------- NormalizeByN: scale a complex vector by 1/N (common IFFT normalizer) ---------- //
inline void NormalizeByN(vector<complex<T>>& v)const
{
  if(v.empty())return;
  const T invn=T(1)/static_cast<T>(v.size());
  for(auto& z:v)z*=invn;
}
// ---------- Chirpyness: types & tiny helper ---------- //
// The chirp types we currently support. Linear means constant angular acceleration,
// Exponential grows (or decays) multiplicatively, and Hyperbolic sweeps with
// 1/(t+c) behavior that compresses time at low frequencies (classic radar-like sweep).
enum class ChirpType{ Linear, Exponential, Hyperbolic };

// Clamp a value into [lo,hi] to honor an audio-band fence (f_limit).
template<typename U>
inline U Clamp(U x,U lo,U hi)const{return x<lo?lo:(x>hi?hi:x);}

// ---------- FFTShift: move DC to the middle (purely for visualization/symmetric slicing) ---------- //
// This helper permutes a length-N spectrum so that index 0 (DC) ends up in the center of the array.
// For even N: left half [0..N/2-1] goes to right, right half [N/2..N-1] goes to left.
// We keep this separate so callers can opt-in to "human-friendly" centered views when taking slices.
inline vector<complex<T>> FFTShift(const vector<complex<T>>& X)const
{
  const size_t N=X.size();
  if(N==0)return {};
  vector<complex<T>> Y(N);
  const size_t h=N/2;
  // Move upper half to the front, lower half to the back.
  for(size_t i=0;i<h;++i)Y[i]=X[i+h];
  for(size_t i=0;i<h;++i)Y[i+h]=X[i];
  // If N is odd (unlikely in our uses here), we could do a rotate by floor(N/2); we stick to even N zoom sizes.
  return Y;
}
// ==================== Stride Permutation FFTs ============================= //
// Reference:  https://github.com/AndaOuyang/FFT/blob/main/fft.cpp
// ========================================================================== //
// Forward FFT Butterfly operation for the Cooley-Tukey FFT algorithm.
/*
 * @param last: The previous stage of the FFT.
 *    Time domain signal iff first iteration of the FFT.
 *    Frequency domain signal iff IFFT.
 * @param curr: The temporary buffer for the FFT in this iteration.
 *   Frequency domain spectrum in the last iteration iff FFT
 *   Time domain signal in the last iteration iff IFFT.
 * @param twiddles: Vector of precomputed twiddles factors.
 * @param rot: The current stage of the FFT, iteration indicator. Starts at 0 for the first stage.
 * @param nBits: log2(N) where N is the length of the signal, total number of FFT stages.
 * Reference: https://github.com/AndaOuyang/FFT/blob/main/fft.cpp
 */

inline void ForwardButterfly(vector<T> &last, vector<T> &curr, const vector<T> &twiddles, const int rot, const int nBits)
{
  if (rot == nBits)                          // Are we at the last stage of the FFT?
    return;                                  // Yes, so stop recursion.
    // ------------------------------------- //
    // Set the butterfuly section size to 2^(rot+1).
    // Each section doubles the size of the previous butterfly section.
    // ------------------------------------- //
  const int sectSiz = 1 << (rot + 1);        // Size of the butterfly section.
    // ------------------------------------- //
    // Number of sections (butterfly groups) the signal is split into at this stage. (phase groups) 
    // Each section is a group of butterflies, and has their phase computation.
    // ------------------------------------- //
  const int numSect = last.size() / sectSiz; // Number of sections the signal is divided into.
  const int phases = numSect;                // Number of phases (sections) in the FFT
    // ------------------------------------- //
    // Iterate over each phase in the FFT
    // Where each phase represents a group of butterfly operation
    // ------------------------------------- //
  for (int i = 0; i < phases; ++i)           // For every phase in the FFT
  {                                          // Perform the butterfly operation.
    const int base = i * sectSiz;            // Base index for the current phase.
    // ------------------------------------- //
    // Process each butterfly group within the current section.
    // The butterfly group is a pair of even and odd indices.
    // ------------------------------------- //
    for (int j = 0; j < sectSiz / 2; ++j) // For every butterfly group in the structure.
    {
    // ------------------------------------- //
    // Compute the even and odd indices in the butterfly group.
    // These elements will be combined to form the next stage of the FFT.
    // ------------------------------------- //      
        const int evenNdx = base + j;        // Even index in the butterfly group.
        const int oddNdx = base + sectSiz / 2 + j;// Odd index in the butterfly group.
    // ------------------------------------- //
    // Multiply the odd element by the twiddles factor for this butterfly group.  
    // The twiddles factor is a complex number that rotates the odd index.
    // and introduces the phase shift needed for the FFT. 
    // ------------------------------------- //   
        last[oddNdx] *= twiddles[j * phases];// Multiply the odd index by the twiddles factor.
    // ------------------------------------- //
    // Combine the next stage of the FFT using the even and odd indices.
    // The even and odd indices are combined to form the next stage of the FFT.
    // ------------------------------------- //      
        curr[evenNdx] = last[evenNdx] + last[oddNdx]; // Compute the even index.
        curr[oddNdx] = last[evenNdx] - last[oddNdx];  // Compute the odd index.
      } // Done with all butterfly groups.
  } // Done with all phases.
    // ------------------------------------- //
    // Recursivle move to the next stage of the FFT.
    // Swap the current and last buffers for the next iteration
    // ------------------------------------- //  
  ForwardButterfly(curr, last, twiddles, rot + 1, nBits); // Recurse to the next stage.
}
template<typename U>
inline void ForwardButterfly(
  vector<std::complex<U>>& last,
  vector<std::complex<U>>& curr,
  const vector<std::complex<U>>& twiddles,
  int rot,int nBits)
{
  if (rot==nBits) return;
  const int sect=1<<(rot+1);            // Size of the butterfly section.
  const int phases=last.size()/sect;    // Number of sections the signal is divided into.
  for (int i=0;i<phases;++i)            // For every phase in the FFT
  {                                      // Perform the butterfly operation.
    const int base=i*sect;               // Base index for the current phase.
    for (int j=0;j<sect/2;++j)           // For every butterfly group in the structure.
    {
      const int evenNdx=base+j;          // Even index in the butterfly group.
      const int oddNdx=base+sect/2+j;    // Odd index in the butterfly group.
      last[oddNdx]*=twiddles[j*phases];  // Multiply the odd index by the twiddles factor.
      curr[evenNdx]=last[evenNdx]+last[oddNdx]; // Compute the even index.
      curr[oddNdx]=last[evenNdx]-last[oddNdx];  // Compute the odd index.
    }                                    // Done with all butterfly groups.
  }                                      // Done with all phases.
  ForwardButterfly(curr, last, twiddles, rot + 1, nBits); // Recurse to the next stage.
}
// Bit reversal permutation for the Cooley-Tukey FFT algorithm.

inline void  BitReversal(vector<T> &s, const int nBits)
{
    // -------------------------------- //
    // Base Case: If the input size is <=2, no permutation necessary
    // For very small signals, bit reversal is not needed.
    // -------------------------------- //
  if (s.size()<=2)                      // Only two or less samples?
    return;                             // Yes, so no need to reverse bits.
    // -------------------------------- //
    // Special Case: If the input is exactly 4 samples, swap the middle
    // two elements. Handle the 2-bit case directly.
    // -------------------------------- //
  if (s.size()==4)                      // Is the signal exactly 4 samples?
  {                                     // Yes, so swap the middle two elements.
    swap(s[1], s[2]);                   // Swap the middle two elements.
    return;                             // Done with the bit reversal.
  }
    // -------------------------------- //
    // General Case: For signals larger than 4 samples, perform bit reversal.
    // Initialize a vector to hold bit-reversed indices and compute the bit
    // reversed indices for the FFT.
    // -------------------------------- //
  vector<int> revNdx(s.size());         // Vector to hold bit-reversed indices.
    // -------------------------------- //
    // Manually set the first 4 indices' bit-reversed values.
    // These are the known bit reversed values for the 2-bit case.
    // -------------------------------- //
  revNdx[0]=0;                          // Bit-reversed index for 0 is 0.
  revNdx[1]=1<<(nBits-1);               // == 100...0 in binary == 2^(nBits-1).
  revNdx[2]=1<<(nBits-2);               // == 010...0 in binary == 2^(nBits-2).
  revNdx[3]=revNdx[1]+revNdx[2];        // == 110...0 in binary == 2^(nBits-1) + 2^(nBits-2).
    // -------------------------------- //
    // Loop through to  compute the rest of the bit-reversed indices.
    // the bit-reversed index is the reverse of the binary representation of the index.
    // -------------------------------- //
    // Theorem: For all nk=2^k-1 where k<= nBits, 
    // revNdx[nk]=revNdx[n(k-1)]+2^(nBits-k)
    // revNdx[nk-i]=revNdx[nk]-revNdx[i]
    // -------------------------------- //
  for (int k=3; k<=nBits;++k)           // For all remaining bits in the signal.
  {
    const int nk=(1<<k)-1;              // Compute nk=2^k-1.
    const int nkmin1=(1<<(k-1))-1;      // Compute n(k-1)=2^(k-1)-1.
    // -------------------------------- //
    // Derive the bit-reversed index for nk using the bit reversal of n(k-1).
    // The bit-reversed index for nk is the bit-reversed index for n(k-1) plus 2^(nBits-k).
    // -------------------------------- //
    revNdx[nk]=revNdx[nkmin1]+(1<<(nBits-k)); // Compute revNdx[nk].
    // -------------------------------- //
    // Loop to compute the remaining bit reversed indices.
    // Compute for the range nk -i using nk and previously computed values.
    // -------------------------------- //
    for (int i=1; i<=nkmin1;++i)        // For the range nk-i.
      revNdx[nk-i]=revNdx[nk]-revNdx[i]; // Compute revNdx[nk-i].
  }
    // -------------------------------- //
    // Permute the signal using the bit-reversed indices.
    // Swap elements if the bit-reversed index is greater than the current index.
    //--------------------------------- //
  for (size_t i=0; i<s.size();++i)      // For all elements in the signal.
    if (static_cast<int>(i)<revNdx[i])  // If the index is less than the bit-reversed index.
      swap(s[i], s[static_cast<size_t>(revNdx[i])]); // Swap the elements.             
}                                       // End of the function.
// Overloaded for complex vectors.
inline void  BitReversal(vector<std::complex<T>> &s, const int nBits)
{
    // -------------------------------- //
    // Base Case: If the input size is <=2, no permutation necessary
    // For very small signals, bit reversal is not needed.
    // -------------------------------- //
  if (s.size()<=2)                      // Only two or less samples?
    return;                             // Yes, so no need to reverse bits.
    // -------------------------------- //
    // Special Case: If the input is exactly 4 samples, swap the middle
    // two elements. Handle the 2-bit case directly.
    // -------------------------------- //
  if (s.size()==4)                      // Is the signal exactly 4 samples?
  {                                     // Yes, so swap the middle two elements.
    swap(s[1], s[2]);                   // Swap the middle two elements.
    return;                             // Done with the bit reversal.
  }
    // -------------------------------- //
    // General Case: For signals larger than 4 samples, perform bit reversal.
    // Initialize a vector to hold bit-reversed indices and compute the bit
    // reversed indices for the FFT.
    // -------------------------------- //
  vector<int> revNdx(s.size());         // Vector to hold bit-reversed indices.
    // -------------------------------- //
    // Manually set the first 4 indices' bit-reversed values.
    // These are the known bit reversed values for the 2-bit case.
    // -------------------------------- //
  revNdx[0]=0;                          // Bit-reversed index for 0 is 0.
  revNdx[1]=1<<(nBits-1);               // == 100...0 in binary == 2^(nBits-1).
  revNdx[2]=1<<(nBits-2);               // == 010...0 in binary == 2^(nBits-2).
  revNdx[3]=revNdx[1]+revNdx[2];        // == 110...0 in binary == 2^(nBits-1) + 2^(nBits-2).
    // -------------------------------- //
    // Loop through to  compute the rest of the bit-reversed indices.
    // the bit-reversed index is the reverse of the binary representation of the index.
    // -------------------------------- //
    // Theorem: For all nk=2^k-1 where k<= nBits, 
    // revNdx[nk]=revNdx[n(k-1)]+2^(nBits-k)
    // revNdx[nk-i]=revNdx[nk]-revNdx[i]
    // -------------------------------- //
  for (int k=3; k<=nBits;++k)           // For all remaining bits in the signal.
  {
    const int nk=(1<<k)-1;              // Compute nk=2^k-1.
    const int nkmin1=(1<<(k-1))-1;      // Compute n(k-1)=2^(k-1)-1.
    // -------------------------------- //
    // Derive the bit-reversed index for nk using the bit reversal of n(k-1).
    // The bit-reversed index for nk is the bit-reversed index for n(k-1) plus 2^(nBits-k).
    // -------------------------------- //
    revNdx[nk]=revNdx[nkmin1]+(1<<(nBits-k)); // Compute revNdx[nk].
    // -------------------------------- //
    // Loop to compute the remaining bit reversed indices.
    // Compute for the range nk -i using nk and previously computed values.
    // -------------------------------- //
    for (int i=1; i<=nkmin1;++i)        // For the range nk-i.
      revNdx[nk-i]=revNdx[nk]-revNdx[i]; // Compute revNdx[nk-i].
  }
    // -------------------------------- //
    // Permute the signal using the bit-reversed indices.
    // Swap elements if the bit-reversed index is greater than the current index.
    //--------------------------------- //
    for (size_t i=0; i<s.size();++i)      // For all elements in the signal.
      if (static_cast<int>(i)<revNdx[i])  // If the index is less than the bit-reversed index.
        swap(s[i], s[static_cast<size_t>(revNdx[i])]); // Swap the elements.             
}                                       // End of the function.
    // ------------------------------------------------------------------------
    // LevinsonDurbin: Given autocorrelation r[0..p], solves for AR coefficients
    //   r[0] a[0] + r[1] a[1] + ? + r[p] a[p] = 0,    (Toeplitz system)
    //   returns (a[1..p], s²) where s² is the final prediction error.
    //   ?order? = p.  We assume r.size() >= p+1.
    // ------------------------------------------------------------------------
    
    inline std::pair<std::vector<T>, T>
    LevinsonDurbin(const std::vector<T>& r, int order) const
    {
        // r: autocorrelation, r[0] ? r[order]
        // order: AR order (p)
        if ((int)r.size() < order+1) {
            throw std::invalid_argument{"LevinsonDurbin: need r.size() >= order+1"};
        }
        std::vector<T> a(order+1, T{0}); // a[0]..a[p], we keep a[0]=1 internally
        std::vector<T> e(order+1, T{0}); // prediction error at each stage
        a[0] = T{1};
        e[0] = r[0];
        if (std::abs(e[0]) < std::numeric_limits<T>::epsilon()) {
            // All-zero autocorrelation ? trivial
            return { std::vector<T>(order, T{0}), T{0} };
        }

        for (int m = 1; m <= order; ++m) {
            // Compute reflection coefficient ?_m
            T num = r[m];                  // numerator = r[m] + sum_{i=1..m-1} a[i]·r[m-i]
            for (int i = 1; i < m; ++i) {
                num += a[i] * r[m - i];
            }
            T kappa = - num / e[m-1];

            // Update a[1..m]:
            std::vector<T> a_prev(m+1);
            for (int i = 0; i <= m; ++i) a_prev[i] = a[i];
            a[m] = kappa;
            for (int i = 1; i < m; ++i) {
                a[i] = a_prev[i] + kappa * a_prev[m - i];
            }

            // Update prediction error
            e[m] = e[m-1] * ( T{1} - kappa * kappa );
            if (std::abs(e[m]) < T{0}) {
                e[m] = T{0};
            }
        }

        // Return only a[1..p] (drop a[0]=1) and final error e[p]
        std::vector<T> arCoeffs(order);
        for (int i = 1; i <= order; ++i) {
            arCoeffs[i-1] = a[i];
        }
        return { arCoeffs, e[order] };
    }

    // ------------------------------------------------------------------------
    // AR_PSD: Given autocorrelation r[0..p], compute the ?all-pole? PSD estimate
    //    at fftsize uniformly spaced frequencies [0, 2p).  We solve AR(p) via
    //    Levinson-Durbin, then evaluate
    //      H(w) = s² / |1 + a[1] e^{-jw} + ? + a[p] e^{-j p w} |²
    //    at Nfft points, returning a vector<complex<T>> of length Nfft
    //    (you can take real(H) or abs(H)² as your PSD). 
    // ------------------------------------------------------------------------
    
    inline std::vector<std::complex<T>>
    AR_PSD(const std::vector<T>& r, int order, int fftsize) const
    {
        if (order < 1 || (int)r.size() < order+1) {
            throw std::invalid_argument{"AR_PSD: order must be =1 and r.size() = order+1"};
        }
        // 1) run Levinson-Durbin on r[0..order]
        auto [a, sigma2] = LevinsonDurbin(r, order);
        // a = vector length p, contains a[1],?a[p], and sigma2 = error at final stage

        // 2) build PSD at fftsize freq bins
        std::vector<std::complex<T>> psd(fftsize);
        const T normFactor = T{2} * M_PI / static_cast<T>(fftsize);
        for (int k = 0; k < fftsize; ++k) {
            T omega = normFactor * static_cast<T>(k); 
            // Evaluate denominator D(w) = 1 + w_{m=1..p} a[m-1] e^{-j m w}
            std::complex<T> denom = T{1};
            for (int m = 1; m <= order; ++m) {
                denom += a[m-1] * std::exp(std::complex<T>(T{0}, -omega * static_cast<T>(m)));
            }
            // PSD(w_k) = s² / |D(w)|²
            std::complex<T> H = std::complex<T>(sigma2) / (denom * std::conj(denom));
            psd[k] = H;
        }
        return psd;
    }


// ======================== Stride FFTs ===================================== //
// Stride FFTs are a special case of the FFT that uses a stride to compute the FFT
// of a signal. This is useful for signals that are not a power of 2 in length.
// The FFTStride method computes the FFT of a signal using the Cooley-Tukey algorithm
// with a stride. The IFFTStride method computes the IFFT of a signal using the
// FFTStride method with the conjugate of the input signal.
// ========================================================================== //
// FFTStrideEig computes the FFT of a signal and returns the spectrum and the
// eigenvectors of the FFT matrix which can be used for spectral analysis
// to obtain phase information.
inline std::pair<vector<complex<T>>,vector<vector<complex<T>>>> FFTStrideEig(const vector<complex<T>> &s)
{
  if (s.empty())                        // Is the input signal empty?
    return {vector<complex<T>>(), vector<vector<complex<T>>>()}; // Yes, so return empty vectors.
  // ---------------------------------- //
  // Calculate the number of bits needed for the FFT rounded to the
  // nearest upper power of 2. This is the number of stages in the FFT butterfly.
  // ---------------------------------- //
  const int NBits=UpperLog2(static_cast<int>(s.size())); // Get the number of bits for the FFT.
  const int N=1<<NBits;                 // Get the FFT length as a power of 2.
  // ---------------------------------- //
  // Create temporary buffers for the FFT.
  // The last buffer holds the previous stage of the FFT.
  // The current buffer holds the current stage of the FFT.
  // ---------------------------------- //
  vector<complex<T>> last(N), curr(N);  // Temporary buffers for the FFT.
  // ---------------------------------- //
  // Copy the input signal to the last buffer, and zero-pad if necessary.
  // ---------------------------------- //
  copy(s.begin(), s.end(), last.begin()); // Copy the input signal to the last buffer.
  // Zero-pad the last buffer to the FFT length.
  if (s.size() < N)                     // Is the input signal smaller than the FFT length?
    fill(last.begin() + s.size(), last.end(), complex<T>(0)); // Yes, so zero-pad the last buffer.
  // ---------------------------------- //
  // Perform the bit reversal permutation on the input signal.
  // This reorders the input signal to prepare for the Cooley-Tukey FFT.
  // ---------------------------------- //
  BitReversal(last, NBits);   // Perform bit reversal permutation.
  // ---------------------------------- //
  // Perform the FFT butterfly operation for the Cooley-Tukey FFT.
  // This computes the FFT in-place using the last and current buffers.
  // This is where the Cooley-Tukey FFT algorithm takes place.
  // ---------------------------------- //
  ForwardButterfly(last, curr, twiddles, 0, NBits); // Perform the FFT butterfly.
  // ---------------------------------- //
  // Here we compute the Fourier matrix and index the eigenvectors.
  // Return the computed FFT spectrum and the eigenvectors of the FFT matrix.
  // The eigenvectors are the Fourier basis vectors, which are the twiddles factors.
  // ---------------------------------- //
  vector<vector<complex<T>>> eigvecs(N, vector<complex<T>>(N)); // Create a matrix for the eigenvectors.
  const T invsqrt=static_cast<T>(1)/sqrt(static_cast<T>(N)); // Inverse square root of N for normalization.
  for (int ell=0;ell<N;ell++)           // For each row...
  {                                     // Compute the eigenvector for the row.
  // The row index ell corresponds to the frequency bin in the FFT.
      for (int k=0;k<N;k++)             // For each col in the eigenvector matrix...
    {                                   // Compute the Fourier matrix.
      long double angle=-2.0L*M_PI*(static_cast<long double>(ell))*(static_cast<long double>(k))/(static_cast<long double>(N));
      eigvecs[ell][k]=complex<T>(std::cos(angle),std::sin(angle))*invsqrt; // Compute the k-th eigenvector.
    }                                   // End of the loop.
  }                                     // End of the loop.
  return {curr, eigvecs};               // Return the computed FFT spectrum and the eigenvectors.
}


inline vector<complex<T>> FFTStride (const vector<complex<T>> &s)
{
    // ---------------------------------- //
    // Base Case: If the input is empty, return an empty vector.
    // ---------------------------------- //
    if (s.empty())                        // Is the input signal empty?
        return vector<complex<T>>();      // Yes, so return an empty vector.
    // ---------------------------------- //
    // Calculate the number of bits needed for the FFT rounded to the 
    // nearest upper power of 2. This is the number of stages in the FFT.
    // ---------------------------------- //
    const int nBits=UpperLog2(s.size());  // Get the number of bits for the FFT.
    // ---------------------------------- //
    // Calculate the FFT length as 2^nBits.
    // This is the length of the FFT signal.
    // ---------------------------------- //
    const int N=1<<nBits;                 // Get the FFT length as a power of 2.
    // ---------------------------------- //
    // Precompute the twiddles factors for the FFT.
    // The twiddles factors are used to rotate the signal in the FFT.
    // ---------------------------------- //
    const vector<complex<T>> twiddles=TwiddleFactor(N); // Phase-frequency vector.
    // ---------------------------------- //
    // Create temporary buffers for the FFT.
    // The last buffer holds the previous stage of the FFT.
    // The current buffer holds the current stage of the FFT.
    // ---------------------------------- //
    vector<complex<T>> last(N), curr(N);  // Temporary buffers for the FFT.
    // ---------------------------------- //
    // Copy the input signal to the last buffer, and zero-pad if necessary.
    // ---------------------------------- //
    copy(s.begin(), s.end(), last.begin()); // Copy the input signal to the last buffer.
    // ---------------------------------- //
    // Perform the bit reversal permutation on the input signal.
    // This reorders the input signal to prepare for the Cooley-Tukey FFT.
    // ---------------------------------- //
    BitReversal(last, nBits);   // Perform bit reversal permutation.
    // ---------------------------------- //
    // Perform the FFT butterfly operation for the Cooley-Tukey FFT.
    // This computes the FFT in-place using the last and current buffers.
    // This is where the Cooley-Tukey FFT algorithm takes place.
    // ---------------------------------- //
    ForwardButterfly(last, curr, twiddles, 0, nBits); // Perform the FFT butterfly.
    // ---------------------------------- //
    // Return the computed FFT spectrum.
    // ---------------------------------- //
    if (nBits %2 == 1)                    // Is the number of bits odd?
        return curr;                      // Yes, so return the current buffer.
    return last;                          // No, so return the last buffer.
}

// Convenience overload: real-valued FFT using complex FFTStride
inline vector<complex<T>> FFTStride (const vector<T> &s)
{
  vector<complex<T>> xc(s.size());
  for (size_t i=0;i<s.size();++i) xc[i] = complex<T>(s[i], T(0));
  return FFTStride(xc);
}
// The IFFT can be computed using the FFT with flipped order of the 
// frequency bins. That is, the complex conjugate of the input signal.
//   and thus the twiddles factors.
// So we just flip the frequency spectrum an normalize by 1/N.
// ------------------------------------------------------------
// Theorem: Let x[n] denote a time-domain signal and X[k] denote a frequency
// domain signal,then: 
// x[n]=(1/N) * SUM[k=0 to N-1] * {X[k] * exp(j*(2*pi/N)*k*n)} == IFFT(X[k]) 
// Let's denote m=-k, then: 
// x[n]=(1/N)*SUM[m=0 to 1-N]*{X[m]*exp(-j*(2*pi/N)*k*n)==FFT(X[m])
// We know that FFT is circularly periodic, thus X[m]=X[-k]=X[n-k]. 
// Therefore we can get X[m], simply by reversing the order of X[k].
// --------------------------------------------------------------

inline vector<complex<T>>  IFFTStride (const vector<complex<T>>& s)
{
  vector<complex<T>> sConj(s);          // Copy the input signal.
  // ---------------------------------- //
  // Flip the frequency spectrum
  // ---------------------------------- //
  reverse(next(sConj.begin()),sConj.end()); // Reverse the frequency spectrum.
  const double siz=sConj.size();        // The length of conjugated spectrum.
  // ---------------------------------- //
  // Normalize the signal by 1/N using lambda function.
  // ---------------------------------- //
  transform(sConj.begin(), sConj.end(), sConj.begin(), 
    [siz](complex<T> x){return x/static_cast<T>(siz);}); // Normalize the signal.
  return FFTStride(sConj);              // Return the FFT of the conjugate.
}

// ---------- BluesteinFFT: Arbitrary-N FFT via Chirp-Z & convolution (O(N log M)) ---------- //
// This method computes the length-N DFT using a pair of chirps and one linear convolution
// implemented by our own FFT-based overlap method. Convolution length M is chosen as the
// next power-of-two >=(2*N-1) so we reuse FFTStride for speed and avoid re-implementing
// a second convolution engine. Very robust when N is not a convenient composite.
inline vector<complex<T>> BluesteinFFT (const vector<complex<T>>& x)
{
  const size_t N=x.size();
  if(N==0)return {};
  if(N==1)return x;

  // Precompute chirp c[n]=exp(+j*pi*n^2/N) and its conjugate table b[k]=conj(c[k]) for convolution.
  const T pi=M_PI;
  vector<complex<T>> a(N),b; // a holds x[n]*conj(c[n]); b will be b[0..M-1] with b[k]=c[k]^* for k in [0..N-1] and mirrored index.
  a.reserve(N);
  for(size_t n=0;n<N;++n)
  {
    T ang=pi*(static_cast<T>(n)*static_cast<T>(n))/static_cast<T>(N);
    complex<T> c=std::exp(complex<T>(0,ang)); // c[n]
    a[n]=x[n]*std::conj(c);                   // a[n]=x[n]*c^*[n]
  }
  // Build b of length >=2N-1 with the sequence b[k]=c[k] for k>=0 but used circularly as in Bluestein:
  // In practice we place b[0..N-1]=exp(+j*pi*k^2/N) and zero-pad to M; the convolution engine handles the rest.
  size_t M=1;while(M<2*N-1)M<<=1;
  b.assign(M,complex<T>(0,0));
  for(size_t k=0;k<N;++k)
  {
    T ang=pi*(static_cast<T>(k)*static_cast<T>(k))/static_cast<T>(N);
    b[k]=std::exp(complex<T>(0,ang)); // b[k]=c[k]
  }

  // Pad 'a' to M and convolve: y=IFFT(FFT(a_pad).*FFT(b))
  vector<complex<T>> a_pad(M,complex<T>(0,0));
  std::copy(a.begin(),a.end(),a_pad.begin());
  vector<complex<T>> A=FFTStride(a_pad);
  vector<complex<T>> B=FFTStride(b);
  vector<complex<T>> Y(M);
  for(size_t i=0;i<M;++i)Y[i]=A[i]*B[i];
  vector<complex<T>> y=IFFTStride(Y); // Linear convolution result length M.

  // Final de-chirp: X[k]=y[k]*c[k] for k in [0..N-1].
  vector<complex<T>> X(N);
  for(size_t k=0;k<N;++k)
  {
    T ang=pi*(static_cast<T>(k)*static_cast<T>(k))/static_cast<T>(N);
    complex<T> c=std::exp(complex<T>(0,ang));
    X[k]=y[k]*c;
  }
  return X;
}

// ---------- BluesteinIFFT: inverse DFT via the same chirp trick (conj/forward/conj/scale) ---------- //
// We avoid duplicating the entire derivation by using the classic identity:
// IFFT_N(X)=conj(FFT_N(conj(X)))/N. This routes through our BluesteinFFT so it works for any N.
inline vector<complex<T>> BluesteinIFFT (const vector<complex<T>>& X)
{
  if(X.empty())return {};
  vector<complex<T>> Xc(X.size());
  for(size_t i=0;i<X.size();++i)Xc[i]=std::conj(X[i]);      // conj input
  vector<complex<T>> xc=BluesteinFFT(Xc);                   // forward using Bluestein
  for(auto& z:xc)z=std::conj(z);                            // conj result
  NormalizeByN(xc);                                         // scale by 1/N
  return xc;
}

// ---------- RaderFFT: Prime-N DFT via length-(N-1) cyclic convolution ---------- //
// For prime N, we map k=0 separately and remap the remaining indices using a primitive root g
// modulo N so that X[k] becomes a cyclic convolution over the multiplicative group Z_N^*.
// We then compute that convolution with FFTStride on a zero-padded length M>=2*(N-1)-1.
inline vector<complex<T>> RaderFFT (const vector<complex<T>>& x)
{
  const int N=static_cast<int>(x.size());
  if(N==0)return {};
  if(N==1)return x;
  // Quick primality check (simple): if not prime, user should not call Rader directly.
  auto is_prime=[&](int n){if(n<2)return false;for(int d=2;d*d<=n;++d)if(n%d==0)return false;return true;};
  if(!is_prime(N))throw std::invalid_argument{"RaderFFT: N must be prime"};

  // DC term is just the sum of all samples.
  complex<T> X0(0,0);
  for(const auto& v:x)X0+=v;

  // Find primitive root g of Z_N.
  const int g=PrimitiveRootPrime(N);

  // Build a (length N-1) and b (length N-1):
  // a[m]=x[g^m mod N], b[m]=exp(-j*2*pi*(g^{-m} mod N)/N) for m=0..N-2 (classic Rader formulation).
  const int L=N-1;
  vector<complex<T>> a(L),b(L);
  // forward powers of g mod N
  int gm=1;
  for(int m=0;m<L;++m){gm=(static_cast<long long>(gm)*g)%N;a[m]=x[gm];}
  // inverse powers g^{-m} mod N using modular inverse of g
  const int ginv=ModInv(g,N);
  int gim=1;
  for(int m=0;m<L;++m)
  {
    gim=(static_cast<long long>(gim)*ginv)%N;
    T ang=-2.0*M_PI*static_cast<T>(gim)/static_cast<T>(N);
    b[m]=std::exp(complex<T>(0,ang));
  }

  // Compute c=a (*) b via FFT-based linear convolution of length M>=2L-1.
  size_t M=1;while(M<static_cast<size_t>(2*L-1))M<<=1;
  vector<complex<T>> A(M,complex<T>(0,0)),B(M,complex<T>(0,0));
  std::copy(a.begin(),a.end(),A.begin());
  std::copy(b.begin(),b.end(),B.begin());
  A=FFTStride(A);B=FFTStride(B);
  vector<complex<T>> C(M);
  for(size_t i=0;i<M;++i)C[i]=A[i]*B[i];
  vector<complex<T>> c=IFFTStride(C); // linear conv result, first L valid for circular sum

  // Assemble output: X[0]=X0, and for k in 1..N-1, X[k]=X0 + c[m] with m mapping by k=g^m.
  vector<complex<T>> X(N);
  X[0]=X0;
  // Map k sequence via forward powers of g again.
  gm=1;
  for(int m=0;m<L;++m){gm=(static_cast<long long>(gm)*g)%N;X[gm]=X0+c[m];}
  return X;
}

// ---------- RaderIFFT: inverse prime-N DFT through the same Rader engine (conj trick) ---------- //
// Same conj/forward/conj/scale approach, but we route through RaderFFT so we keep the prime mapping.
inline vector<complex<T>> RaderIFFT (const vector<complex<T>>& X)
{
  if(X.empty())return {};
  vector<complex<T>> Xc(X.size());
  for(size_t i=0;i<X.size();++i)Xc[i]=std::conj(X[i]);      // conj input
  vector<complex<T>> xc=RaderFFT(Xc);                       // forward (prime)
  for(auto& z:xc)z=std::conj(z);                            // conj back
  NormalizeByN(xc);                                         // 1/N
  return xc;
}

// ---------- GoodThomasFFT: Prime-Factor FFT for co-prime n1,n2 (N=n1*n2) ---------- //
// This maps 1-D index k to a 2-D lattice (k1,k2) using the Chinese Remainder Theorem
// with gcd(n1,n2)==1. The transform factorizes into an n1-point and an n2-point FFT
// with ZERO twiddle multiplications (butterfly-free twiddles). Finally we CRT-map back.
// We reuse FFTStride on the inner transforms by physically gathering rows/cols.
inline vector<complex<T>> GoodThomasFFT (
  const vector<complex<T>>& x,
  int n1,
  int n2)
{
  if(static_cast<int>(x.size())!=n1*n2)throw std::invalid_argument{"GoodThomasFFT: size!=n1*n2"};
  auto gcd=[&](int a,int b){while(b){int t=a%b;a=b;b=t;}return a;};
  if(gcd(n1,n2)!=1)throw std::invalid_argument{"GoodThomasFFT: n1 and n2 must be co-prime"};

  const int N=n1*n2;
  // Precompute CRT mapping coefficients (u,v such that u*n1+v*n2=1).
  const int u=ModInv(n1,n2); // n1*u = 1 (mod n2)
  const int v=ModInv(n2,n1); // n2*v = 1 (mod n1)

  // 1) Scatter x[k] into x2D[k1][k2] with bijection: k = k1*n2*v + k2*n1*u (mod N)
  vector<vector<complex<T>>> x2D(n1,vector<complex<T>>(n2));
  for(int k=0;k<N;++k)
  {
    // Solve k1=k(mod n1) and k2=k(mod n2) (direct residues).
    int k1=k%n1;int k2=k%n2;
    x2D[k1][k2]=x[k];
  }

  // 2) FFT along k2 dimension for each fixed k1 (n2-length FFTs).
  for(int k1=0;k1<n1;++k1)
  {
    vector<complex<T>> row=x2D[k1];
    row=FFTStride(row);
    x2D[k1]=row;
  }

  // 3) FFT along k1 dimension for each fixed k2 (n1-length FFTs).
  for(int k2=0;k2<n2;++k2)
  {
    vector<complex<T>> col(n1);
    for(int k1=0;k1<n1;++k1)col[k1]=x2D[k1][k2];
    col=FFTStride(col);
    for(int k1=0;k1<n1;++k1)x2D[k1][k2]=col[k1];
  }

  // 4) Gather back to 1-D spectrum using CRT inverse index: K=K1*n2*v+K2*n1*u (mod N).
  vector<complex<T>> X(N);
  for(int K1=0;K1<n1;++K1)
  {
    for(int K2=0;K2<n2;++K2)
    {
      int K=( (static_cast<long long>(K1)*n2%N)*v%N + (static_cast<long long>(K2)*n1%N)*u%N )%N;
      X[K]=x2D[K1][K2];
    }
  }
  return X;
}

// ---------- GoodThomasIFFT: inverse PFA using two small IFFTs and CRT gather ---------- //
// Because Good-Thomas has zero twiddles in the middle, the inverse is simply:
// 1) reshape via the same CRT grid, 2) IFFT along each dimension, 3) CRT-gather back.
inline vector<complex<T>> GoodThomasIFFT (
  const vector<complex<T>>& X,
  int n1,
  int n2)
{
  if(static_cast<int>(X.size())!=n1*n2)throw std::invalid_argument{"GoodThomasIFFT: size!=n1*n2"};
  auto gcd=[&](int a,int b){while(b){int t=a%b;a=b;b=t;}return a;};
  if(gcd(n1,n2)!=1)throw std::invalid_argument{"GoodThomasIFFT: n1 and n2 must be co-prime"};

  const int N=n1*n2;
  const int u=ModInv(n1,n2); // n1*u=1(mod n2)
  const int v=ModInv(n2,n1); // n2*v=1(mod n1)

  // Scatter spectrum into 2-D: K1=K mod n1, K2=K mod n2 (same residue split as forward).
  vector<vector<complex<T>>> X2D(n1,vector<complex<T>>(n2));
  for(int K=0;K<N;++K){int K1=K%n1;int K2=K%n2;X2D[K1][K2]=X[K];}

  // IFFT along K1 dimension for each fixed K2 (n1-length).
  for(int K2=0;K2<n2;++K2)
  {
    vector<complex<T>> col(n1);
    for(int K1=0;K1<n1;++K1)col[K1]=X2D[K1][K2];
    col=IFFTStride(col);
    for(int K1=0;K1<n1;++K1)X2D[K1][K2]=col[K1];
  }

  // IFFT along K2 dimension for each fixed K1 (n2-length).
  for(int K1=0;K1<n1;++K1)
  {
    vector<complex<T>> row=X2D[K1];
    row=IFFTStride(row);
    X2D[K1]=row;
  }

  // Gather back to 1-D time sequence with CRT index: k=k1*n2*v+k2*n1*u (mod N).
  vector<complex<T>> x(N);
  for(int k1=0;k1<n1;++k1)
    for(int k2=0;k2<n2;++k2)
    {
      int k=( (static_cast<long long>(k1)*n2%N)*v%N + (static_cast<long long>(k2)*n1%N)*u%N )%N;
      x[k]=X2D[k1][k2];
    }
  // Note: IFFTStride already applied 1/n1 and 1/n2 internally for each dimension via its own scaling.
  // Together they amount to 1/N overall as required, so no extra scaling here.
  return x;
}

// ---------- StockhamAutosortFFT: Constant-geometry, in-place ping-pong stages ---------- //
// This version performs radix-2 Stockham with ping-pong buffers so that each stage writes
// in natural order for the NEXT stage (hence "auto-sort"), eliminating separate bit-reversal.
// Great cache behavior; no explicit permutation pass at the beginning or end.
// Saves the bit-reversal permutation step of the Cooley-Tukey FFT so it is great for 
// images, and video where FFTs have to be performed accross 2/3 dimensions, per kernel
// across all neighbouring pixels and their connectivities to pixels in their hood and
// outside of.
inline vector<complex<T>> StockhamAutosortFFT (const vector<complex<T>>& x)
{
  size_t N=x.size();
  if(N==0)return {};
  if((N&(N-1))!=0)throw std::invalid_argument{"StockhamAutosortFFT: N must be power-of-two"};

  vector<complex<T>> a=x,b(N);
  const int nbits=UpperLog2(static_cast<int>(N));
  // Precompute twiddles for the maximum size; reuse via striding.
  vector<complex<T>> W=TwiddleFactor(static_cast<int>(N)); // length N/2 roots

  // Outer loop over stages s=0..nbits-1, each with butterfly size m=2^{s+1}
  for(int s=0;s<nbits;++s)
  {
    size_t m=1ull<<(s+1);          // butterfly span this stage
    size_t mh=m>>1;                // half span
    size_t blocks=N/m;             // number of groups
    // Each output index n maps from input index with bit shuffles; Stockham layout ensures coalesced access.
    for(size_t k=0;k<blocks;++k)
    {
      size_t base=k*m;
      for(size_t j=0;j<mh;++j)
      {
        // Twiddle for this j across blocks repeats every N/m
        size_t wstep=N/m;
        complex<T> w=W[(j*wstep)%(N/2)];
        // Butterfly: write to b in order (base+j) and (base+j+mh)
        complex<T> u=a[base+j];
        complex<T> t=w*a[base+j+mh];
        b[base+2*j]=u+t;
        b[base+2*j+1]=u-t;
      }
    }
    a.swap(b); // ping-pong
  }
  // If nbits is odd, result is in 'a'; if even, last swap put it in 'a' as well (since we swap every stage).
  return a;
}

// ---------- StockhamAutosortIFFT: constant-geometry inverse (ping-pong, conj-twiddles) ---------- //
// Mirrors the forward Stockham but uses conjugated twiddles and final 1/N scaling. Because
// each stage writes in the order the next stage will read, we keep that cache-friendly auto-sort.
inline vector<complex<T>> StockhamAutosortIFFT (const vector<complex<T>>& X)
{
  size_t N=X.size();
  if(N==0)return {};
  if((N&(N-1))!=0)throw std::invalid_argument{"StockhamAutosortIFFT: N must be power-of-two"};

  vector<complex<T>> a=X,b(N);
  const int nbits=UpperLog2(static_cast<int>(N));
  vector<complex<T>> W=TwiddleFactor(static_cast<int>(N)); // forward roots length N/2

  for(int s=0;s<nbits;++s)
  {
    size_t m=1ull<<(s+1);      // span
    size_t mh=m>>1;
    size_t blocks=N/m;
    for(size_t k=0;k<blocks;++k)
    {
      size_t base=k*m;
      for(size_t j=0;j<mh;++j)
      {
        size_t wstep=N/m;
        complex<T> w=W[(j*wstep)%(N/2)];
        w=std::conj(w);                               // inverse uses conj twiddle
        // Inverse butterfly
        complex<T> u=a[base+2*j];
        complex<T> v=a[base+2*j+1];
        // Solve for originals: u'=u+v, t'=(u-v), then multiply odd branch by conj(w)
        complex<T> up=u+v;
        complex<T> tp=u-v;
        b[base+j]=up*complex<T>(0.5,0.0);            // delay scale until end? Keep 1/2 here for numerical symmetry
        b[base+j+mh]=tp*w*complex<T>(0.5,0.0);
      }
    }
    a.swap(b); // ping-pong
  }
  // The per-stage 1/2 factors above accumulate to 1/N. If you prefer, remove those 0.5
  // and call NormalizeByN(a) here once. We already accounted per-stage, so no extra scale now.
  return a;
}
// ---------- ZoomFFT: narrowband spectrum around f_center using heterodyne + decimate + short FFT ---------- //
// Concept: We translate the band-of-interest down to baseband (complex mix), reduce the sampling rate so that
// the new Nyquist barely covers the requested span (keeps data light, max res at window interval), apply a short window, compute a short FFT,
// and finally center-clip to n_out bins around DC (which corresponds to original f_center).
// Notes:
//  - We intentionally keep filtering minimal (this is a "utility" Zoom-FFT). For aggressive decimation you should
//    pre-bandlimit externally or swap the trivial window for a stronger low-pass. The comments below call that out.
//  - Output is FFT-shifted so DC is at the center (human-friendly for taking symmetric slices).
inline vector<complex<T>> ZoomFFT (
  const vector<complex<T>>& x, // Input time-series (complex). If you pass real, pre-cast to complex<T>(r,0).
  const double fs,             // Original sample rate in Hz.
  const double f_center,       // Center frequency (absolute Hz) of the zoom band.
  const double f_span,         // Half-bandwidth of interest (Hz). We want a ±f_span window around f_center.
  const int n_out)             // Number of output bins to return (must be <=FFT length chosen inside).
{
  if(x.empty()||fs<=0||f_span<=0||n_out<=0)return {};
  // ------------------------------ //
  // 1) Choose decimation so that new fs approx=2*f_span (Nyquist just covers span). We clamp to >=1.
  //    IMPORTANT: Without a sharp LPF, large decimation risks aliasing when the source has energy outside ±f_span.
  //    For safe use, keep dec small unless you've pre-filtered or the signal is naturally narrowband.
  // ------------------------------ //
  int dec=static_cast<int>(std::floor(fs/(2.0*f_span)));
  if(dec<1)dec=1;
  const double fs_dec=fs/static_cast<double>(dec);
  // ------------------------------ //
  // 2) Build the decimated, mixed-to-baseband buffer y[m]=x[n]*exp(-j*2p f_center*n/fs) with n=m*dec.
  //    We take the newest contiguous block that fits (right-aligned), padding with zeros if short.
  //    Then we pick an FFT length N=n_out, power-of-two for speed.
  // ------------------------------ //
  const size_t n_dec_avail=x.size()/static_cast<size_t>(dec);
  size_t N=1u<<UpperLog2(static_cast<int>(std::max<size_t>(n_out,1)));
  if(N<n_out)N<<=1; // ensure N>=n_out
  if(N>n_dec_avail)N=static_cast<size_t>(1u<<UpperLog2(static_cast<int>(std::max<size_t>(n_dec_avail,1))));
  if(N<static_cast<size_t>(n_out))N=static_cast<size_t>(n_out); // final guard
  // We'll read the last N decimated samples (most recent window).
  vector<complex<T>> y(N,complex<T>(0,0));
  const size_t last_base=(x.size()>=N*static_cast<size_t>(dec)?x.size()-N*static_cast<size_t>(dec):0);
  for(size_t m=0;m<N;++m)
  {
    size_t n=last_base+m*static_cast<size_t>(dec);
    if(n>=x.size())break;
    const double ang=-2.0*M_PI*f_center*(static_cast<double>(n)/fs);
    const complex<T> w=std::exp(complex<T>(0,static_cast<T>(ang)));
    y[m]=x[n]*w; // heterodyne to baseband
  }
  // ------------------------------ //
  // 3) Apply a light analysis window to reduce spectral leakage (Hann). No heap churn, trivial math.
  //    You can wire this to your existing Window<T> infra if you prefer. Keeping local keeps this drop-in.
  // ------------------------------ //
  for(size_t m=0;m<N;++m)
  {
    const T win=static_cast<T>(0.5*(1.0-std::cos(2.0*M_PI*static_cast<double>(m)/static_cast<double>(N-1))));
    y[m]*=win;
  }
  // ------------------------------ //
  // 4) Short FFT at fs_dec. Then fftshift to put DC at the center, and finally center-clip n_out bins.
  // ------------------------------ //
  vector<complex<T>> Y=FFTStride(y);
  Y=FFTShift(Y); // center DC for intuitive slicing
  vector<complex<T>> Z;Z.reserve(n_out);
  const size_t mid=N/2;
  const int half=n_out/2;
  const size_t start=(mid>=static_cast<size_t>(half)?mid-static_cast<size_t>(half):0);
  for(size_t i=0;i<static_cast<size_t>(n_out)&&start+i<Y.size();++i)Z.push_back(Y[start+i]);
  // Return the zoomed, centered spectrum. Bin spacing is ?f=fs_dec/N. DC==original f_center, and bins run ± around it.
  return Z;
}
// ---------- ZoomIFFT: inverse of the *full* zoom-baseband spectrum (IFFT ? optional upconvert) ---------- //
// This routine expects the *full* baseband spectrum length N used by ZoomFFT *before* you center-clip it,
// i.e., the spectrum right after FFTShift(Y) but with N bins (not n_out). If you only kept a slice, you
// lost information and there is no perfect inverse unless you zero-pad the missing bins first.
// It returns the time-domain narrowband signal at the *decimated* rate fs_dec=fs/dec (baseband by default).
// If 'upconvert' is true we heterodyne it back around f_center, but the sampling rate remains fs_dec, which
// is intentionally "zoomed" and not suitable to represent a high RF at its original Nyquist?this is for
// narrowband playback/analysis, not for reconstructing the original wideband record.
inline vector<complex<T>> ZoomIFFT(
  const vector<complex<T>>& X_full_centered, // Full-length N, centered (fftshifted) baseband spectrum (NOT a slice).
  const double fs,                            // Original sample rate in Hz.
  const double f_center,                      // Center frequency used on forward path.
  const double f_span,                        // Half-bandwidth used on forward path (determines dec).
  const bool upconvert=false)                 // If true, modulate back to f_center (at decimated rate).
{
  const size_t N=X_full_centered.size();
  if(N==0||fs<=0||f_span<=0)return {};
  // Recompute decimation to match the forward path?s policy.
  int dec=static_cast<int>(std::floor(fs/(2.0*f_span)));if(dec<1)dec=1;
  const double fs_dec=fs/static_cast<double>(dec);
  // Undo the fftshift (centered?standard order), then IFFT to obtain baseband time-series y[m] at fs_dec.
  // We implement unshift by the same helper (shift twice == rotate by N/2), but here we?ll just rotate back:
  vector<complex<T>> X_unshift(N);
  const size_t h=N/2;
  for(size_t i=0;i<h;++i)X_unshift[i]=X_full_centered[i+h];
  for(size_t i=0;i<h;++i)X_unshift[i+h]=X_full_centered[i];
  vector<complex<T>> y=IFFTStride(X_unshift); // y at baseband, decimated rate
  // Optional gentle synthesis window (mirror of analysis). For perfect recon you?d
  // want to use the same window pair; in practice Hann is near-symmetric, so we often skip here.
  // If you want it, uncomment:
  // for(size_t m=0;m<N;++m){T win=static_cast<T>(0.5*(1.0-std::cos(2.0*M_PI*double(m)/double(N-1))));y[m]*=win;}
  // Optional upconvert back near f_center (still at fs_dec). This is for ?auditioning? the narrowband chunk.
  if(upconvert)
  {
    for(size_t m=0;m<N;++m)
    {
      const double t=static_cast<double>(m)/fs_dec; // decimated timeline
      const double ang=+2.0*M_PI*f_center*t;
      const complex<T> w=std::exp(complex<T>(0,static_cast<T>(ang)));
      y[m]*=w;
    }
  }
  return y;
}
// ---------- FFTAutoSelect: policy router over Stockham/Rader/Good-Thomas/Bluestein ---------- //
// Chooses a numerically sensible, cache-friendly path based on N.
// 1) Power-of-two ? Stockham auto-sort (no bit reversal, great locality).
// 2) Prime ? Rader (length-(N-1) cyclic convolution via our FFTStride).
// 3) Composite with co-prime split ? Good-Thomas (zero twiddles in the middle).
// 4) Otherwise ? Bluestein (arbitrary-N via Chirp-Z + convolution).
inline vector<complex<T>> FFTAutoSelect (const vector<complex<T>>& x)
{
  const int N=static_cast<int>(x.size());
  if(N==0)return {};
  if(IsPowerOfTwo(static_cast<size_t>(N)))return StockhamAutosortFFT(x);
  if(IsPrime(N))return RaderFFT(x);
  if(auto pair=FindCoprimeFactorization(N))return GoodThomasFFT(x,pair->first,pair->second);
  return BluesteinFFT(x);
}

// ---------- IFFTAutoSelect: inverse policy ? Stockham/Rader/Good-Thomas/Bluestein ---------- //
inline vector<complex<T>> IFFTAutoSelect (const vector<complex<T>>& X)
{
  const int N=static_cast<int>(X.size());
  if(N==0)return {};
  if(IsPowerOfTwo(static_cast<size_t>(N)))return StockhamAutosortIFFT(X);
  if(IsPrime(N))return RaderIFFT(X);
  if(auto pair=FindCoprimeFactorization(N))return GoodThomasIFFT(X,pair->first,pair->second);
  return BluesteinIFFT(X);
}

// ---------- FFTAutoExplain: tiny reason string for your logs/tests ---------- //
inline std::string FFTAutoExplain (size_t N)const
{
  if(N==0)return "empty";
  if(IsPowerOfTwo(N))return "stockham-pow2";
  if(IsPrime(static_cast<int>(N)))return "rader-prime";
  if(FindCoprimeFactorization(static_cast<int>(N)))return "good-thomas-coprime";
  return "bluestein-general";
}

// ============================= FFT and IFFT Algorithms ============================= //

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FFT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// So to get the FFT of a signal x(n) of length N we have to divide and conquer.
// We do this by using the Cooley-Tukey Algorithm, described here as:
// 1. Divide the signal into even and odd samples, so we have:
//      for (i.begin(); i.end() )
//        evenT[i]=x[i*2]
//        oddT[i]=x[i*2+1]
// 2. Conquer the signal; we recursively apply the FFT on both halves:
//      X_even(k) = FFT(evenT)
//      X_odd(k) = FFT(oddT)
// 3. Now we precompute the twiddles factors, this saves A LOT of time:
//      TwiddleFactor(k) = exp( -j * (2*pi/n)*k)
// 4. Having the twiddles Factors we compute the FFT butterfly to obtain the full
//    frequency spectrum - the amplitude and phase of sines and cosines that
//    composit it.
//      for (k.begin; k.end/2)
//        t=TwiddleFactor(k)*X_odd(k)
//        X(k)=X_even(k)+t
//        X(k+N/2)=X_even(k)-t
// 5. Return the spectrum of the signal
// Note that this algorithm should be slower than the FFT Stride above, but it 
// is also clearer. 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

inline vector<complex<T>>  FFT(const vector<complex<T>>& s)
{
  const int N=s.size();                 // The length of the input signal.
    // -------------------------------- //
    // Base Case: When the input is 1, return the signal.
    // -------------------------------- //
  if (N<=1)                             // Is it a single point?
    return s;                           // Yes, return the signal.
    // -------------------------------- //
    // Divide Step: Divide the signal into even and odd samples.
    // -------------------------------- //
  vector<complex<T>> evenT(N/2), oddT(N/2);
  for (int i=0; i<N/2;++i)              // Up to the folding frequency.
  {
    evenT[i]=s[i*2];                    // Get the even samples.
    oddT[i]=s[i*2+1];                   // Get the odd samples.
  }                                     // Done decimating in time.
    // -------------------------------- //
    // Conquer Step: Recurse, apply FFT to evens and odds.
    // -------------------------------- //
  vector<complex<T>> evenF=FFT(evenT);  // Transform even samples.
  vector<complex<T>> oddF=FFT(oddT);    // Transform odd samples.
    // -------------------------------- //
    // Precompute the twiddles factors.
    // -------------------------------- //
  vector<complex<T>> tf=TwiddleFactor(N);// Get the phase-freq rotation vector.
    // -------------------------------- //
    // Compute the FFT butterfly for this section
    // -------------------------------- //
  vector<complex<T>> S(N);              // Initialize freq-domain vector.
  complex<T> t{0.0,0.0};                // Single root of unity.
  for (int k=0; k<N/2; ++k)             // Up to the folding frequency.
  {
    // -------------------------------- //
    // Get the amplitude phase contribution for current butterfly phase.
    // -------------------------------- //
    t=twiddles[k]*oddF[k];               // Scale and get this freq bin.
    // -------------------------------- //
    // Prepare results for next butterfly phase.
    // -------------------------------- //
    S[k]=evenF[k]+t;                    // Produce even result for nxt butterfly.
    S[k]=oddF[k]-t;                     // Produce odd result for nxt butterfly.
  }                                     // Done computing butterfly.
    // -------------------------------- //
    // Return computed spectrum. 
    // -------------------------------- //
  return S;                             // The computed spectrum.
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IFFT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// So how do we calculate the Inverse FFT? To get the inverse of a signal X(k)
// of length k an elegant trick is performed.
// The IFFT is nothing more than the FFT multiplied by 1/N, and with a twiddles
// factor that rotates clockwise, instead of counter-clockwise.
// It is nothing more than the conjugate of the FFT multiplied by (1/N).
//
// The operation goes as follows:
// 1. Conjugate the discrete frequency input signal X(k):
//    X_conjugate(k) = Re(X(k)) - j*Im(X(k))
// 2. Next we perform the FFT on the conjugated signal, this performs the IDFT
//    but does not scale - that comes afterwards:
//    X(k) = SUM[n=0 to N-1] {x(n) * exp(-j*(2*pi/N)*k*n) }
// 3. Now we conjugate again and multiply by (1/N), this returns our signal to
//    the time domain: (wizardry at hand!)
//    x(n) = (1/N) * SUM[k=0 to N-1] * {X(k) * exp(-j*(2*pi/N)*k*n)}
// 4. Return the time-domain samples of the signal.
// Note that this algorithm should be slower than the IFFT Stride above, but it 
// is also clearer.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

inline vector<complex<T>>  IFFT (const vector<complex<T>> &s)
{
  const int N=s.size();                 // Get the length of the signal.
    // -------------------------------- //
    // 1. Reverse the frequency spectrum by conjugating the input signal.
    // -------------------------------- //
  vector<complex<T>> sConj(N);          // Reversed spectrum buffer.
  for (int i=0; i<N;++i)                // For every sample in the spectrum
    sConj[i]=conj(s[i]);                //   reverse the frequency bins.
    // -------------------------------- //
    // 2. Perform FFT on the conjugated spectrum.
    // -------------------------------- //
  vector<complex<T>> S=FFT(sConj);      // Reverse-spectrum buffer.
    // -------------------------------- //
    // 3. Conjugate and normalize the signal.
    // -------------------------------- //
  vector<complex<T>> sig(N);            // Signal buffer.
  for (int i=0;i<N;++i)                 // For all samples in reversed spectrum
    sig[i]=conj(S[i])/static_cast<T>(N);// Reverse and normalize.
  return sig;                          // Return time-domain signal.  
} 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Method to perform the convolution of two signals. Typically in our context
// the convolution is done between an input signal s(n) of length N  and
// filter of length N h(n).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

inline vector<complex<T>> Convolution( // Obtain the product of two signals.
    const vector<complex<T>> &s,        // Our input signal.
    const vector<complex<T>> &h)        // Our filter.
{                                       // ---------- Convolution ----------- //
    const int n = s.size();             // Length of the input signal.
    const int m = h.size();              // Length of the filter.
    int N = 1;                          // Size of the N-point FFT.
    while (N < n + m - 1)               // While N is less than the convolution
        N <<= 1;                        // Find smallest power of 2 >= n+m-1
    // -------------------------------- //
    // Zero-Pad the signal and filter to the length of the N-Point FFT.
    // -------------------------------- //
    vector<complex<T>> sPad = ZeroPad(s); // Zero pad the input signal.
    vector<complex<T>> hPad = ZeroPad(h); // Zero pad the filter.
    // -------------------------------- //
    // Apply the FFT on the Zero-Padded signal.
    // -------------------------------- //
    vector<complex<T>> S = FFTStride(sPad);   // The FFT of the input signal.
    vector<complex<T>> H = FFTStride(hPad);   // The FFT of the filter.
    // -------------------------------- //
    // Now the filtered signal is just the product of their spectrum.
    // -------------------------------- //
    vector<complex<T>> Y(N);            // Place to store resulting spectrum.
    for (int i = 0; i < N; ++i)         // For the N-Points of the FFT.
        Y[i] = S[i] * H[i];             // Get the filtered spectrum.
    // -------------------------------- //
    // Obtain the time-domain resulting signal using the IFFT.
    // -------------------------------- //
    vector<complex<T>> y(N);            // Place to store resulting signal.
    y = IFFTStride(Y);                        // Get the resulting signal.
    y.resize(n + m - 1);                // Truncate to the original size.
    return y;                           // Return the filtered signal.
}                                       // ---------- Convolution ----------- //

// Multiply two-length-N spectra element by element.

inline vector<complex<T>> PointwiseMul(const std::vector<std::complex<T>>& A,const std::vector<std::complex<T>>& B) const
{                                       // ---------- PointwiseMul ----------- //
  assert(A.size()==B.size());           // Ensure both spectra are of the same length.
  std::vector<std::complex<T>> C(A.size()); // Create a vector to hold the result.
  for (size_t i=0;i<A.size();++i)       // For each spectral element...
    C[i]=A[i]*B[i];                     // Multiply the two spectra element by element.
  return C;                             // Return the resulting spectrum.
}                                       // ---------- PointwiseMul ----------- //

// Short-Time Fourier Transform

inline vector<vector<complex<T>>> STFT(const vector<complex<T>> &s, 
  const WindowType &w, 
  int wSiz, 
  const float overlap)
{
  int step = wSiz * (1 - overlap / 100.0);// step size based on the ovlap %
  int nSegs = (s.size() - wSiz + step) / step;// # segs for the signal.
  vector<vector<complex<T>>> sMat(nSegs, vector<complex<T>>(wSiz));
  vector<T> window = this->window->GetWindow();// Window to be applied.
    // -------------------------------- //
    // Process each seg of the signal. Each row of the matrix is a frame of
    // the signal, and every column a frequency bin inside the windowed segment.
    // -------------------------------- //
  for (int i=0;i<nSegs;++i)
  {   
      int start = i * step;             // Starting ndx for our current segment.
      vector<complex<T>> seg(wSiz, 0.0); // Initialize the seg with zeros
    // -------------------------------- //    
    // Apply the window to the segment and copy the windowed signal.
    // For the size of the window and remaining frequency bins of the signal.
    // -------------------------------- //
      for (int j = 0; j <wSiz && start+j< s.size(); ++j)
        seg[j] = s[start + j] * window[j];
    // -------------------------------- //
    // Compute the FFT of the windowed seg and store it in the STFT matrix
    // -------------------------------- //
      sMat[i] = FFTStride(seg);               // Compute the FFT of the windowed segment.
  }                                     // Done with all segments.            
  return sMat;                          // The windowed spectrum.
}

// Inverse Short-Time-Fourier-Transform Method.
inline vector<complex<T>> ISTFT(
    const vector<vector<complex<T>>> &sMat,// Input STFT matrix
    const WindowType &w,                // Window type
    const int wSiz,                     // Window size (should be power of 2)
    const float ovlap)                  // Overlap percentage (e.g., 50%)
{
    int step = wSiz*(1 -ovlap/100.0);// step size based on the ovlap %
    int len = step*(sMat.size()-1)+wSiz;// Length of the original signal.
    // --------------------------------- //
    // Calculate the overlap count for the normalization step.
    // -------------------------------- //
    vector<complex<T>> sig(len,0.0);// Initialize the result signal
    vector<T> nOverlaps(len,0.0); // Initialize the overlap count
    // Generate the window to be applied during the inverse process
    vector<T> window = this->window.GenerateWindow(w, wSiz); // Get the window to be applied.
    // -------------------------------- //
    //  Process each seg of the signal. Each row of the matrix is a frame of
    //  of the signal, and each column a frequency bin inside the window.
    // -------------------------------- //
    for (int i = 0; i < sMat.size(); ++i)
    {
      int start = i*step;               // Starting ndx of the frame (segment).
    // -------------------------------- //
    // Compute the IFFT of the current segment, and get the time-domain short
    // signal. This allows us to reconstruct it back using its segments. 
    // -------------------------------- //
      vector<complex<T>> seg = IFFTStride(sMat[i]);
    // -------------------------------- //
    // Overlap-add the IFFT result to the output signal. Because the segments
    // were windowed, we overlap-add each segment to obtain total signal's energy
    // contribution. Because the energy of the signal is greater in the
    // samples that lie in the overlapped region, we keep track of these in the 
    // second step. This allows us not to overshoot the signal's original ampli
    // -tude in these regions, as the signal's energy is purposely overlapped
    // by the spectral windows.
    // -------------------------------- // 
      for (int j=0;j<wSiz && (start+j)<sig.size();++j)
      {
        sig[start+j]+=seg[j]*window[j]; // Frame contribution to long signal.
        nOverlaps[start+j]+=window[j];  // Keep track of Overlapp-Add Windowing process 
      }
    }
    // -------------------------------- //
    // Normalize the result by dividing by the overlap count. We want to know 
    // how many times each segment of the signal was affected by the OLA Window
    // process, so that we can scale the original signal's amplitude accurately.
    // -------------------------------- //
    for (int i = 0; i < sig.size(); ++i)
      if (nOverlaps[i] != 0.0)
        sig[i] /= nOverlaps[i];
    // Return the reconstructed time-domain signal
    return sig;
}

// ---------- SpectralFreeze: freeze the timbre by holding magnitudes over time ---------- //
// Idea: compute STFT, pick a "freeze" frame (or average of a small region), then reuse that
// magnitude for all frames while preserving phase-advance to keep the output alive and not static.
// This yields gorgeous pads, drones, and sustained textures from short snippets.
inline vector<complex<T>> SpectralFreeze(
  const vector<complex<T>>& x,          // input (complex ok; for real just pass real+0j)
  const WindowType& w,                  // analysis/synthesis window
  int wSiz,float ovlap,                 // FFT size/window size and overlap (e.g., 1024, 50)
  int freeze_frame=-1,                  // which frame to freeze; if <0 we pick the max-energy frame
  T mix=static_cast<T>(1))              // 1=full freeze, 0=original; can blend
{
  // 1) STFT
  vector<vector<complex<T>>> X=STFT(x,w,wSiz,ovlap);
  if(X.empty())return {};

  // 2) Choose freeze frame (max energy if user did not specify)
  int fz=freeze_frame;
  if(fz<0)
  {
    T best=-1;fz=0;
    for(int i=0;i<X.size();++i)
    {
      T e=0;for(auto& z:X[i])e+=static_cast<T>(std::norm(z));
      if(e>best){best=e;fz=i;}
    }
  }
  fz=std::clamp<int>(fz,0,(int)X.size()-1);

  // 3) Grab target magnitude
  const vector<complex<T>>& Xfz=X[fz];
  vector<T> mag_fz(Xfz.size());
  for(size_t k=0;k<Xfz.size();++k)mag_fz[k]=static_cast<T>(std::abs(Xfz[k]));

  // 4) Impose (blended) magnitudes while leaving phase progression from the running STFT
  for(size_t i=0;i<X.size();++i)
  {
    for(size_t k=0;k<X[i].size();++k)
    {
      T ph=std::arg(X[i][k]);
      T m=static_cast<T>(std::abs(X[i][k]));
      T newm=m*(1-mix)+mag_fz[k]*mix;
      X[i][k]=std::polar(newm,ph);
    }
  }

  // 5) ISTFT back
  return ISTFT(X,w,wSiz,ovlap);
}

// ---------- TimeStretchPhaseVocoder: classic phase-vocoder with phase advance ---------- //
// stretch>1 slows down, <1 speeds up. We track inter-frame phase increments and resample frames.
inline vector<complex<T>> TimeStretchPhaseVocoder(
  const vector<complex<T>>& x,const WindowType& w,int wSiz,float ovlap,T stretch)
{
  if(stretch<=0)stretch=1;
  // 1) STFT
  vector<vector<complex<T>>> X=STFT(x,w,wSiz,ovlap);
  if(X.size()<2)return ISTFT(X,w,wSiz,ovlap);

  const size_t K=X[0].size();
  // 2) Accumulate phase and resample frames along time axis
  vector<vector<complex<T>>> Y;Y.reserve((size_t)(X.size()*stretch)+1);
  vector<T> phi(K,0),prev_phase(K,0),expected(K,0);
  const T two_pi=static_cast<T>(2*M_PI);
  // Initialize phases from first frame
  for(size_t k=0;k<K;++k){phi[k]=std::arg(X[0][k]);prev_phase[k]=phi[k];}

  for(T pos=0;pos<(T)X.size()-1;pos+=1/stretch)
  {
    // fractional frame index and linear mag interp
    int i0=(int)pos;T a=pos-(T)i0;
    const vector<complex<T>>& F0=X[i0];
    const vector<complex<T>>& F1=X[i0+1];
    vector<complex<T>> out(K);

    // phase advance: unwrap delta_phase=(phase1-phase0) - expected
    for(size_t k=0;k<K;++k)
    {
      T mag=(1-a)*static_cast<T>(std::abs(F0[k]))+a*static_cast<T>(std::abs(F1[k]));
      T ph0=std::arg(F0[k]);
      T ph1=std::arg(F1[k]);
      T d=ph1-ph0;
      // principal value to (-pi,pi]
      d=std::remainder((double)d,(double)two_pi);
      // advance accumulated phase
      phi[k]+=d;
      out[k]=std::polar(mag,phi[k]);
    }
    Y.push_back(out);
  }
  return ISTFT(Y,w,wSiz,ovlap);
}

// ---------- PitchShiftPhaseVocoder: resample in freq after stretching ---------- //
// Simple strategy: time-stretch by 1/r, then resample by r (phase-locked magnitudes).
inline vector<complex<T>> PitchShiftPhaseVocoder(
  const vector<complex<T>>& x,const WindowType& w,int wSiz,float ovlap,T pitch_ratio)
{
  if(pitch_ratio<=0)pitch_ratio=1;
  // Stretch so duration compensates pitch move (classic PSOLA-ish PV combo)
  vector<complex<T>> y=TimeStretchPhaseVocoder(x,w,wSiz,ovlap,static_cast<T>(1.0/pitch_ratio));
  // Frequency-domain resample per frame
  vector<vector<complex<T>>> Y=STFT(y,w,wSiz,ovlap);
  for(auto& frame:Y)
  {
    vector<complex<T>> z(frame.size(),complex<T>(0,0));
    for(size_t k=0;k<frame.size();++k)
    {
      T src=(T)k/pitch_ratio;
      size_t k0=(size_t)src;
      T a=src-(T)k0;
      if(k0+1<frame.size())
      {
        complex<T> v=frame[k0]*(1-a)+frame[k0+1]*a;
        z[k]=v;
      }
    }
    frame.swap(z);
  }
  return ISTFT(Y,w,wSiz,ovlap);
}

// ---------- SpectralMorphCross: blend magnitudes of A&B while keeping phase from a donor ---------- //
// Rich, vocoder-like textures. Use mag of A and B (blend by 'mix'), and phases from 'phase_from'
// =0 ? use A's phase, =1 ? use B's phase. Great for turning noise into vowels, etc.
inline vector<complex<T>> SpectralMorphCross(
  const vector<complex<T>>& a,const vector<complex<T>>& b,
  const WindowType& w,int wSiz,float ovlap,T mix=static_cast<T>(0.5),int phase_from=0)
{
  vector<vector<complex<T>>> A=STFT(a,w,wSiz,ovlap);
  vector<vector<complex<T>>> B=STFT(b,w,wSiz,ovlap);
  size_t F=min(A.size(),B.size());
  if(F==0)return {};
  A.resize(F);B.resize(F);
  vector<vector<complex<T>>> Y(F,vector<complex<T>>(A[0].size()));

  for(size_t i=0;i<F;++i)
  {
    for(size_t k=0;k<A[i].size();++k)
    {
      T ma=static_cast<T>(std::abs(A[i][k]));
      T mb=static_cast<T>(std::abs(B[i][k]));
      T m=ma*(1-mix)+mb*mix;
      T ph=(phase_from==0?std::arg(A[i][k]):std::arg(B[i][k]));
      Y[i][k]=std::polar(m,ph);
    }
  }
  return ISTFT(Y,w,wSiz,ovlap);
}

// vector<complex<T>> OLAProcessor(const vector<complex<T>> &s, const vector<complex<T>> &h, const WindowType &w, const int wSiz, const float ovlap)
inline vector<complex<T>> OLAProcessor(
    const vector<complex<T>> &s, // The input signal.
    const vector<complex<T>> &h, // The desired FIR filter.
    const WindowType &w,         // The window used.
    const int wSiz,              // The size of the window.
    const float ovlap)           // The percentage of overlap
{
    vector<vector<complex<T>>> sMat = STFT(s, w, wSiz, ovlap); // STFT of input signal.
    vector<complex<T>> H = FFTStride(h);      // FFT of the FIR filter.
    const int frames = sMat.size();     // Number of frames in the STFT matrix.
    vector<vector<complex<T>>> sig(frames, vector<complex<T>>(wSiz));
    // -------------------------------- //
    // Perform element-wise multiplication of the STFTs
    // -------------------------------- //
    for (int i = 0;i<frames;++i)        // For the number of frames...
      for (int j=0;j <wSiz; ++j)        // .. and the freq bins in the window.
        sig[i][j]=sMat[i][j]*H[j];      // Convlute the short signal.
    // -------------------------------- //
    // Perform the inverse STFT to get the filtered time-domain signal.
    // -------------------------------- //
    return ISTFT(sig, w, wSiz, ovlap);
}
// vector<complex<T>> OLAProcessor(const vector<complex<T>> &s, const vector<complex<T>> &h, const WindowType &w, const int wSiz, const float ovlap)

inline vector<complex<T>> OLAProcessor(
    const vector<complex<T>> &s, // The input signal.
    const WindowType &w,         // The window used.
    const int wSiz,              // The size of the window.
    const float ovlap)           // The percentage of overlap
{
    vector<vector<complex<T>>> sMat = STFT(s, w, wSiz, ovlap);// STFT of input signal.
    vector<complex<T>> H(wSiz,complex<T>(0.0,0.0));// Dummy Impulse filter.
    H=FFTStride(H);                           // FFT of the FIR filter.
    const int frames = sMat.size();     // Number of frames in the STFT matrix.
    vector<vector<complex<T>>> sig(frames, vector<complex<T>>(wSiz));
    // -------------------------------- //
    // Perform element-wise multiplication of the STFTs
    // -------------------------------- //
    for (int i = 0;i<frames;++i)        // For the number of frames...
      for (int j=0;j <wSiz; ++j)        // .. and the freq bins in the window.
        sig[i][j]=sMat[i][j]*H[j];      // Convlute the short signal.
    // -------------------------------- //
    // Perform the inverse STFT to get the filtered time-domain signal.
    // -------------------------------- //
    return ISTFT(sig, w, wSiz, ovlap);
}
// Determine the power sepctral density of a windowed signal using Welch's method.

inline vector<T> WelchPSD(
 const vector<T> &s,                    // The signal to process.
 const WindowType& w,                   // The window to apply to the signal.
 const int wSiz,                        // The size of the window.
 const float ovlap,                     // Overlap percentage (50% typical)
 const int fftsiz)                      // The size of the FFT.
{
    // -------------------------------- //
    // Compute the STFT of the signal.
    // -------------------------------- //
  vector<vector<complex<T>>> stftMat=STFT(s,w,wSiz,ovlap);
    // -------------------------------- //
    // Determine the scale factor of the window.
    // -------------------------------- //
  vector<T> pxxAvg(length,T(0));             // Initialize PSD Buffer
  const double winScl=pow(norm(this->window.GenerateWindow(w,wSiz),2),2); // Get Scale Factor.
    // -------------------------------- //
    // Now we accumulate the PSD for each segment in the STFT matrix.
    // -------------------------------- //
  for (int i=0; i<stftMat.size();++i)   // For each fft segment.
  {                                     // Where each row in the STFT matrix is
    const vector<complex<T>>& fftSegment=stftMat[i];//  an FFT segment
    // -------------------------------- //
    // Compute the Power Spectal Density
    // -------------------------------- //
    for (int j=0;j<fftsiz;++j)          // For every sample to process by the FFT
      pxxAvg[j]+=norm(fftSegment[j])/winScl;// Accumulate sq. magnitude.
  }                                     // Done accumulating sq.magntiude segments.
    // -------------------------------- //
    // Now we average the PSD over the segments
    // -------------------------------- //
  for (int i=0; i<fftsiz; ++i)          // For every sample to process by the FFT
    pxxAvg[i]/=stftMat.size();          // Average PSD over all segments.
    // -------------------------------- //
    // and normalize the PSD over 2*pi periodic interval.
    // -------------------------------- //
  for (int i=0; i<fftsiz;++i)           // For every sample...
    pxxAvg[i]/=(2*M_PI);                //
    // -------------------------------- //
    // Ensure the total energy of the sigal is conserved in the sprectrm.
    // Parseval's Theorem: SUM[n to N-1] x[n]^2 = (1/N) SUM[n to N-1] X[k]^2
    // -------------------------------- //
  pxxAvg[0]/=2;                         // Avergave freq bin 1 (DC component).
  for (int i=0;i<fftsiz;++i)            // For ever sample count the energy in
    pxxAvg[i]*=2;                       //  positive and negative halves
    // -------------------------------- //
    // Return the first half of the PSD (our FFT is symmetric) and we already
    // have recollected all the power.
    // -------------------------------- //
  return vector<T>(pxxAvg.begin(),pxxAvg.being()+fftsiz/2+1);

}
// Method to perform a frequency shift of the center frequency by a const amount

inline vector<complex<T>> Shift(  // Shift the signal in frequency domain.
  const vector<complex<T>>& s,           // The input signal.
  const double fShift,                  // The amount to shift it by
  const double fs)                      // The sample rate of the signal
{                                       // ---------- Shift ----------------- //
  vector<complex<T>> sDelay(s.size());  // Our shifted spectrum.
  T phaseShift=(-2.0*M_PI*fShift/fs);   // Precompute phase shift.
  complex<T> delayMod(1.0,0.0);         // Initialize modulator to angle 0.
  for (size_t i=0; i<s.size(); ++i)     // For the existence of our signal..
  {
    sDelay[i]=s[i]*delayMod;            // Calculate this frequency bin.
    delayMod*=polar(1.0,phaseShift);    // Increment the phase.
  }                                     // Done delay shifting the signal.
  return sDelay;
}                                       // ---------- Shift ----------------- //
// Method to perform a carrier sweep with a start and stop frequency about the 
// center frequency

inline vector<vector<complex<T>>> Sweep(
  const vector<complex<T>>&s,           // The input signal
  const double fStart,                  // The Start Frequency.
  const double fCenter,                 // The center frequency
  const double fStop,                   // The stop frequency
  const double step,                    // The number of bins to jump
  const double fs,                      // The sample rate of the signal
  const WindowType &w,                  // The window to apply to the signal
  const int wSiz,                       // The size of the window
  const float ovlap)                    // The overlap factor
{                                       // ---------- Sweep ----------------- //
    // -------------------------------- //
    // First we shift the input signal about the center frequency in 
    // our spectum down to 0 Hz. This normalizes our signal to simplify analysis.
    // -------------------------------- //
  vector<complex<T>> cSig=Shift(s,-fCenter, fs);
  vector<vector<complex<T>>> sMat; 
    // -------------------------------- //
    // Scan through different frequency bands relative to the center frequency.
    // -------------------------------- //
  for (double freq=fStart; freq<=fStop; freq+=step)
  {
    // -------------------------------- //
    // Having our signal at 0 Hz we observe the behaviour of our signal when 
    // shifted by various frequencies relative to the central moment. 
    // -------------------------------- //
    vector<complex<T>> sShift=Shift(cSig,freq,fs);
    // -------------------------------- //
    // Apply the window to the signal to reduce spectral leakage.
    // -------------------------------- //
    vector<complex<T>> S=OLAProcessor(sShift,w,wSiz,ovlap);
    // -------------------------------- //
    // Perform an FFT on the windowed signal to analyze the energy 
    // of the signal at this frequency offset.
    // -------------------------------- //
    vector<complex<T>> spectrum=FFTStride(S);
    // -------------------------------- //
    // Normalize the FFT result by a suitable factor (maybe wSiz?) This should
    // distribute the energy across the window.
    // -------------------------------- //
    for (size_t i=0;i<spectrum.size();++i)
      spectrum[i]/=static_cast<T>(wSiz);
    // -------------------------------- //
    // Next we recollect the resulting FFT spectrum for this frequency offset,
    // to obtain the full spectra that represents the signal's behaviour accross
    // the entire sweep range.
    // -------------------------------- //
    sMat.push_back(spectrum);
  }
    // -------------------------------- //
    // Having collected all of the signal's behaviour across the frequency 
    // sweep, return the full signal's spectra.
    // -------------------------------- //
  return sMat;
}
// ---------- GenerateChirp: create a complex baseband chirp with ? control & band-limit ---------- //
// This routine generates a complex exponential chirp y[n]=exp(j*f[n]) where the instantaneous
// angular velocity ?(t)=df/dt is driven by your chosen model (Linear/Exponential/Hyperbolic).
// You directly set the *initial* angular velocity f0 (rad/s) and an *angular acceleration* a,
// and we cap the instantaneous *frequency* to f_limit (Hz) to keep the chirp inside the audio band.
// Notes on models:
//  - Linear:        ?(t)=f0+a*t. This gives constant angular acceleration a (rad/s^2).
//  - Exponential:   ?(t)=f0*exp(k*t). We choose k=a/max(f0,e) so that d?/dt|t=0?a.
//  - Hyperbolic:    ?(t)=K/(t+c). We choose c=Clamp(-f0/a,1.0e-9,1.0e9) so that ?(0)?f0 and d?/dt|0?a.
// In all cases, before integrating ? into f, we clamp ? to ±?_lim where ?_lim=2p*f_limit to respect the
// requested audio limit. This means "up to a specified audio frequency" is always honored in the sweep.
//
// Parameters:
//   N         : number of samples to synthesize
//   fs        : sample rate in Hz
//   omega0    : initial angular velocity f0 in rad/s (e.g., 2p*1000 for 1 kHz start)
//   alpha     : angular acceleration parameter (see model mapping above), rad/s^2 for Linear,
//               and an initial-slope proxy for Exponential/Hyperbolic
//   f_limit   : absolute frequency fence (Hz) for |f_inst|; set to e.g. 20000 for 20 kHz audio band
//   type      : ChirpType::Linear / Exponential / Hyperbolic
//   phi0      : starting phase (radians)
//
// Returns:
//   vector<complex<T>> length N with unit magnitude complex chirp.
//
inline vector<complex<T>> GenerateChirp(
  size_t N,double fs,double omega0,double alpha,double f_limit,
  ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> y(N,complex<T>(0,0));
  if(N==0||fs<=0)return y;
  const double Ts=1.0/fs;
  const double two_pi=2.0*M_PI;
  const double omega_lim=two_pi*std::abs(f_limit); // we clamp by magnitude
  // Map a into each model's ?(t). For Exponential we choose k so that d?/dt|0?a.
  const double eps=1.0e-12;
  const double k_exp=(std::abs(omega0)>eps)?(alpha/std::abs(omega0)):0.0; // ?(t)=f0*exp(k*t)
  // For hyperbolic, choose c so that ?(0)?f0 and slope ~a; ?(t)=K/(t+c), K=f0*c.
  double c_hyp;
  if(std::abs(alpha)>eps)c_hyp=Clamp(-omega0/alpha,1.0e-9,1.0e9); else c_hyp=1.0e6;
  const double K_hyp=omega0*c_hyp;

  double phi=phi0; // running phase (integral of ?)
  for(size_t n=0;n<N;++n)
  {
    const double t=static_cast<double>(n)*Ts;
    double omega;
    switch(type)
    {
      case ChirpType::Linear:      omega=omega0+alpha*t;break;
      case ChirpType::Exponential: omega=omega0*std::exp(k_exp*t);break;
      case ChirpType::Hyperbolic:  omega=K_hyp/(t+c_hyp);break;
      default:                     omega=omega0+alpha*t;break;
    }
    // Enforce audio fence on instantaneous frequency by clipping |?|.
    if(omega>omega_lim)omega=omega_lim;
    if(omega<-omega_lim)omega=-omega_lim;

    // Integrate angular velocity to phase via simple rectangle rule. For high precision you
    // could do trapezoidal, but at audio rates rectangle is more than fine for long sweeps.
    phi+=omega*Ts;

    // Emit the complex chirp sample at this step: e^{j f[n]}.
    y[n]=std::exp(complex<T>(0,static_cast<T>(phi)));
  }
  return y;
}
// ---------- ApplyChirpyness: modulate any input by the generated chirp (keeps amplitude) ---------- //
// This routine takes your input signal x[n] (real or complex) and multiplies it by a unit-magnitude
// chirp generated by GenerateChirp(...) so that the "carrier" sweeps according to your ?-controls.
// The chirp is clamped to f_limit so it never exceeds the specified audio band.
// Overloads: one for real-valued input, one for complex-valued input.
inline vector<complex<T>> ApplyChirpyness(
  const vector<T>& x,double fs,double omega0,double alpha,double f_limit,
  ChirpType type,double phi0=0.0)const
{
  const size_t N=x.size();
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  for(size_t n=0;n<N;++n)c[n]*=complex<T>(x[n],0);
  return c;
}
inline vector<complex<T>> ApplyChirpyness(
  const vector<complex<T>>& x,double fs,double omega0,double alpha,double f_limit,
  ChirpType type,double phi0=0.0)const
{
  const size_t N=x.size();
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  vector<complex<T>> y(N);
  for(size_t n=0;n<N;++n)y[n]=x[n]*c[n];
  return y;
}
// ---------- GenerateRealChirp: cosine-only version when you need a strictly real sweep ---------- //
// Sometimes you want a pure real chirp (for DACs or export). We simply take Re{exp(jf)}=cos(f).
inline vector<T> GenerateRealChirp(
  size_t N,double fs,double omega0,double alpha,double f_limit,
  ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  vector<T> y(N);
  for(size_t n=0;n<N;++n)y[n]=c[n].real();
  return y;
}
// ---------- MakeChirp: user-friendly dispatcher (f0,f1,duration ? omega0/alpha) ---------- //
// This is the high-level entry point for chirp synthesis. You tell us where to start (f0 in Hz),
// where to end (f1 in Hz), and over what time (duration in seconds). We compute the model-appropriate
// angular parameters (omega0,alpha) and then delegate to GenerateChirp(...) which integrates angular
// velocity to phase while honoring the requested audio-band fence (f_limit).
//
// Important modeling notes (how we map (f0,f1,T) to angular params for each chirp type):
//  * Linear model (theta(t)=f0+a*t):
//      w0=2pi*f0, a=2pi*(f1-f0)/T so that w(T)=2pi*f1. Classic constant angular acceleration sweep.
//  * Exponential model (theta(t)=theta0*exp(k*t)):
//      Choose k=ln(|theta1|/|theta0|)/T (magnitudes for robustness). Initial slope is dtheta/dt|0=k*theta0,
//      so we pass a=w0*k. This makes GenerateChirp?s exponential branch reproduce the intended curve.
//  * Hyperbolic model (w(t)=K/(t+c)):
//      Enforce w(0)=w0 and w(T)=w1 ? c=T*w1/(w0-w1) (guards below handle degeneracies).
//      Initial slope dw/dt|0=-(f0/c), so we pass a=-f0/c; GenerateChirp reconstructs c via c-f0/a.
//  * Audio fence: we clamp both endpoints (f0,f1) to ±f_limit before mapping to angular units, and
//    GenerateChirp further clamps instantaneous ? at every sample, so the sweep never exceeds the band.
//
// Parameters:
//   N         : number of samples to synthesize (length of the chirp buffer)
//   fs        : sample rate in Hz
//   f0,f1     : start/end frequencies in Hz (we clamp to ±f_limit to keep things safe)
//   duration  : intended sweep time in seconds; if <=0 we infer T=N/fs
//   f_limit   : absolute audio fence in Hz (e.g., 20000)
//   type      : ChirpType::Linear / Exponential / Hyperbolic
//   phi0      : initial phase in radians
//
// Returns: complex unit-magnitude chirp of length N.
//
inline vector<complex<T>> MakeChirp(
  size_t N,double fs,double f0,double f1,double duration,
  double f_limit,ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> y(N,complex<T>(0,0));
  if(N==0||fs<=0)return y;

  // Resolve sweep time T. If caller passes non-positive, infer from N and fs to keep behavior intuitive.
  double Td=duration>0.0?duration:static_cast<double>(N)/fs;
  if(Td<=0.0)Td=static_cast<double>(N)/fs; // last guard

  // Enforce the audio fence early (endpoint clamp) so our angular mapping is already bounded.
  double f0c=Clamp(f0,-f_limit,f_limit);
  double f1c=Clamp(f1,-f_limit,f_limit);

  // Map to angular endpoints f0, ?1. We work in radians/sec throughout.
  const double two_pi=2.0*M_PI;
  double omega0=two_pi*f0c;
  double omega1=two_pi*f1c;

  // Compute model-specific "alpha" parameter that our GenerateChirp expects.
  // See big comment above: Linear uses a directly; Exponential uses a=f0*k; Hyperbolic uses a=-f0/c.
  double alpha=0.0;
  const double eps=1.0e-12;

  switch(type)
  {
    case ChirpType::Linear:
    {
      // w(T)=f0+a*T=w1 ? a=(w1-f0)/T
     alpha=(omega1-omega0)/Td;
    }break;

    case ChirpType::Exponential:
    {
      // w(t)=f0*exp(k*t), target |w1|=|f0|*exp(k*T) ? k=ln(|w1|/|f0|)/T.
      // Initial slope is dw/dt|0=k*f0 ? a=f0*k feeds our generator's exponential branch.
      double w0mag=std::max(std::abs(omega0),eps);
      double w1mag=std::max(std::abs(omega1),eps);
      double k=std::log(w1mag/w0mag)/std::max(Td,eps);
      // Preserve sign of omega0 in the slope naturally via a=f0*k.
      alpha=omega0*k;
    }break;

    case ChirpType::Hyperbolic:
    {
      // w(t)=K/(t+c), with w(0)=f0 ? K=f0*c. Also require w(T)=w1 ? c=T*w1/(f0-w1).
      // Guard degenerate cases (equal endpoints, near-zero denominators).
      double denom=omega0-omega1;
      double c;
      if(std::abs(denom)<eps)
      {
        // If w0<w1 the hyperbolic law degenerates toward constant w; pick a large c (slow variation) and a=0.
        c=1.0e6;
      }
      else
      {
        c=(Td*omega1)/denom;
        // Keep c positive and reasonable to avoid 1/(t+c) singularities near t=0.
        if(c<1.0e-9)c=1.0e-9;
      }
      alpha=-omega0/c; // so GenerateChirp can reconstruct c?-f0/a internally
    }break;

    default:
    {
     alpha=(omega1-omega0)/Td; // fall back to linear behavior
    }break;
  }

  // Delegate to the core generator which integrates ? and clamps per-sample to ±2p*f_limit.
  return GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
}

// ---------- MakeRealChirp: cosine-only convenience wrapper for DAC-friendly output ---------- //
// Same dispatcher as above, but we return a strictly real sweep using cos(f). Handy when you want
// an audio buffer you can ship straight to playback without dealing with I/Q pairs.
inline vector<T> MakeRealChirp(
  size_t N,double fs,double f0,double f1,double duration,
  double f_limit,ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> c=MakeChirp(N,fs,f0,f1,duration,f_limit,type,phi0);
  vector<T> y(N);
  for(size_t i=0;i<N;++i)y[i]=c[i].real();
  return y;
}

// ---------- ApplyChirpynessFromF: multiply an input by a dispatcher-driven chirp ---------- //
// This is a friendly "f0,f1,T" front end for ApplyChirpyness. We synthesize the matching chirp and
// modulate the caller-provided buffer so it "rides" the requested sweep inside the audio fence.
inline vector<complex<T>> ApplyChirpynessFromF(
  const vector<complex<T>>& x,double fs,double f0,double f1,double duration,
  double f_limit,ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> c=MakeChirp(x.size(),fs,f0,f1,duration,f_limit,type,phi0);
  vector<complex<T>> y(x.size());
  for(size_t n=0;n<x.size();++n)y[n]=x[n]*c[n];
  return y;
}
inline vector<complex<T>> ApplyChirpynessFromF(
  const vector<T>& x,double fs,double f0,double f1,double duration,
  double f_limit,ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> c=MakeChirp(x.size(),fs,f0,f1,duration,f_limit,type,phi0);
  vector<complex<T>> y(x.size());
  for(size_t n=0;n<x.size();++n)y[n]=complex<T>(x[n],0)*c[n];
  return y;
}
// ---------- ChirpParamsFromF: compute (omega0,alpha) for logging/inspection/UI ---------- //
// Friendly wrapper: maps (f0,f1,duration) to angular units exactly the way MakeChirp() does,
// but without generating a buffer. Useful for displaying slopes, debugging parameterization,
// or storing presets.
//
// Returns: pair<omega0,alpha> in radians/sec and radians/sec²-equivalent (depending on model).
//   * Linear: omega(t)=omega0+a*t, a has units rad/s²
//   * Exponential: omega(t)=omega0*exp(k*t), we return a=f0*k so the slope at t=0
//   * Hyperbolic: omega(t)=K/(t+c), we return a=-f0/c which lets GenerateChirp reconstruct c
//
inline std::pair<double,double> ChirpParamsFromF(
  double f0,double f1,double duration,double f_limit,ChirpType type)const
{
  // Enforce audio fence
  double f0c=Clamp(f0,-f_limit,f_limit);
  double f1c=Clamp(f1,-f_limit,f_limit);

  const double two_pi=2.0*M_PI;
  double omega0=two_pi*f0c;
  double omega1=two_pi*f1c;

  // Resolve duration T
  double Td=duration>0.0?duration:1.0; // if duration<=0, fallback to 1s nominal for param
  const double eps=1.0e-12;

  double alpha=0.0;
  switch(type)
  {
    case ChirpType::Linear:
  alpha=(omega1-omega0)/Td;
      break;
    case ChirpType::Exponential:
    {
      double w0mag=std::max(std::abs(omega0),eps);
      double w1mag=std::max(std::abs(omega1),eps);
      double k=std::log(w1mag/w0mag)/std::max(Td,eps);
      alpha=omega0*k;
    }break;
    case ChirpType::Hyperbolic:
    {
      double denom=omega0-omega1;
      double c;
      if(std::abs(denom)<eps) c=1.0e6;
      else
      {
        c=(Td*omega1)/denom;
        if(c<1.0e-9)c=1.0e-9;
      }
      alpha=-omega0/c;
    }break;
    default:
      alpha=(omega1-omega0)/Td;break;
  }
  return {omega0,alpha};
}
private:
    vector<T> signal;                        // The signal to process.
    vector<T> subCarrier;                    // 
    WindowType window;                       // The window to apply to the signal.
    int windowSize=0;                    // The size of the window.
    float overlap;                     // The overlap factor.
    vector<complex<T>> twiddles;             // Precomputed twiddles factors.
    double sRate;                      // The rate at which we sampled the RF.
    int length;                        // The length of the signal.
};

 // Approximate Maximum Likelihood Fundamental Frequency Estimator
 template<typename T>
 static inline std::optional<T> FreqMLEReal(
   const vector<T>& s,          // The input signal.
   const T fStart,              // The start frequency.
   const T fStop,               // The stop frequency.
   const T fs)                  // The number of samples per second.
 {                               // ---------- FreqMLEReal ----------------- //
  static_assert(is_floating_point<T>::value, "T must be a floating point type.");
  // Validate inputs
  if (!(fs> T(0))) return std::nullopt;   // Invalid sample rate
  if (!(fStart>= T(0) && fStop>= T(0))) return std::nullopt; // Negative band
  if (fStart>=fStop) return std::nullopt; // Inverted or empty band
  if (s.size()<4) return std::nullopt;     // Too short

    // -------------------------------- //
    // 1. Calculate the FFT of the input signal.
    // -------------------------------- //
  SpectralOps<T> engine;                // Use SpectralOps for FFT utilities.
  vector<complex<T>> X=engine.FFTStride(s);    // Get the FFT of the signal.
  const std::size_t N=X.size();         // Get the length of the FFT.
  if (N<4) return std::nullopt;         // If the FFT is too short, return no result.
    // -------------------------------- //
    // 2. Translate from Start Frequency and Stop Frequency to bin 
    // indices in the FFT.
    // -------------------------------- //
  const T bins=fs/static_cast<T>(N);   // Hz per FFT bin
  size_t kilo=std::clamp<size_t>(static_cast<size_t>(std::floor(fStart/bins)),1,N/2-1);
  size_t kimax=std::clamp<size_t>(static_cast<size_t>(std::ceil (fStop /bins)), 1, N/2-1);
  if (kilo>kimax) return std::nullopt; // Safety: invalid range in bins
    // -------------------------------- //
    // 3. Find Peak magnitude square in the search band
    // -------------------------------- //
     auto itmax=std::max_element(X.begin()+static_cast<std::ptrdiff_t>(kilo),
                                 X.begin()+static_cast<std::ptrdiff_t>(kimax)+1,
       [](const std::complex<T>& a, const std::complex<T>& b){return std::norm(a)<std::norm(b);});// peak in band
     if (itmax==X.end()) return std::nullopt;
     size_t k=static_cast<size_t>(std::distance(X.begin(),itmax)); // index of peak
    // -------------------------------- //
    // 4. 3-point parabolic interpolation (f0+-1 bins)
    // -------------------------------- //
   if (k==0||k>=N/2||k<=kilo||k>=kimax) // avoid edges of DC/Nyquist and search edges
     return std::nullopt;
    // -------------------------------- //
    // Get the power of the peak and its neighbours.
    // -------------------------------- //
     // Floor magnitude to avoid log(0) -> -inf
     const T eps = static_cast<T>(1e-30);
     const T mL = std::max<T>(std::norm(X[k-1]), eps);
     const T mC = std::max<T>(std::norm(X[k  ]), eps);
     const T mR = std::max<T>(std::norm(X[k+1]), eps);
     T alpha=std::log(mL);
     T beta =std::log(mC);
     T gamma=std::log(mR);
     if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(gamma))
       return std::nullopt;
     T denom=(alpha - T(2)*beta + gamma); // Interpolation denominator
    // -------------------------------- //
    // Clealculate the descent step based on the parabolic interpolation.
    // If the denominator is zero, we cannot interpolate.
    // -------------------------------- //
     T delta = T(0);
     if (std::isfinite(denom) && std::abs(denom) > static_cast<T>(1e-20))
       delta = static_cast<T>(0.5)*(alpha - gamma)/denom; // (-0.5?0.5) ideally
     if (!std::isfinite(delta)) delta = T(0);
     // Clamp to a sane range
     if (delta < T(-0.5)) delta = T(-0.5);
     if (delta > T( 0.5)) delta = T( 0.5);
    // -------------------------------- //
    // 5. Refine the frequency estimate.
    // -------------------------------- //
    T fhat=(static_cast<T>(k)+delta)*bins;// Refine the frequency estimate.
    if (!std::isfinite(fhat)) return std::nullopt;
    if (fhat<fStart||fhat>fStop) return std::nullopt; // Out of bounds
    return fhat;                       // Return the estimated frequency.
 }                                     // ---------- FreqMLEReal ----------------- //
 // YIN Pitch Estimator
 template<typename T>
 static inline T PitchYIN (
  const vector<T>& s, // The input signal.
  const T fs,         // The sample rate of the signal.
  const size_t bs,    // The block size of the signal.
  T thresh=static_cast<T>(0.15))         // Tolerance for approx output
 {                                      //%PitchYIN
  static_assert(is_floating_point<T>::value, "T must be a floating point type.");
  // Use actual signal size; cap bs in case caller passed larger than s.size().
  const size_t N = std::min(bs, s.size());
  if(N < 8) return T(0);
  const size_t taumax = N/2;        // conventional YIN search limit
  if(taumax < 4) return T(0);
  std::vector<T> diff(taumax, T(0));
  std::vector<T> cum (taumax, T(0));
    // -------------------------------- //
    // 1.Difference function d(tau) = SUM[n=0 to N-1-tau] {x[n] - x[n+tau]}^2
    // -------------------------------- //
    for (size_t tau=1; tau<taumax; ++tau) {
      T acc=0;
      const size_t limit = N - tau; // ensure n+tau < N
      for(size_t n=0; n<limit; ++n){
        const T dif = s[n]-s[n+tau];
        acc += dif*dif;
      }
      diff[tau]=acc;
    }
    // -------------------------------- //
    // 2.Cumulative Running sum c(tau): c(tau) = SUM[tau=1 to taumax] d(tau)
    // -------------------------------- //
    cum[0]=1;                           // Initialize the cumulative sum.
    T rsum=0;                           // The running sum variable.
    for (size_t tau=1;tau<taumax;++tau) // For each time step...
    {
      rsum+=diff[tau];                  // Add the difference to running sum.
      cum[tau]=diff[tau]*tau/(rsum+std::numeric_limits<T>::epsilon());// Normalize the cumulative sum.
    }                                   // Done with running sum.
    // -------------------------------- //
    // 3. Absolute thresholding: if c(tau) < thresh, then tau is a pitch candidate.
    // -------------------------------- //
  size_t taue=0;                      // Our estimation variable.
    for (size_t tau=2;tau<taumax;++tau) // For each bin...
    {                                   // Threshold and rough estimate.
      if (cum[tau]<thresh)              // Is this sample less than our wanted power?
      {                                 // Yes, proceed to estimate
        // Get the parabolic minimum    //
        while (tau+1<taumax&&cum[tau+1]<cum[tau]) ++tau;
        taue=tau;                       // Store the estimated pitch.
        break;                          // Break out of the loop.
      }
    }
    // -------------------------------- //
    // 4. Parabolic refinement:
    // -------------------------------- //
    // If no threshold crossing found, pick argmin of cum (excluding boundaries)
    if(taue==0)                        // pick global minimum inside safe range
    { 
      T best = std::numeric_limits<T>::max();
      for(size_t k=2;k+2<taumax;++k)
      {
        if(cum[k]<best)
        {
          best=cum[k];
          taue=k;
        }
      }
    }
  // Ensure taue landed in a valid interior region; otherwise clamp.
    if(taue < 2) taue = 2;            // clamp low
    if(taue+1 >= taumax) taue = (taumax>3?taumax-2:2); // clamp high keeping room for +1
    if(taumax < 4 || taue >= taumax-1) return T(0); // still unsafe; bail
    const T y0=cum[taue-1];           // safe: taue>=2
    const T y1=cum[taue];
    const T y2=cum[taue+1];           // safe: taue+1<taumax
    const T denom=y0+y2-2*y1;           // Denom for parabolic interpolation.
    T tip=static_cast<T>(taue);         // Initialize interpolation var
    // -------------------------------- //
    // If the denominator is not zero, we can interpolate.
    // -------------------------------- //
    if (std::fabs(denom)>std::numeric_limits<T>::epsilon())
      tip+=(y0-y1)/denom;               // Appromiate please.
    return fs/tip;                      // Return the estimated pitch frequency.
  }                                     // ---------- PitchYIN ----------------- //
}
// ============================================================================
// Lightweight STFT / ISTFT and SpectralFreeze helpers (appended)
// ============================================================================
namespace sig::spectral {

// Short-Time Fourier Transform (returns frequency-domain frames)
template<typename T>
inline std::vector<std::vector<std::complex<T>>> STFT(
    const std::vector<std::complex<T>>& x,
    const typename Window<T>::WindowType& wType,
    int winSize,
    float overlapPc)
{
  std::vector<std::vector<std::complex<T>>> frames;
  if (winSize <= 0 || x.empty()) return frames;
  int hop = std::max(1, (int)std::round(winSize * (1.f - overlapPc/100.f)));
  hop = std::min(hop, winSize);
  Window<T> W; W.SetWindowType(wType, (size_t)winSize);
  const auto& w = W.GetData();
  SpectralOps<T> engine;
  for (int pos = 0; pos + winSize <= (int)x.size(); pos += hop) {
    std::vector<std::complex<T>> frame(winSize);
    for (int n=0;n<winSize;++n) {
      const auto& s = x[pos+n];
      frame[n] = { s.real()* (T)w[n], s.imag()* (T)w[n] };
    }
    auto X = engine.FFTStride(frame);
    frames.push_back(std::move(X));
  }
  return frames;
}

// Inverse STFT (reconstruct time-domain complex signal)
template<typename T>
inline std::vector<std::complex<T>> ISTFT(
    const std::vector<std::vector<std::complex<T>>>& frames,
    const typename Window<T>::WindowType& wType,
    int winSize,
    float overlapPc)
{
  if (frames.empty() || winSize<=0) return {};
  int hop = std::max(1, (int)std::round(winSize * (1.f - overlapPc/100.f)));
  hop = std::min(hop, winSize);
  Window<T> W; W.SetWindowType(wType, (size_t)winSize);
  const auto& w = W.GetData();
  SpectralOps<T> engine;
  size_t outLen = (frames.size()-1)*hop + winSize;
  std::vector<std::complex<T>> y(outLen, std::complex<T>(0,0));
  std::vector<T> weight(outLen, (T)0);
  size_t fi=0;
  for (const auto& X: frames) {
    auto t = engine.IFFTStride(X);
    if ((int)t.size() < winSize) t.resize(winSize, std::complex<T>(0,0));
    size_t base = fi*hop;
    for (int n=0;n<winSize;++n) {
      T winS = (T)w[n];
      y[base+n] += t[n]*winS;
      weight[base+n] += winS*winS;
    }
    ++fi;
  }
  for (size_t i=0;i<y.size();++i) if (weight[i]>(T)0) y[i]/=weight[i];
  return y;
}

// Spectral Freeze helper
template<typename T>
inline std::vector<std::complex<T>> SpectralFreeze(
  const std::vector<std::complex<T>>& x,
  const typename Window<T>::WindowType& w,
  int wSiz,float ovlap,
  int freeze_frame=-1,
  T mix=static_cast<T>(1))
{
  auto X = STFT<T>(x,w,wSiz,ovlap);
  if (X.empty()) return {};
  int fz = freeze_frame;
  if (fz < 0) {
    T best = (T)-1; fz = 0;
    for (int i=0;i<(int)X.size();++i){
      T e=0; for (auto& z: X[i]) e += (T)std::norm(z);
      if (e>best){best=e;fz=i;}
    }
  }
  fz = std::clamp<int>(fz,0,(int)X.size()-1);
  const auto& Xfz = X[fz];
  std::vector<T> mag_fz(Xfz.size());
  for (size_t k=0;k<Xfz.size();++k) mag_fz[k]=(T)std::abs(Xfz[k]);
  for (auto& frame: X){
    for (size_t k=0;k<frame.size();++k){
      T ph = (T)std::arg(frame[k]);
      T m  = (T)std::abs(frame[k]);
      T newm = m*(1-mix)+mag_fz[k]*mix;
      frame[k] = std::polar(newm, ph);
    }
  }
  return ISTFT<T>(X,w,wSiz,ovlap);
}

} // namespace sig::spectral

// ============================================================================
// Additional phase-vocoder style helpers for front-end FX (lightweight refs)
// ============================================================================
namespace sig::spectral {

// Simple single-frame cross-morph: combines magnitudes, selects phase from A or B.
template<typename T>
inline std::vector<std::complex<T>> SpectralMorphCross(
  const std::vector<std::complex<T>>& a,
  const std::vector<std::complex<T>>& b,
  const typename Window<T>::WindowType& wType,
  int fftSize,
  float /*overlapPc*/,
  T mix,
  int phaseFrom)
{
  const int N = std::min<int>({fftSize,(int)a.size(),(int)b.size()});
  if (N<=0) return {};
  Window<T> W; W.SetWindowType(wType, (size_t)N);
  SpectralOps<T> engine;
  // Time -> freq
  std::vector<std::complex<T>> ta(N), tb(N);
  for (int i=0;i<N;++i){
    T win = (T)W.GetData()[i];
    ta[i] = a[i]*win;
    tb[i] = b[i]*win;
  }
  auto FA = engine.FFTStride(ta);
  auto FB = engine.FFTStride(tb);
  const size_t M = std::min(FA.size(), FB.size());
  for (size_t k=0;k<M;++k){
    T magA = (T)std::abs(FA[k]);
    T magB = (T)std::abs(FB[k]);
    T mag  = magA*(1-mix)+magB*mix;
    T ph   = (phaseFrom<=0? (T)std::arg(FA[k]) : (T)std::arg(FB[k]));
    FA[k]  = std::polar(mag, ph);
  }
  auto y = engine.IFFTStride(FA);
  // remove analysis window energy (simple window^2 comp)
  for (int i=0;i<N && i<(int)W.GetData().size(); ++i){
    T win = (T)W.GetData()[i];
    if (win != (T)0) y[i] /= win; // rough de-window
  }
  return y;
}

// Naive pitch shift: resample magnitude envelope in frequency domain.
template<typename T>
inline std::vector<std::complex<T>> PitchShiftPhaseVocoder(
  const std::vector<std::complex<T>>& x,
  const typename Window<T>::WindowType& wType,
  int fftSize,
  float /*overlapPc*/,
  T pitchRatio)
{
  const int N = std::min<int>(fftSize,(int)x.size());
  if (N<=0) return {};
  Window<T> W; W.SetWindowType(wType,(size_t)N);
  SpectralOps<T> engine;
  std::vector<std::complex<T>> tmp(N);
  for (int i=0;i<N;++i){T win=(T)W.GetData()[i]; tmp[i]=x[i]*win;}
  auto F = engine.FFTStride(tmp);
  std::vector<std::complex<T>> G(F.size(), std::complex<T>(0,0));
  const size_t M = F.size();
  for (size_t k=0;k<M;++k){
    double src = k / std::max<double>(pitchRatio,1e-6);
    size_t k0 = (size_t)std::floor(src);
    size_t k1 = std::min(M-1, k0+1);
    double a = src - k0;
    std::complex<T> v0 = (k0<M?F[k0]:std::complex<T>{});
    std::complex<T> v1 = (k1<M?F[k1]:std::complex<T>{});
    G[k] = v0*(T)(1.0-a) + v1*(T)a;
  }
  auto y = engine.IFFTStride(G);
  for (int i=0;i<N && i<(int)W.GetData().size(); ++i){T win=(T)W.GetData()[i]; if(win!=(T)0) y[i]/=win;}
  return y;
}

// Naive time stretch (single-frame placeholder): blends linear resample in time domain.
template<typename T>
inline std::vector<std::complex<T>> TimeStretchPhaseVocoder(
  const std::vector<std::complex<T>>& x,
  const typename Window<T>::WindowType& wType,
  int fftSize,
  float /*overlapPc*/,
  T stretch)
{
  const int N = std::min<int>(fftSize,(int)x.size());
  if (N<=0) return {};
  if (std::fabs(stretch - (T)1) < (T)1e-6) return x; // identity
  int target = (int)std::max<T>(16, std::round(N*stretch));
  std::vector<std::complex<T>> y(target);
  for (int i=0;i<target;++i){
    double src = i / stretch;
    int i0 = (int)std::floor(src);
    int i1 = std::min(N-1, i0+1);
    double a = src - i0;
    auto v0 = x[i0];
    auto v1 = x[i1];
    y[i] = v0*(T)(1.0-a) + v1*(T)a;
  }
  // Optional: apply window normalization (skip for placeholder)
  return y;
}

} // namespace sig::spectral
