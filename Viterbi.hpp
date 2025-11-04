 /* 
* *
* * Filename: ConvolutionalEncoder.hpp
* *
* * Description:
* *   CCSDS-standard Rate 1/2 Convolutional Encoder, K=7, Polynomials 171(octal),133(octal).
* *  - Constraint length K=7 (6 memory elements). Rate 3/4 via puncturing over 3 input bits P=[[1.1.1],[1.0.1]]
* * 
* *
* * Author:
* *   JEP, J. Enrique Peraza
* *
* * Organization:
* *   Trivium Solutions LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
* *
*/
#pragma once
#include <cstdint>
#include <vector>
#include "../logger/Logger.h"

namespace sdr::mdm
{
  struct ConvConfig
  {
    bool p34{false};               // True for rate 3/4 via puncturing; false for rate 1/2
    ConvConfig (void)=default;     // Default constructor
  };
  class ConvolutionalEncoder
  {
    public:
      ConvolutionalEncoder (void)
      {
        Reset();                    // Initialize shift register
        lg=logx::Logger::NewLogger(); // Own a live logger instance
      }
      ~ConvolutionalEncoder (void)=default;
      inline void Reset (void) { z=0; } // Reset shift register and config
      inline void Assemble (const ConvConfig& ccfg)
      {                               // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
        this->ccfg=ccfg;              // Store configuration
      }                               // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Parity-check bit generation for one input byte (8 bits)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline uint8_t Parity (uint32_t bs) // Bit sequence to parity check
      {                                 // ~~~~~~~~~~ Parity ~~~~~~~~~~ //                                
        bs^=bs>>16;                     // Fold upper bits
        bs^=bs>>8;                      // Fold again
        bs^=bs>>4;                      // Fold again
        bs^=bs>>2;                      // Fold again
        bs^=bs>>1;                      // Fold again
        return static_cast<uint8_t>(bs&0x1);// Return parity bit
      }                                 // ~~~~~~~~~~ Parity ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Encode bit sequence
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void Encode (
        uint8_t u,                      // Input bit to encode (0 or 1)
        std::vector<uint8_t>* const o)  // Output encoded bits
      {                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
        if (lg)
          lg->Deb(" [BEGIN] Convolutional Encode: Input Bit=%d",u);
        z=((z<<1)|(u&0x01))&0x7F;       // Update shift register with input bit
        if (lg)
          lg->Deb(" Convolutional Encode: Shift Reg=%02X",z);
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Generators (octal): 171=0b1111001, 133=0b1011011 (reversed for shift register)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint32_t g1=0b1111001;          // Generator 1
        uint32_t g2=0b1011011;          // Generator 2
        uint8_t p1=Parity(z&g1);        // Compute parity bit 1
        uint8_t p2=Parity(z&g2);        // Compute parity bit 2
        if (lg)
          lg->Deb(" Convolutional Encode: Input Bit=%d Shift Reg=%02X Parity Bits=[%d,%d]",u,z,p1,p2);
        if (!ccfg.p34)                  // Rate 1/2?
        {                               // Yes, output both parity bits
          o->push_back(p1);             // Output parity bit 1
          o->push_back(p2);             // Output parity bit 2
        }                               // Done rate 1/2
        else                            // Esle its is punctured rate 3/4
        {                               //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Rate 3/4 puncture over groups of 3 inputs -> pattern indexes
          // Implemented using a rotating phase counter mod 3
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          static int ph{0};             // Puncturing phase counter
          if (ph==0)                    // Phase 0: output both parity bits
          {                             //
            o->push_back(p1);           // Output parity bit 1
            o->push_back(p2);           // Output parity bit 2
          }                             //
          else if (ph==1)               // Phase 1: output only parity bit 1
          {                             //
            o->push_back(p1);           // Output parity bit 1
          }                             //
          else                          // Phase 2: output only parity bit 2
          {                             //
            o->push_back(p2);           // Output parity bit 2
          }                             //
          ph=(ph+1)%3;                  // Advance puncturing phase
          if (lg)
            lg->Deb(" [END] Convolutional Encode: Puncturing Phase=%d",ph);
        }                               // Done rate 3/4
      }                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Encode a block of input bits
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void EncodeBits (
        const std::vector<uint8_t>* in, // Input bits
        std::vector<uint8_t>* const o)  // Output encoded bits
      {                                 // ~~~~~~~~~~ EncodeBits ~~~~~~~~~~ //
        if (in==nullptr||o==nullptr)    // Bad args?
          return;                       // Nothing to do
        o->clear();                     // Clear output buffer
        o->reserve(in->size()*2);       // Reserve space (max rate 1/2)
        for (const auto& b:(*in))       // For each input bit
          Encode(b,o);                  // Encode bit
      }                                 // ~~~~~~~~~~ EncodeBits ~~~~~~~~~~ //
    private:
      uint32_t z{0};                    // Shift register state
      ConvConfig ccfg;                  // Configuration parameters
      std::unique_ptr<logx::Logger> lg{}; // Logger instance owned by encoder
  };
  class ViterbiDecoder
  {
    public:
      ViterbiDecoder (void)
      {                                 // ~~~~~~~~~~ Constructor ~~~~~~~~~~ //
        lg = logx::Logger::NewLogger(); // Own a live logger instance
        Assemble(ConvConfig{});         // Default config
      }                                 // ~~~~~~~~~~ Constructor ~~~~~~~~~~ //
      ~ViterbiDecoder (void)
      {
        if (lg)
        {
          // Do not call ExitLog() from destructors; that shuts down the
          // shared logger and forces other modules to print to stderr.
          lg->Inf("ViterbiDecoder destructor called");
          lg.reset();
        }
      }
      inline void Assemble (const ConvConfig& conf)
      {                                 // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
        ccfg=conf;                      // Store configuration
        trellis.BuildTrellis(lg.get()); // Build trellis structure
      }                                 // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
      inline void Reset (void)
      {                                 // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
        for (int s=0;s<64;++s)
          pm[s]=INF;                    // Initialize path metrics to infinity
        pm[0]=0;                        // Start at state 0
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Puncturing phase aligns with encoder (0-> p1,p2; 1 -> p1; 2 -> p2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        ph=0;                           // Reset puncturing phase
        steps=0;                        // Reset step counter
        prev.clear();                   // Clear survivor paths
        bit.clear();                    // Clear input bits
      }                                 // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Decode a stream of hard bits (0/1). Output recovered input bits.
      // For rate 1/2: ch.size() must be even -> one input bit per 2 channel bits (what the encoder produced)
      // For rate 3/2: uses puncturing phases; every 4 channel bits correspond to 3 input bits.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void DecodeBits (
        const std::vector<uint8_t>* ch, // Input channel bits (hard 0/1)
        std::vector<uint8_t>* const o)  // Output decoded input bits
      {                                 // ~~~~~~~~~~ DecodeBits ~~~~~~~~~~ //
        if (ch==nullptr||o==nullptr||ch->empty())
          return;
        o->clear();                     // Clear output buffer
        if (lg)
          lg->Inf("[BEGIN] Viterbi DecodeBits: input channel bits=%zu",ch->size());
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Prepare survivor path storage
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        prev.reserve(prev.size()+ch->size()+64);// Reserve space
        bit.reserve(bit.size()+ch->size()+64);  // Reserve space
        size_t idx{0};                  // Index into channel bitsream
        const size_t nbits=ch->size();  // Number of channel bits
        while (true)                    // For.... almost ever.....
        {                               // Process channel bits
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Determine how many parity bits are available to this input step
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t hp1{0},hp2{0};        // Received parity bits
          uint8_t r1{0},r2{0};          // Received flags
          if (!ccfg.p34)                // Rate 1/2?
          {                             // Yes, so get both channel bits per input bit
            if (idx+1>=nbits)           // Not enough bits left?
              break;                    // Done processing
            r1=(*ch)[idx++]&1;          // Get parity bit 1
            r2=(*ch)[idx++]&1;          // Get parity bit 2
            hp1=1;                      // Mark parity bit 1 received
            hp2=1;                      // Mark parity bit 2 received
            if (lg)
              lg->Deb(" DecodeBits: Rate 1/2, r1=%d r2=%d",r1,r2);
          }
          else                          // Else it is 3/4 punctured
          {                             // 
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Rate 3/4 puncturing phase: (2,1,1) bits per 3 input bits
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (ph==0)                  // Is the phase counter 0?
            {                           // Yes, so get both parity bits
              if (idx+1>=nbits)         // Not engough bits left?
                break;                  // Done processing
              r1=(*ch)[idx++]&1;        // Get parity bit 1
              r2=(*ch)[idx++]&1;        // Get parity bit 2
              hp1=1;                    // Mark parity bit 1 received
              hp2=1;                    // Mark parity bit 2 received
              if (lg)
                lg->Deb(" DecodeBits: Rate 3/4, ph=%d r1=%d r2=%d",ph,r1,r2);
            }                           // Done with phase 0
            else if (ph==1)             // At phase 1?
            {                           // Yes, get only parity bit 1
              if (idx>=nbits)           // Not enough bits left?
                break;                  // Done processing
              r1=(*ch)[idx++]&1;        // Get parity bit 1
              hp1=1;                    // Mark parity bit 1 received
              hp2=0;                    // Parity bit 2 not received
              if (lg)
                lg->Deb(" DecodeBits: Rate 3/4, ph=%d r1=%d",ph,r1);
            }                           // Done with phase 1
            else                        // Else phase 2
            {                           // Get only parity bit 2
              if (idx>=nbits)           // Not enough bits left?
                break;                  // Done processing
              r2=(*ch)[idx++]&1;        // Get parity bit 2
              hp1=0;                    // Parity bit 1 not received
              hp2=1;                    // Mark parity bit 2 received
              if (lg)
                lg->Deb(" DecodeBits: Rate 3/4, ph=%d r2=%d",ph,r2);
            }                           // Done with phase 2
          }                             // Done with 3/4 punctured code
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // One Viterbi step or phase recovers or corrects (one information bit)
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          ViterbiStep(r1,r2,hp1,hp2);   // Perform one Viterbi time step
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Advance puncturing phase
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          if (ccfg.p34)                 // Rate 3/4?
            ph=(ph+1)%3;                // Advance puncturing phase
        }                               // Done processing channel bits
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Traceback to find best path
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        std::vector<uint8_t> urev{};    // Reversed input bits
        Traceback(&urev);               // Perform traceback
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Output in forward order
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        o->assign(urev.rbegin(),urev.rend());// Assign in forward order
        if (lg)
          lg->Inf("[END] Viterbi DecodeBits: output input bits=%zu",o->size());
      }                                 // ~~~~~~~~~~ DecodeBits ~~~~~~~~~~ //
      inline void DecodeBitsWeighted (
       const std::vector<uint8_t>* ch,  // Input channel bits (hard 0/1)
        std::vector<float>* w,          // Weights for input channel bits
        std::vector<uint8_t>* const o)  // Output decoded input bits
      {                                 // ~~~~~~~~~~ DecodeBitsWeighted ~~~~~~~~~~ //
        if (ch==nullptr||w==nullptr||o==nullptr||ch->empty())
          return;
       if (ch->size()!=w->size())      // Size mismatch?
          return;                      // Nothing to do, size must match.
        o->clear();                    // Clear output buffer
        if (lg)
          lg->Inf("[BEGIN] Viterbi DecodeBitsWeighted: input channel bits=%zu",ch->size());
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Prepare survivor path storage
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        prev.reserve(prev.size()+ch->size()+64);// Reserve space
        bit.reserve(bit.size()+ch->size()+64);  // Reserve space
        size_t idx{0};                  // Index into channel bitsream
        const size_t nbits=ch->size();  // Number of channel bits
        while (true)                    // For.... almost ever.....
        {
          uint8_t r1{0},r2{0};          // Received parity bits
          uint8_t hp1{0},hp2{0};        // Received flags
          float w1{0.f},w2{0.f};        // Weights for received bits
          if (!ccfg.p34)                // Rate 1/2?
          {
            if (idx+1>=nbits)           // Not enough bits left?
              break;                    // Done processing
            r1=(*ch)[idx]&1;            // Get parity bit 1
            w1=std::clamp((*w)[idx],0.f,1.f);// Get weight for parity bit 1
            idx++;                      // Advance index
            r2=(*ch)[idx]&1;            // Get parity bit 2
            w2=std::clamp((*w)[idx],0.f,1.f);// Get weight for parity bit 2
            idx++;                      // Advance index
            hp1=1;                      // Mark parity bit 1 received
            hp2=1;                      // Mark parity bit 2 received
            if (lg)
              lg->Deb(" DecodeBitsWeighted: Rate 1/2, r1=%d w1=%.2f r2=%d w2=%.2f",r1,w1,r2,w2);
          }                             // Done with 1/2
          else
          {
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Rate 3/4 puncturing phase: (0/1/2) bits per 3 input bits
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (ph==0)                  // Phase counter zero?
            {
              if (idx+1>=nbits)         // Not engough bits left?
                break;                  // Done processing
              r1=(*ch)[idx]&1;          // Get parity bit 1
              w1=std::clamp((*w)[idx],0.f,1.f);// Get weight for parity bit 1
              idx++;                    // Advance index
              r2=(*ch)[idx]&1;          // Get parity bit 2
              w2=std::clamp((*w)[idx],0.f,1.f);// Get weight for parity bit 2
              idx++;                    // Advance index
              hp1=1;                    // Mark parity bit 1 received
              hp2=1;                    // Mark parity bit 2 received
              if (lg)
                lg->Deb(" DecodeBitsWeighted: Rate 3/4, ph=%d r1=%d w1=%.2f r2=%d w2=%.2f",ph,r1,w1,r2,w2);
            }                           // Done with phase 0
            else if (ph==1)             // Phase 1?
            {                           // Yes
              if (idx>=nbits)           // Not enough bits left?
                break;                  // Done processing
              r1=(*ch)[idx]&1;          // Get parity bit 1
              w1=std::clamp((*w)[idx],0.f,1.f);// Get weight for parity bit 1
              idx++;                    // Advance index
              hp1=1;                    // Mark parity bit 1 received
              hp2=0;                    // Parity bit 2 not received
              if (lg)
                lg->Deb(" DecodeBitsWeighted: Rate 3/4, ph=%d r1=%d w1=%.2f",ph,r1,w1);
            }                           // Done with phase 1
            else                        // Else phase 2
            {                           //
              if (idx>=nbits)           // Not enough bits left?
                break;                  // Done processing
              r2=(*ch)[idx]&1;          // Get parity bit 2
              w2=std::clamp((*w)[idx],0.f,1.f);// Get weight for parity bit 2
              idx++;                    // Advance index
              hp1=0;                    // Parity bit 1 not received
              hp2=1;                    // Mark parity bit 2 received
              if (lg)
                lg->Deb(" DecodeBitsWeighted: Rate 3/4, ph=%d r2=%d w2=%.2f",ph,r2,w2);
            }                           // Done with phase 2
          }                             // Done with 3/4 punctured code
          ViterbiStepWeighted(r1,r2,hp1,hp2,w1,w2);// Perform one Viterbi time step
          if (ccfg.p34)                 // Rate 3/4?
            ph=(ph+1)%3;                // Advance puncturing phase
        }                               // Done processing channel bits
        std::vector<uint8_t> urev{};    // Reversed input bits
        Traceback(&urev);               // Perform traceback
        o->assign(urev.rbegin(),urev.rend());// Assign in forward order
        if (lg)
          lg->Inf("[END] Viterbi DecodeBitsWeighted: output input bits=%zu",o->size());
      }
    private:
      ConvConfig ccfg{};                // Convolutional Codec conf parameters
      int ph{0};                        // Puncturing phase counter
      static constexpr uint16_t INF=0x3FFF;// big and safe for short frames
      static constexpr uint16_t BM_SCALE=256; // Branch metric scaling factor
      uint16_t pm[64]{};                 // Path metrics
      size_t steps{0};                  // Number of steps Viterbi has gone through
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Survivor paths: one row per step, 64 colums (state)
      // Each row stores predecessor state and the input bit taken to enter that state
      // and lead us to that path at that step.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      std::vector<std::array<uint8_t,64>> prev{}; // Survivor paths
      std::vector<std::array<uint8_t,64>> bit{};  // Input bits
      // Log object
      std::unique_ptr<logx::Logger> lg{}; // Logger instance
      // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Trellis structure for Viterbi Decoder
      // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
      struct TrellisNode
      {
        private:
        uint8_t state{0};         // Current state
        uint8_t input{0};         // Input bit (0,1)
        uint8_t next[64][2];      // Next state for each input bit (0,1)
        uint8_t out[64][2][2];       // Output parity bits for each input bit (0,1)
        public:
        TrellisNode (void)=default;
        ~TrellisNode (void)=default;
        inline void Clear (void)
        {                         // ~~~~~~~~~~ Clear ~~~~~~~~~~ //
          for (int i=0;i<64;i++)  // For each state
          {                       //
            next[i][0]=0;         // Clear next state for input 0
            next[i][1]=0;         // Clear next state for input 1
            for (int j=0;j<2;j++)   // For each input bit
            {                     //
              out[i][j][0]=0;     // Clear output parity bit 1
              out[i][j][1]=0;     // Clear output parity bit 2
            }                     // Done for each input bit
          }                       //
        }                         // ~~~~~~~~~~ Clear ~~~~~~~~~~ //
        inline void Set (
          int state,              // Current state
          int input,              // Input bit (0,1)
          int nextstate,          // Next state
          int outbits)            // Output parity bits
        {                         // ~~~~~~~~~~ Set ~~~~~~~~~~ //
          next[state][input]=static_cast<uint8_t>(nextstate);// Set next state
          out[state][input][0]=static_cast<uint8_t>(outbits);  // Set output parity bits
        }                         // ~~~~~~~~~~ Set ~~~~~~~~~~ //
        inline void Get (
          int state,              // Current state
          int input,              // Input bit (0,1)
          int& nextstate,         // Next state (output)
          int& outbits) const     // Output parity bits (output)
        {                         // ~~~~~~~~~~ Get ~~~~~~~~~~ //
          nextstate=next[state][input];// Get next state
          outbits=out[state][input][0];  // Get output parity bits
        }                         // ~~~~~~~~~~ Get ~~~~~~~~~~ //
        inline uint8_t* GetNext (void)
        {
          return &this->next[0][0];
        }
        inline uint8_t* GetOut (void)
        {
          return &this->out[0][0][0];
        }
        inline uint8_t* GetNext (int state, int input)
        {
          return &this->next[state][input];
        }
        inline uint8_t* GetOut (int state, int input, int bit)
        {
          return &this->out[state][input][bit];
        }
        inline uint8_t GetCurrentState (void) const
        {
          return this->state;
        }
        inline void SetCurrentState (int state)
        {
          this->state=static_cast<uint8_t>(state);
        }
        inline uint8_t GetInputBit (void) const
        {
          return this->input;
        }
        inline void SetInputBit (int input)
        {
          this->input=static_cast<uint8_t>(input);
        }
        inline uint8_t Parity (uint32_t bs) // Bit sequence to parity check
        {                               // ~~~~~~~~~~ Parity ~~~~~~~~~~ //                                
          bs^=bs>>16;                   // Fold upper bits
          bs^=bs>>8;                    // Fold again
          bs^=bs>>4;                    // Fold again
          bs^=bs>>2;                    // Fold again
          bs^=bs>>1;                    // Fold again
          return static_cast<uint8_t>(bs&0x1);// Return parity bit
        }                               // ~~~~~~~~~~ Parity ~~~~~~~~~~ //
        inline void BuildTrellis (logx::Logger* lg)
        {                               // ~~~~~~~~~~~~ BuildTrellis ~~~~~~~~~~~~ //
          Clear();                      // Clear existing trellis
          const uint32_t g1=0b1111001;  // Generator 1
          const uint32_t g2=0b1011011;  // Generator 2
          for (uint8_t s=0;s<64;++s)    // For each state
          {                             // and for each input bit
            for (uint8_t b=0;b<2;++b)   // and for each input bit.....
            {                           // Compute Trellis diagram
              const int zprm=((s<<1)|b)&0x7F; // Shift register with input bit
              next[s][b]=zprm&0x3F;     // Next state (6 LSBs)
              out[s][b][0]=Parity(static_cast<uint32_t>(zprm)&g1);// Output parity bit 1
              out[s][b][1]=Parity(static_cast<uint32_t>(zprm)&g2);// Output parity bit 2
              if (lg!=nullptr)
                lg->Deb(" BuildTrellis: State %02X Input %d -> Next %02X Output [%d,%d]",
                  s,b,next[s][b],out[s][b][0],out[s][b][1]);
            }                           // Done for each input bit
          }                             // Done for each state
        }                               // ~~~~~~~~~~~~ BuildTrellis ~~~~~~~~~~~~ //
        inline void Print (logx::Logger* lg) const
        {                               // ~~~~~~~~~~ Print ~~~~~~~~~~ //
          for (uint8_t s=0;s<64;++s)    // For each state
          {                             //
            for (uint8_t b=0;b<2;++b)   // For each input bit
            {                           //
              uint8_t ns=next[s][b];    // Next state
              uint8_t ob1=out[s][b][0]; // Output bit 1
              uint8_t ob2=out[s][b][1]; // Output bit 2
              std::printf(" State %02X Input %d -> Next %02X Output [%d,%d]\n",
                s,b,ns,ob1,ob2);
              if (lg!=nullptr)
                lg->Inf(" State %02X Input %d -> Next %02X Output [%d,%d]",
                  s,b,ns,ob1,ob2);
            }                           // Done for each input bit
          }                             // Done for each state
        }                               // ~~~~~~~~~~ Print ~~~~~~~~~~ //        
      } trellis{};
      // One ACS (Add path weights, Compare, Select best path) step
      inline void ViterbiStep (
        uint8_t r1,                     // Received parity bit 1
        uint8_t r2,                     // Received parity bit 2
        uint8_t isp1,                   // Do we have parity bit 1?
        uint8_t isp2)                   // Do we have parity bit 2?
      {                                 // ~~~~~~~~~~ ViterbiStep ~~~~~~~~~~ //
        if (lg)
          lg->Deb(" [BEGIN] ViterbiStep: r1=%d r2=%d isp1=%d isp2=%d",r1,r2,isp1,isp2);
        // Get new path metrics
        uint16_t qm[64]{};              // New path metrics
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Store survivors for this step
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t ps[64];                 // Predecessor state (prev row)
        uint8_t ib[64];                 // Input bit that led to this survivor state.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Init new path metrics to Inf
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int s=0;s<64;++s)          // For each state
          qm[s]=INF;                    // Init to Inf
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // For each state, compute branch metrics for input bits 0 and 1
        // and update path metrics accordingly
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int s=0;s<64;++s)          // For each state
        {                               // Viterbi Add path weigths, compare and select.....
          const uint64_t base=pm[s];    // Base path metric for this state
          if (base>=INF)                // Invalid path?
            continue;                   // Skip it
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Try two possibilities: input bit 0 and 1 and their respective outputs
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          for (int b=0;b<2;++b)         // For each input bit
          {                             //
            const int ns=static_cast<int>(trellis.GetNext(s,b)[0]); // Next state
            const uint8_t* e1=trellis.GetOut(s,b,0); // Output bits for this input
            const uint8_t* e2=trellis.GetOut(s,b,1); // Output bits for this input
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Compute Hamming distance (branch metric) from received bits to
            // expected output bits for this transition
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            uint8_t bm{0};              // Branch metric
            if (isp1!=0)                // Parity bit 1 present?
              bm+=(*e1!=r1);            // Done computing branch metric
            if (isp2!=0)                // Parity bit 2 present?
              bm+=(*e2!=r2);            // Done computing branch metric
            if (lg)
              lg->Deb(" ViterbiStep: State %02X Input %d -> Next %02X Expected [%d,%d] Received [%d,%d] BM=%d",
                s,b,ns,*e1,*e2,
                (isp1!=0)? r1:0xFF,
                (isp2!=0)? r2:0xFF,
                bm);
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Compute new path metric for this transition
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            const uint16_t npm=static_cast<uint16_t>(base+bm);// New path metric
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Compare with existing path metric for next state
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (npm<qm[ns])             // Better path?
            {                           // Yes, so update optimal path
              qm[ns]=npm;               // Update path metric
              ps[ns]=static_cast<uint8_t>(s);// Store predecessor state
              ib[ns]=static_cast<uint8_t>(b);// Store input bit that led to this state
              if (lg)
                lg->Deb(" ViterbiStep: Update State %02X New PM=%d Prev State %02X Input Bit %d",
                  ns,npm,s,b);
            }                           // Done updating optimal path
          }                             // Done for each input bit
        }                               // Done for each state
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Commit new metrics and push survivors for traceback
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int s=0;s<64;++s)          // For each state
          pm[s]=qm[s];                  // Commit new path metric
        prev.emplace_back();            // Add new row for predecessor states
        bit.emplace_back();             // Add new row for input bits
        for (int s=0;s<64;++s)          // For each state
        {                               // Store last step survivors and input bits
          prev.back()[s]=ps[s];         // Predecessor state
          bit.back()[s]=ib[s];          // Input bit that led to this state
        }                               // Done for each state
        ++steps;                        // Advance number of steps
        if (lg)
          lg->Deb(" [END] ViterbiStep: Completed step %zu",steps);
      }                                 // ~~~~~~~~~~ ViterbiStep ~~~~~~~~~~ //
      inline void ViterbiStepWeighted (
        uint8_t r1,                     // Received parity bit 1
        uint8_t r2,                     // Received parity bit 2
        uint8_t isp1,                   // Do we have parity bit 1?
        uint8_t isp2,                   // Do we have parity bit 2?
        float w1,                       // Weight for parity bit 1
        float w2)                       // Weight for parity bit 2
      {                                 // ~~~~~~~~~~ ViterbiStepWeighted ~~~~~~~~~~ //
        if (lg)
          lg->Deb(" [BEGIN] ViterbiStepWeighted: r1=%d r2=%d isp1=%d isp2=%d w1=%.2f w2=%.2f",
            r1,r2,isp1,isp2,w1,w2);
        uint16_t qm[64];                // New path metrics
        std::fill(std::begin(qm),std::end(qm),INF);// Init to Inf
        uint8_t ps[64]{};               // Predecessor state (prev row)
        uint8_t ib[64]{};               // Input bit that led to this survivor state.
        const uint16_t w1s=static_cast<uint16_t>(std::clamp(w1,0.f,1.f)*BM_SCALE+0.5f);// Scaled weight 1
        const uint16_t w2s=static_cast<uint16_t>(std::clamp(w2,0.f,1.f)*BM_SCALE+0.5f);// Scaled weight
        for (int s=0;s<64;++s)          // For each state...
        {
          const uint16_t base=pm[s];    // Base path metric for this state
          if (base>=INF)                // Invalid path?
            continue;                   // Skip it
          for (int b=0;b<2;++b)         // For each input bit
          {
            const int ns=static_cast<int>(trellis.GetNext(s,b)[0]);// Next state
            const uint8_t* e1=trellis.GetOut(s,b,0);// Output bits for this input
            const uint8_t* e2=trellis.GetOut(s,b,1);// Output bits for this input
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Compute weighted branch metric from received bits to expected output bits
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            uint16_t bm{0};             // Branch metric
            if (isp1!=0)                // Parity bit present?
            bm+=((*e1!=r1)?w1s:0);      // Add weighted branch metric
            if (isp2!=0)                // Parity bit present?
              bm+=((*e2!=r2)?w2s:0);    // Add weighted branch metric
            const uint16_t npm=static_cast<uint16_t>(std::min<uint32_t>(INF,base+bm));// New path metric
            if (npm<qm[ns])             // Better path?
            {                           // Yes, so update optimal path
              qm[ns]=npm;               // Update path metric
              ps[ns]=static_cast<uint8_t>(s);// Store predecessor state
              ib[ns]=static_cast<uint8_t>(b);// Store input bit that led to this state
              if (lg)
                lg->Deb(" ViterbiStepWeighted: Update State %02X New PM=%d Prev State %02X Input Bit %d",
                  ns,npm,s,b);
            }                           // Done updating optimal path
          }                             // Done for each input bit
        }                               // Done for each state
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Commit new metrics and push survivors for traceback
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int s=0;s<64;++s)          // For each state
          pm[s]=qm[s];                  // Commit new path metric
        prev.emplace_back();            // Add new row for predecessor states
        bit.emplace_back();             // Add new row for input bits
        for (int s=0;s<64;++s)          // For each state
        {                               // Store last step survivors and input bits
          prev.back()[s]=ps[s];         // Predecessor state
          bit.back()[s]=ib[s];          // Input bit that led to this state
        }                               // Done for each state
        ++steps;                        // Advance number of steps
        if (lg)
          lg->Deb(" [END] ViterbiStepWeighted: Completed step %zu",steps);
      }                                 // ~~~~~~~~~~ ViterbiStepWeighted ~~~~~~~~~~ //
      inline void Traceback (
        std::vector<uint8_t>* const urev) // Reversed output bits
      {                                 // ~~~~~~~~~~ Traceback ~~~~~~~~~~~~ //
        if (urev==nullptr||steps==0)              // Bad arg?
          return;                       // Nothing to do
        if (lg)
          lg->Inf("[BEGIN] Viterbi Traceback");
        urev->clear();                  // Clear output buffer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Find state with best path metric
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint16_t bstm=INF;              // Best path metric
        int bsts{0};                    // State with best path metric
        for (int s=0;s<64;++s)          // For each state
        {                               // Find best path metric
          if (pm[s]<bstm)               // Do we have a better path metric?
          {                             // Yes
            bstm=pm[s];                 // Update best path metric
            bsts=s;                     // Update best state
          }                             // Done checking this state
        }                               // Done for all states
        if (lg)
          lg->Inf(" Viterbi Traceback: Best State=%02X Best Path Metric=%d",bsts,bstm);
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Traceback through survivor paths
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        const int TB{30};               // Traceback depth
        int tbstps=(steps>TB)?(steps-TB):0;// Traceback start step
        int ste=bsts;                   // Best state to start traceback
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Perform a walkback through all steps, but emit all decisions
        // only after reaching the traceback depth.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int t=static_cast<int>(steps)-1;t>=0;--t)// For each step in diagram....
        {                               // Trace backwards....
          const uint8_t u=bit[static_cast<size_t>(t)][ste];// Input bit that led to this state
          const int ps=prev[static_cast<size_t>(t)][ste];// Predecessor state
          if (t<tbstps)                 // Beyond traceback depth?
            urev->push_back(u);         // Yes, so keep reliable tail first
          else                          // Else not yet at traceback depth
            urev->push_back(u);         // Keep all bits 
          ste=ps;                       // Move to predecessor state
          if (lg)
            lg->Deb(" [END] Viterbi Traceback: Step %d Current State %02X Input Bit %d Predecessor State %02X",
              t,ste,u,ps);
        }                               // Done for all steps
      }

  };
}
