include "globals.mzn";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT DATA

int: numBins;
set of int: sBin = 1..numBins;

int: numSlots;
set of int: sSlot  = 1..numSlots;
set of int: sSlot_ = 0..numSlots;

int: numDescriptors;
set of int: sDescriptor = 1..numDescriptors;
array[sDescriptor] of int: weights;

int: chainLen;
set of int: sChain = 1..chainLen;
array[sChain] of sDescriptor: chain;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DECISION VARIABLES AND CONSTRAINTS


array[sChain] of var sBin: placement;
array[sBin] of var sSlot_: occupacy;

constraint forall (bin in sBin)
(
    occupacy[bin] = sum(c in sChain where placement[c]=bin) (weights[chain[c]])
);

constraint forall (bin in sBin)
(    
    occupacy[bin] <= numSlots
);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FUNCTION

var int: reward = sum(bin in sBin) (pow(2,occupacy[bin]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLVE

solve maximize reward;
%solve minimize reward;
%solve satisfy;

output ["\nService = ",show(chain),
        "\nPlacement = ",show(placement),
        "\nReward = ",show(reward),
        "\nOccupacy = ", show(occupacy),
       ];