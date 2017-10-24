function Y = euclideanloss(X, c, dzdy)
%EUCLIDEANLOSS  computes euclidean distance between X and c
% To be used mainly as displayed error function for autoencoder training
%in our codes
% Author: Rahaf Aljundi
%
% See the COPYING file.


assert(numel(X) == numel(c));
d = size(X);
assert(all(d == size(c)));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    Y = 1 / 2 * sum(subsref((X - c) .^ 2, substruct('()', {':'}))); 
elseif nargin == 3 && ~isempty(dzdy)
    assert(numel(dzdy) == 1);   
    Y = dzdy * (X - c);     
end

end

