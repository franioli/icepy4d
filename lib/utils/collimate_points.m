function [varargout] = collimate_points(im, varargin)
%collimatePoints
%  points = collimate_points(im, npts=1, axLims=axLims, printPoints=false);
% 
% To do:
% - add labels to points
% - add points in input

%% Prase input
defnpts = Inf;
defprintsOut = false;
defAxLim = {};

p = inputParser;
addRequired(p,'im',@isnumeric);
addParameter(p,'npts',defnpts,@isnumeric)
addParameter(p,'printPoints',defprintsOut,@islogical)
addParameter(p,'axLims',defAxLim)
parse(p,im,varargin{:})

npts  = p.Results.npts;
printPoints  = p.Results.printPoints;
axLims = p.Results.axLims;

%%
p = struct();
fig = figure("Name",'Press Enter to confirm the position of collimated point', 'Units','normalized', 'Position',[0 0 1 1]); 
ax = axes('Parent',fig);
imshow(im); 
hold on; axis on;

if ~isempty(axLims)
    xlim(axLims{1})
    ylim(axLims{2})    
end

ii = 1;
while ishghandle(fig) && ii <= npts
    roi = drawpoint(ax);
    pause;
    p(ii).loc = roi.Position;
    clearvars roi
    ii = ii + 1;
end
close(fig)

points = zeros(length(p),2);
for jj = 1:length(p)
    points(jj,:) = [p(jj).loc];
end

if printPoints
    ptsOutFig = figure("Name",'Collimated points', 'Units','normalized', 'Position',[0 0 1 1]); 
    imshow(im); hold on;
    plot(points(:,1), points(:,2), 'gx');
end

if nargout == 1
    varargout = {points};
else
    varargout = {points};
end

end