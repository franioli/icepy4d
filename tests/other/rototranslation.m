function [] = rototranslation(cam0, cam1, points)

%%
    %- Collimate or load target coordinates in the first epoch
%     im = cell(1, numCams);    
%     impts0 =  cell(1, numCams);
%     imptsUnd0 =  cell(1, numCams);
%     for jj=1:numCams 
%         im{jj} = readimage(images(jj).imds, 1);
%         impts0{jj} = collimatePoints(im{jj}, 1);
%         imptsUnd0{jj} = undistortPoints(impts0{jj}, cameras(jj).cameraParams);       
%     end   
%     save('imPts_target.mat', 'impts0', "imptsUnd0")
    load('imPts_target.mat')

    epoch = 1;
    wpts0_ = triang_lin_batch( {cameras(1).P{epoch}, cameras(2).P{epoch}}, {imptsUnd0{1}', imptsUnd0{2}'} );
    for i = 1:size(wpts0_,2)
        wpts0_ = triang_nonlin(wpts0_(:,i), ...
             {cameras(1).P{epoch}, cameras(2).P{epoch}}, {imptsUnd0{1}', imptsUnd0{2}'});
    end
    
    % Fix reference system at the first epoch --> by three (or more points)
    wpts0 = cat(1, wpts0_', cameras(1).X0{epoch}', cameras(2).X0{epoch}');

%     X01_meta = [416651.52489669225	5091109.91215075	1858.908434299682]';    % IMG_2092
%     X02_meta = [416622.27552777925	5091364.507128085	1902.4053286545502]';   % IMG_0481
%     camWorldBaseline = norm(X01_meta - X02_meta);                               % [m] From Metashape model at epoch t0
%     X01w = [0, 0, 0]'; 
%     x02w = [0, -camWorldBaseline, 0]';    
%     wpts0 = cat(1, wpts0_', cameras(1).X0{epoch}', cameras(2).X0{epoch}');

%% Fast Template Matching - Define template

tWidth      = 8;
saWidth     = 40;

tcenter = cell(1, numCams);  
template = cell(1, numCams);    
for jj = 1: numCams
    im{jj} = readimage(images(jj).imds, 1);
    tcenter{jj}     = round(impts0{jj});
    tlim            = [tcenter{jj}(1)-tWidth, tcenter{jj}(2)-tWidth, 2*tWidth, 2*tWidth];
%     template{jj}   = im2double(rgb2gray(imcrop(im{jj}, tlim)));
    template{jj}   = im2double(imsharpen(rgb2gray(imcrop(im{jj}, tlim)), 'Radius',0.5,'Amount',2));
end
figure;
subplot(2,2,1); imshow(im{1}); hold on; plot(tcenter{1}(1), tcenter{1}(2), 'xr'); axis on
subplot(2,2,2); imshow(template{1}); hold on; plot(tWidth, tWidth+1, 'xr'); axis on
subplot(2,2,3); imshow(im{2});  hold on; plot(tcenter{2}(1), tcenter{2}(2), 'xr'); axis on
subplot(2,2,4); imshow(template{2}); hold on; plot(tWidth+1, tWidth+1, 'xr'); axis on


%% Run FAST template match on all epochs

searchRange.minScale = 0.7;
searchRange.maxScale = 1.3;
searchRange.minRotation = -pi/8;
searchRange.maxRotation = pi/8;
searchRange.minTx = -saWidth/2;
searchRange.maxTx = saWidth/2;
searchRange.minTy = -saWidth/2;
searchRange.maxTy = saWidth/2;

target_est = {length(epoches2process),numCams};
for epoch = 2:epoches2process(end)+1
    SA ={1, numCams};
    for jj = 1:numCams 
        im{jj} = readimage(images(jj).imds, epoch);        
        salim = [tcenter{jj}(1)-saWidth, tcenter{jj}(2)-saWidth, 2*saWidth, 2*saWidth];
        SA{jj}   = im2double(imsharpen(rgb2gray(imcrop(im{jj}, salim)),'Radius',0.5,'Amount',3));  
     
     %  FastMatch(template,img,[templateMask],[epsilon=0.15],[delta=0.25],[photometricInvariance=false],[searchRange]) 
     %  The img and template should both be of class ''double'' (in the range [0,1])')
        [bestConfig,bestTransMat,sampledError] = FastMatch(template{jj},SA{jj},[], 0.15, 0.25, true, searchRange); %   

%         [optError,fullError,overlapError] = MatchingResult(template{jj},SA{jj},bestTransMat, [], 'test');
        
        %- Retrieve image coordinates of adjusted template center on I2
        [h1,w1,~] = size(template{jj});
        [h2,w2,~] = size(SA{jj});
        r1x = 0.5*(w1-1);
        r1y = 0.5*(h1-1);
        r2x = 0.5*(w2-1);
        r2y = 0.5*(h2-1);
        tx = bestTransMat(1,3);
        ty = bestTransMat(2,3);
        centerpoint = [r1x+1;r1y+1];
        tcenterpoint = bestTransMat(1:2,1:2)*centerpoint ; % + [tx; ty];
        dtcenterpoint = tcenterpoint - [r1x; r1x];
        target_est{epoch, jj} =  tcenter{jj} + [tx; ty]';  %  + dtcenterpoint'

        figure('Name', sprintf('Epoch %i, cam %i', epoch, jj));
        subplot(1,2,1), hold on
        title('Template');
        imshow(template{jj}); axis on
        plot(r1x+1,r1y+1, 'rx');
%         pltx = [txlim(1),txlim(2),txlim(2),txlim(1),txlim(1)];
%         plty = [tylim(1),tylim(1),tylim(2),tylim(2),tylim(1)];
        subplot(1,2,2); imshow(im{jj}); hold on; axis on
        plot(target_est{epoch, jj}(1),target_est{epoch, jj}(2), 'rx');
        xlim([target_est{epoch, jj}(1)-saWidth/4, target_est{epoch, jj}(1)+saWidth/4])
        ylim([target_est{epoch, jj}(2)-saWidth/4, target_est{epoch, jj}(2)+saWidth/4])
        title('Detected point in I2');
    end
end
save('imPts_target_est.mat', 'target_est')


%% Estimate Rigid Body transformation between epochs and register point clouds
% Fix reference system at the first epoch --> by three (or more points)

%- Load FastMatch detected targets
% load('imPts_target_est.mat')
% saWidth = 40;
% 
%- And/Or Manually collimate target on images
% target_est_man =  cell(length(epoches2process), numCams);
% im = {};    
% for jj=1:numCams 
%     for epoch = 2:epoches2process(end)+1
%         im{1} = readimage(images(jj).imds, epoch);
%         axLims = {[target_est{epoch, jj}(1)-saWidth/3, target_est{epoch, jj}(1)+saWidth/3], ...
%                 [target_est{epoch, jj}(2)-saWidth/3, target_est{epoch, jj}(2)+saWidth/3] };
%         target_est_man{epoch, jj} = collimatePoints(im{1}, npts=1, axLims=axLims);
%     end 
% end
% save('imPts_target_man.mat', 'target_est_man')
load('imPts_target_man.mat')


wpts = cell(length(epoches2process),1); 
tform = cell(length(epoches2process),1);
for epoch = 2:epoches2process(end)+1
    imptsUnd =  cell(length(epoches2process), numCams);
    for jj=1:numCams 
%         imptsUnd{jj} = undistortPoints(target_est{epoch, jj}, cameras(jj).cameraParams);   
        imptsUnd{jj} = undistortPoints(target_est_man{epoch, jj}, cameras(jj).cameraParams);               
    end   

    wpts_ = triang_lin_batch( {cameras(1).P{epoch}, cameras(2).P{epoch}}, {imptsUnd{1}', imptsUnd{2}'} );
    for i = 1:size(wpts_,2)
        wpts_ = triang_nonlin(wpts_(:,i), ...
             {cameras(1).P{epoch}, cameras(2).P{epoch}}, {imptsUnd{1}', imptsUnd{2}'});
    end
    
    wpts{epoch} = cat(1, wpts_', cameras(1).X0{epoch}', cameras(2).X0{epoch}');
    tform{epoch} = estimateGeometricTransform3D(wpts{epoch},wpts0,"rigid");
end

fprintf('Translation:\t X [m] \t\t\tY [m] \t\t\tZ [m]\n')
for epoch = 2:epoches2process(end)+1
    fprintf('Epoch %i: \t\t%+.4f \t\t%+.4f \t\t%+.4f\n', epoch-1, tform{epoch}.Translation)
end
fprintf('Rotations:\t omega [deg] \t\tPhi [deg] \t\tKappa [deg]\n')
for epoch = 2:epoches2process(end)+1
    fprintf('Epoch %i: \t\t%+.4f \t\t%+.4f \t\t%+.4f\n', epoch-1, rad2deg(ieul(tform{epoch}.Rotation)))
end

%- Apply transformation to point cloud
for epoch = 2:epoches2process(end)+1
    XYZ = tform{2}.transformPointsForward(sparsePts(epoch).ptCloud.Location);
    sparsePts(epoch).ptCloudReg   = pointCloud(XYZ, "Color", sparsePts(epoch).ptCloud.Color);
    pcwrite(sparsePts(epoch).ptCloudReg, ['ptcloud/sparsepts_t', num2str(epoch-1), '_reg.ply'])
end



%%

end