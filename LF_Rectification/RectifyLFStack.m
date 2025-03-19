
%location of th LF stack
imNameLF='../s2a1d1_LF_1P_1x1_400mA_100Hz_func_500frames_no4AP_2_MMStack_Default.ome.tif';

centFls=[1024,1024]; % Center of the middle microlens

% Estimate transformation on half the LF images
a=imfinfo(imNameLF);
inpIMg=zeros(2048,2048);
lfTrainTmp=zeros(size(a,1),2033,2033,'uint16');
iniIndex=0;

for i=1:floor(size(a,1)/2)
    inpIMg=inpIMg+double(imread(imNameLF,i));    
end

inpIMg=padarray(inpIMg,[19*19 19*19],'replicate');
centImg=centFls+[19*19,19*19];

rIni=[-0.2849137166797637, 19.690246479477306]; % Initial vector indicating next lens (usually doesn't need updating)
nCent=41; % Number of microlens to be detected

vin=rIni; 
vort=[vin(2),-vin(1)]; % Calculate orthogonal vector
inpIMg=max(inpIMg,0); % Ensure image values are non-negative
inpIMg=inpIMg/max(inpIMg(:)); % Normalize image intensity values to [0, 1]
inpIMgThres=inpIMg(901:end-900,901:end-900); % Extract central part of the image for thresholding
thres=prctile(inpIMgThres(:),36); % Calculate intensity threshold as the 36th percentile

% Estimate coordinates and vectors for the X direction
[parL1,newVeX,dotPrNwVPntsX]=est1CoordC(centImg,vin,nCent,inpIMg,thres);
% Estimate coordinates and vectors for the Y (orthogonal) direction
[parL2,newVeY,dotPrNwVPntsY]=est1CoordC(centImg,vort,nCent,inpIMg,thres);

% Calculate new center point based on the intersection of parL1 and parL2 lines
dotPrNwVCentX=(parL2(2)-parL1(2))/(parL1(1)-parL2(1));
dotPrNwVCentY=dotPrNwVCentX*parL1(1)+parL1(2);
centImg=[dotPrNwVCentY,dotPrNwVCentX]; % Update center image coordinates
A=[newVeX',newVeY']; % Matrix of new vectors

% Calculate corrected center point coordinates in the transformed space
tmpxy=(A'*A)*inv(A)*centImg.';
dotPrNwVCentX=tmpxy(1);
dotPrNwVCentY=tmpxy(2);

% Compute new coordinates for X and Y points in the transformed space
coorPntsNewX=A*inv(A'*A)*[dotPrNwVPntsX,repmat(dotPrNwVCentY,size(dotPrNwVPntsX))]';
coorPntsNewX=coorPntsNewX';
coorPntsNewY=A*inv(A'*A)*[repmat(dotPrNwVCentX,size(dotPrNwVPntsY)),dotPrNwVPntsY]';
coorPntsNewY=coorPntsNewY';
% Set the central microlens coordinates to the calculated center image coordinates
coorPntsNewX(ceil(nCent/2),:)=centImg;
coorPntsNewY(ceil(nCent/2),:)=centImg;

% Calculate period in the X direction, removing outliers
periodx=diff(dotPrNwVPntsY);
absDif=abs(periodx-mean(periodx));
periodx=mean(periodx(absDif<(quantile(absDif,0.7)))); % Delete outliers

% Calculate period in the Y direction, removing outliers
periody=diff(dotPrNwVPntsX);
absDif=abs(periody-mean(periody));
periody=mean(periody(absDif<(quantile(absDif,0.7)))); % Delete outliers


% Apply the found transformation on the rest of the images
for lfIndexes=1:size(a,1)
    rawImg=double(imread(imNameLF,lfIndexes));
    rawImg=padarray(rawImg,[19*19 19*19],'replicate');
    inpIMRec=rawImg;
    centImgForR=centImg;

    % c1=1;
    % c2=periody/periodx;

    %make new period in both dimnsions equal to 19
    c1=19/periody;
    c2=19/periodx;

    VD=[0,c1;c2,0];
    VG=[newVeX',newVeY'];
    T=VD/VG;
    Te=[T(end:-1:1,end:-1:1).',[0;0]];
    Te=[Te;[0,0,1]];
    tform = affine2d(Te);
    [J,RA] = imwarp(inpIMRec,tform,'SmoothEdges',false,'FillValues',0.58*thres);

    centImg2=T*centImgForR';
    centImg2=centImg2-[RA.YWorldLimits(1)-1;RA.XWorldLimits(1)-1];

    % Translation
    sizImg=size(J);
    if mod(sizImg(1),2)==0
        J=padarray(J,[1 0],0,'post');
    end
    if mod(sizImg(2),2)==0
        J=padarray(J,[0 1],0,'post');
    end
    sizImg=size(J);
    centImgIN=ceil((sizImg+1)/2);
    [XImg,YImg]=meshgrid(1:sizImg(2),1:sizImg(1));
    Xtr=XImg+centImg2(2)-centImgIN(2);
    Ytr=YImg+centImg2(1)-centImgIN(1);
    rectFImg=interp2(XImg,YImg,J,Xtr,Ytr,'spline',0);

    coorPntsNewYR=T*coorPntsNewY';
    coorPntsNewYR=coorPntsNewYR';
    coorPntsNewXR=T*coorPntsNewX';
    coorPntsNewXR=coorPntsNewXR';

    coorPntsNewYR=coorPntsNewYR+repmat(-centImg2'+centImgIN-[RA.YWorldLimits(1)-1,RA.XWorldLimits(1)-1],size(coorPntsNewY,1),1);
    coorPntsNewXR=coorPntsNewXR+repmat(-centImg2'+centImgIN-[RA.YWorldLimits(1)-1,RA.XWorldLimits(1)-1],size(coorPntsNewX,1),1);

    % Crop Image Manually
    halfNewSize=53*19+9; % Here you change how much the image is cropped
    centImR=ceil((size(rectFImg)+1)/2);
    projLF=rectFImg(centImR(1)-halfNewSize:centImR(1)+halfNewSize,centImR(2)-halfNewSize:centImR(2)+halfNewSize);
    
    lfTrainTmp(lfIndexes,:,:)=uint16(projLF);
    disp(lfIndexes)
end
%% Convert to uint8
maxLF=double(max(lfTrainTmp(:)));
for i=1:size(lfTrainTmp,1)
    tmpImg=double(lfTrainTmp(i,:,:));
    lfTrainTmp(i,:,:)=uint8(255*tmpImg/maxLF);
end
lfTrainTmp=uint8(lfTrainTmp);

%% Reshape and Save
lfTrainTmp=reshape(lfTrainTmp,[size(lfTrainTmp,1),19,107,19,107]);
lfTrainTmp=permute(lfTrainTmp,[1,2,4,3,5]);
lfTrainTmp=reshape(lfTrainTmp,[size(lfTrainTmp,1),361,107,107]);
% save('lfTmps2a1d1.mat','lfTrainTmp','-v7.3');