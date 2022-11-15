clear all
close all
addpath 'C:\Users\apall\Documents\MIT Docs\Semester 7 - Fall 2022\6.4420\6.4420-Variable-Infill-Project\FEA-scripts'
%%
FEA = FEAfunctions;
%%
smodel = FEA.pdeModel('../mandoline/test_models/cube.stl');
figure(1)
pdegplot(smodel,'VertexLabels','on','FaceLabels','on', 'FaceAlpha',.75)
%%
mesh = FEA.createMesh(smodel);
figure(2)
pdeplot3D(smodel)
%%
E = 4.107E9;
nu = 0.3;
fixedVertices = [];
fixedFaces = [[5]];
loadedFaces = [[1],[3]];
Faceforces = [[10;0;0],[-10;0;0]];
loadedVertices = [[]];
Vertexforces = [[]];
[Rs, stressColors] = FEA.applyFEA(smodel,E,nu,fixedVertices,fixedFaces,loadedVertices,Vertexforces,loadedFaces,Faceforces);
figure(3)
pdeplot3D(smodel,'ColorMapData',Rs.VonMisesStress,'Deformation',Rs.Displacement,'DeformationScaleFactor',100)
%%
layerHeight = 0.2;
layers = FEA.generateAllLayers(smodel,mesh,Rs,stressColors,layerHeight,false);
%% 
save('layers.mat','layers')