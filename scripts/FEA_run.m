clear all
close all
%%
FEA = FEAfunctions;
%%
smodel = FEA.pdeModel('./cube.stl');
%%
mesh = FEA.createMesh(smodel);
%%
E = 4.107E9;
nu = 0.3;
fixedVertices = [];
fixedFaces = [[3,5]];
loadedFaces = [[2]];
Faceforces = [[0;0;-1000]];
loadedVertices = [[]];
Vertexforces = [[]];
[Rs, stressColors] = FEA.applyFEA(smodel,E,nu,fixedVertices,fixedFaces,loadedVertices,Vertexforces,loadedFaces,Faceforces);
%%
layerRange = [0 200];
[layerX,layerY,layerStresses,Nb] = FEA.sliceLayer(smodel,mesh,Rs, stressColors, layerRange,true);
%%
layerHeight = 10;
layers = FEA.generateAllLayers(smodel,mesh,Rs,stressColors,layerHeight,false)