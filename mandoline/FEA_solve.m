function FEA_solve(smodel,stl_path, fixedVertices,fixedFaces,loadedFaces,faceForces,loadedVertices,vertexForces)
    FEA = FEAfunctions;
    mesh = FEA.createMesh(smodel);
    figure(2)
    pdeplot3D(smodel)
    %%
    E = 4.107E9;
    nu = 0.3;
    fixedVertices = [eval(fixedVertices)]
    fixedFaces = [eval(fixedFaces)]
    loadedVertices = [eval(loadedVertices)]
    vertexForces = eval(vertexForces)
    loadedFaces = [eval(loadedFaces)]
    faceForces = eval(faceForces)
%     fixedVertices = [];
%     fixedFaces = [[5,3]];
%     loadedFaces = [[2]];
%     faceForces = [[0;0;-10]];
%     loadedVertices = [[]];
%     vertexForces = [[]];
    [Rs, stressColors] = FEA.applyFEA(smodel,E,nu,fixedVertices,fixedFaces,loadedVertices,vertexForces,loadedFaces,faceForces);
    figure(3)
    pdeplot3D(smodel,'ColorMapData',Rs.VonMisesStress,'Deformation',Rs.Displacement,'DeformationScaleFactor',100)
    %%
    layerHeight = 0.2;
    layers = FEA.generateAllLayers(smodel,mesh,Rs,stressColors,layerHeight,false);
    %%
    save('../fea_output.mat','layers')
    fprintf('Layers for %s saved to: ../fea_output.mat',stl_path)
end