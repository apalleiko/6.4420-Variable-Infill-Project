classdef FEAfunctions
    methods
        function smodel = pdeModel(obj,stl)
            smodel = createpde('structural','static-solid'); %returns a structural analysis model for the specified analysis type. This model lets you solve small-strain linear elasticity problems.
            importGeometry(smodel,stl);
        end

        function mesh = createMesh(obj,smodel)
            mesh = generateMesh(smodel); %creates a mesh and stores it in the model object. model must contain a geometry
        end   

        function [Rs, stressColors] = applyFEA(obj,smodel,E,nu,fixedVertices,fixedFaces,loadedVertices,Vertexforces,loadedFaces,Faceforces)
            structuralProperties(smodel,'YoungsModulus',E,'PoissonsRatio',nu);
            if ~isempty(fixedVertices)
                structuralBC(smodel,'Vertex',fixedVertices,'Constraint','fixed');
            end
            if ~isempty(fixedFaces)
                structuralBC(smodel,'Face',fixedFaces,'Constraint','fixed');
            end
            for loadIdx = 1:size(loadedFaces,1)
                structuralBoundaryLoad(smodel,'Face',loadedFaces(loadIdx,:),'SurfaceTraction',Faceforces(:,loadIdx));
            end
            for loadIdx = 1:size(loadedVertices,1)
                structuralBoundaryLoad(smodel,'Vertex',loadedVertices(loadIdx,:),'Force',Vertexforces(:,loadIdx));
            end            
            Rs = solve(smodel);
            colorGradient = jet(256);
            stressColorIdxs = ceil(256.*Rs.VonMisesStress./max(Rs.VonMisesStress));
            stressColors = colorGradient(stressColorIdxs,:);
        end

        function [layerX,layerY,layerStresses,Nb] = sliceLayer(obj,smodel,mesh,Rs,stressColors,layerRange,plotOn)
            [mins,minIdx] = min(mesh.Nodes');
            [maxs,maxIdx] = max(mesh.Nodes');
            Nb = findNodes(mesh,"box",[mins(1) maxs(1)],[mins(2) maxs(2)],layerRange);
            layerX = mesh.Nodes(1,Nb).';
            layerY = mesh.Nodes(2,Nb).';
            boundaryIdx = boundary(layerX,layerY);
            layerStresses = Rs.VonMisesStress(Nb);
            layerColors = stressColors(Nb,:);
            if plotOn
                figure
                scatter(layerX,layerY,100,layerColors,'filled')
                hold on
                plot(layerX(boundaryIdx),layerY(boundaryIdx));
                hold off
                figure(5)
                pdemesh(smodel);
                hold on
                plot3(mesh.Nodes(1,Nb),mesh.Nodes(2,Nb),mesh.Nodes(3,Nb),"r.","MarkerSize",25) 
                hold off
            end
        end

        function layers = generateAllLayers(obj,smodel,mesh,Rs,stressColors,layerHeight,plotOn)
            [mins,minIdx] = min(mesh.Nodes');
            [maxs,maxIdx] = max(mesh.Nodes');
            layerRanges = mins(3):layerHeight:maxs(3);
            if any(layerRanges~=maxs(3))
                layerRanges=[layerRanges,maxs(3)];
            end
            layerCount = length(layerRanges);
            layer.X = [];
            layer.Y = [];
            layer.Z = [];
            layer.stresses = [];
            layers = repmat(layer,1,layerCount-1);
            figure
            pastStressColors = [];
            for layerIdx = 1:layerCount-1
                layerRange = [layerRanges(layerIdx), layerRanges(layerIdx+1)];
                [layerX,layerY,layerStresses,Nb] = sliceLayer(obj,smodel,mesh,Rs,stressColors,layerRange,plotOn);
%                 TODO fix when slice range is empty. need to interpolate
%                 above and below
                if isempty(layerX)
                    layers(layerIdx).X = layers(layerIdx-1).X;
                    layers(layerIdx).Y = layers(layerIdx-1).Y;
                    layers(layerIdx).Z = layers(layerIdx-1).Z;
                    layers(layerIdx).stresses = layers(layerIdx-1).stresses;
                    cur_stress_colors = pastStressColors;
                else
                    layerZ = zeros(length(layerX),1) + layerRanges(layerIdx);
                    layers(layerIdx).X = layerX;
                    layers(layerIdx).Y = layerY;
                    layers(layerIdx).Z = layerZ;
                    layers(layerIdx).stresses = layerStresses;
                    cur_stress_colors = stressColors(Nb,:);
                    pastStressColors = cur_stress_colors;
                end
                boundaryIdx = boundary(layers(layerIdx).X,layers(layerIdx).Y);
                scatter3(layers(layerIdx).X,layers(layerIdx).Y,layers(layerIdx).Z,20,cur_stress_colors,'filled')
                hold on
                plot3(layers(layerIdx).X(boundaryIdx),layers(layerIdx).Y(boundaryIdx),layers(layerIdx).Z(boundaryIdx),'-k','LineWidth',3)
                hold on
            end
            hold off
        end        
    end
end
