function smodel = FEA_run_model(stl_path)
    disp('FEA initialized...')
    %%
    FEA = FEAfunctions;
    %%
    smodel = FEA.pdeModel(stl_path);
    figure(1)
    pdegplot(smodel,'VertexLabels','on','FaceLabels','on', 'FaceAlpha',.75)
end