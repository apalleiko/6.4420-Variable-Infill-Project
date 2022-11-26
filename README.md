# 6.4420-Variable-Infill-Project
6.4420 Final Project: FEA Informed Variable FDM Infill
Andrew Palleiko, Andrew Johnson, Dylan Ryan

### Installing Matlab Engine in Python enviornment:
[Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- Download Matlab R2022b on machine
- Enter 'matlabroot' in command prompt on matlab. Copy output
- Navigate to base directory in terminal, and enter cd "<matlabroot>\extern\engines\python"
- The install the engine: python -m pip install matlabengine

### Running slicer with FEA
- navigate to mandoline directory
- Enter mandoline <stl path> --fea
- stl file must be in stl_files folder
- output of matlab FEA scripts will land in fea_output.mat, and fea python class will handle conversion to python friendly data structure
