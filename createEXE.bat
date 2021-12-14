rem Get the location of reikna
FOR /F "tokens=*" %%i IN ('python -c "import importlib; print(importlib.util.find_spec('reikna').submodule_search_locations[0])" ') do set REIKNAPATH=%%i

echo REIKNAPATH = %REIKNAPATH%

rem Create correct pyinstaller spec file

rem Windowed
rem pyi-makespec TDCT_main.py --name "3DCorrelationToolbox-RFI" -i "icons\3DCT_icon.ico" --windowed --hidden-import="skimage.filters.rank.core_cy_3d" --add-data "TDCT_correlation.ui;." --add-data "TDCT_main.ui;." --add-data "icons_rc.py;." --add-data "%REIKNAPATH%;reikna"

rem Not windowed
pyi-makespec TDCT_main.py --name "3DCorrelationToolbox-RFI" -i "icons\3DCT_icon.ico" --hidden-import="skimage.filters.rank.core_cy_3d" --add-data "TDCT_correlation.ui;." --add-data "TDCT_main.ui;." --add-data "icons_rc.py;." --add-data "%REIKNAPATH%;reikna"

pyinstaller --clean --noconfirm 3DCorrelationToolbox-RFI.spec