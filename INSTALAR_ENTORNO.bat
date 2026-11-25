@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if errorlevel 1 goto folder_error
if not exist "requirements.txt" goto missing_requirements

echo.
echo Preparacion del entorno - Proyecto de radiografias de torax
echo Esta operacion descarga las bibliotecas y necesita Internet.
echo Se prepara Python para CPU dentro de la carpeta .venv del proyecto.
echo.

if exist ".venv\Scripts\python.exe" goto validate_environment
if exist ".venv" goto incomplete_environment

py -3.12 --version
if errorlevel 1 goto missing_python
py -3.12 -c "import sys,struct; sys.exit(0 if struct.calcsize('P') == 8 else 1)"
if errorlevel 1 goto missing_python

set "INSTALL_STAGE=Creacion del entorno virtual"
py -3.12 -m venv .venv
if errorlevel 1 goto installation_failed

:validate_environment
".venv\Scripts\python.exe" -c "import sys,struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 goto incompatible_environment

set "INSTALL_STAGE=Actualizacion de pip"
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto installation_failed

set "INSTALL_STAGE=Instalacion de PyTorch para CPU"
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto installation_failed

set "INSTALL_STAGE=Instalacion de requirements.txt"
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto installation_failed

set "INSTALL_STAGE=Comprobacion de importacion de las bibliotecas"
".venv\Scripts\python.exe" -c "import torch, torchvision, streamlit, PIL, numpy, pandas; print('Bibliotecas cargadas. PyTorch:', torch.__version__, '| Streamlit:', streamlit.__version__, '| Pillow:', PIL.__version__)"
if errorlevel 1 goto installation_failed

echo.
echo INSTALACION COMPLETADA.
echo Ahora ejecuta INICIAR_PAGINA.bat y abre http://localhost:8501.
echo Para la comprobacion opcional de los 16 casos, consulta GUIA_INICIO.html.
echo No necesitas repetir esta instalacion cada vez que abras la pagina.
pause
endlocal & exit /b 0

:missing_python
echo.
echo Se necesita Python 3.12 de 64 bits y el lanzador py de Windows.
echo Consulta https://www.python.org/downloads/windows/
echo Instala esa version, abre una terminal nueva y ejecuta este archivo otra vez.
echo La extension de Python de VS Code no sustituye al interprete.
pause
endlocal & exit /b 2

:incomplete_environment
echo.
echo Ya existe .venv, pero falta su ejecutable de Python.
echo El instalador no borrara esa carpeta automaticamente.
echo Revisa su contenido o instala el proyecto en una carpeta nueva.
pause
endlocal & exit /b 2

:incompatible_environment
echo.
echo La .venv existente no es Python 3.12 de 64 bits o no puede iniciarse.
echo El instalador no la reemplazara automaticamente.
echo Conserva lo que necesites y crea el proyecto en una carpeta nueva.
echo Consulta GUIA_INICIO.html.
pause
endlocal & exit /b 2

:installation_failed
set "FAIL_CODE=%errorlevel%"
echo.
echo No se pudo completar: %INSTALL_STAGE%.
echo Revisa el mensaje de error anterior y tu conexion a Internet.
echo Puedes volver a ejecutar el instalador para continuar cuando se resuelva.
pause
endlocal & exit /b %FAIL_CODE%

:missing_requirements
echo Falta requirements.txt. Extrae el ZIP completo dentro de neumonia_app.
pause
endlocal & exit /b 2

:folder_error
echo No se pudo abrir la carpeta del proyecto.
pause
endlocal & exit /b 2
