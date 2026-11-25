@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if errorlevel 1 goto folder_error

if not exist "app.py" goto missing_project
if not exist ".venv\Scripts\python.exe" goto missing_environment

".venv\Scripts\python.exe" -c "import sys,struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 goto incompatible_environment

echo.
echo Clasificacion de radiografias de torax - Grupo 2
echo Abre http://localhost:8501 en tu navegador.
echo Deja esta ventana abierta mientras utilizas la pagina.
echo Para detener el servidor, pulsa Ctrl+C.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
set "APP_EXIT=%errorlevel%"
if "%APP_EXIT%"=="0" goto stopped
if "%APP_EXIT%"=="-1073741510" goto stopped

echo.
echo La aplicacion termino con codigo %APP_EXIT%.
echo Revisa el mensaje anterior. La guia GUIA_INICIO.html contiene ayuda.
echo Si faltan bibliotecas, ejecuta INSTALAR_ENTORNO.bat.
pause
endlocal & exit /b %APP_EXIT%

:stopped
endlocal & exit /b 0

:missing_environment
echo.
echo No se encontro .venv\Scripts\python.exe en esta carpeta.
echo Ejecuta INSTALAR_ENTORNO.bat una vez para preparar esta ubicacion.
echo Si actualizas una version anterior, conserva su .venv en su ruta original.
pause
endlocal & exit /b 2

:incompatible_environment
echo.
echo El entorno de esta carpeta no es Python 3.12 de 64 bits o no puede iniciarse.
echo Revisa GUIA_INICIO.html. No copies entornos .venv entre carpetas o equipos.
pause
endlocal & exit /b 2

:missing_project
echo.
echo Falta app.py. Extrae todo el ZIP y abre este archivo dentro de neumonia_app.
pause
endlocal & exit /b 2

:folder_error
echo No se pudo abrir la carpeta del proyecto.
pause
endlocal & exit /b 2
