:: Skip LibTorch tests when building a GPU binary and testing on a CPU machine
:: because LibTorch tests are not well designed for this use case.
if "%USE_CUDA%" == "0" IF NOT "%CUDA_VERSION%" == "cpu" exit /b 0

call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
if errorlevel 1 exit /b 1

:: Save the current working directory so that we can go back there
set CWD=%cd%

set CPP_TESTS_DIR=%TMP_DIR_WIN%\build\torch\bin
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%

set TORCH_CPP_TEST_MNIST_PATH=%CWD%\test\cpp\api\mnist
python tools\download_mnist.py --quiet -d %TORCH_CPP_TEST_MNIST_PATH%

python test\run_test.py --cpp --verbose -i cpp/test_api
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
for /r "." %%a in (*.exe) do (
    call :libtorch_check "%%~na" "%%~fa"
    if errorlevel 1 exit /b 1
)

goto :eof

:libtorch_check

cd %CWD%
set CPP_TESTS_DIR=%TMP_DIR_WIN%\build\torch\test

:: Skip verify_api_visibility as it a compile level test
if "%~1" == "verify_api_visibility" goto :eof

:: See https://github.com/pytorch/pytorch/issues/25161
if "%~1" == "c10_metaprogramming_test" goto :eof
if "%~1" == "module_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/25312
if "%~1" == "converter_nomigraph_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35636
if "%~1" == "generate_proposals_op_gpu_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35648
if "%~1" == "reshape_op_gpu_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35651
if "%~1" == "utility_ops_gpu_test" goto :eof

echo Running "%~2"
if "%~1" == "c10_intrusive_ptr_benchmark" (
  :: NB: This is not a gtest executable file, thus couldn't be handled by pytest-cpp
  call "%~2"
  goto :eof
)

python test\run_test.py --cpp --verbose -i "cpp/%~1"
if errorlevel 1 (
  echo %1 failed with exit code %errorlevel%
  exit /b 1
)
if not errorlevel 0 (
  echo %1 failed with exit code %errorlevel%
  exit /b 1
)

goto :eof
