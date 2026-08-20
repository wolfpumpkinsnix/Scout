$ErrorActionPreference = "Stop"

function Install-Cpu {
    Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
    & uv pip install --reinstall --no-cache-dir llama-cpp-python
    if ($LASTEXITCODE -ne 0) {
        throw "CPU llama-cpp-python installation failed"
    }
    Write-Host "Installed CPU llama-cpp-python backend."
}

function Initialize-VulkanSdk {
    if (-not $env:VULKAN_SDK -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Vulkan SDK not found; installing Khronos Vulkan SDK..."
        & winget install --id KhronosGroup.VulkanSDK --exact --silent `
            --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Vulkan SDK installation failed."
            return $false
        }
    }

    if (-not $env:VULKAN_SDK -and (Test-Path "C:\VulkanSDK")) {
        $sdk = Get-ChildItem "C:\VulkanSDK" -Directory |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($sdk) {
            $env:VULKAN_SDK = $sdk.FullName
        }
    }
    if ($env:VULKAN_SDK -and (Test-Path "$env:VULKAN_SDK\Bin")) {
        $env:Path = "$env:VULKAN_SDK\Bin;$env:Path"
    }
    return [bool]($env:VULKAN_SDK -and (Test-Path "$env:VULKAN_SDK\Bin\glslc.exe"))
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $cudaWheel = if ($env:LLAMA_CUDA_WHEEL) { $env:LLAMA_CUDA_WHEEL } else { "cu124" }
    & uv pip install --reinstall --no-cache-dir llama-cpp-python `
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$cudaWheel"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installed NVIDIA CUDA llama-cpp-python backend ($cudaWheel)."
        exit 0
    }
    Write-Warning "CUDA backend installation failed; trying Vulkan."
}

if (Initialize-VulkanSdk) {
    $env:CMAKE_ARGS = "-DGGML_VULKAN=on"
    & uv pip install --reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installed Vulkan llama-cpp-python backend."
        exit 0
    }
    Write-Warning "Vulkan backend installation failed; using CPU."
} else {
    Write-Warning "Vulkan SDK not found; using CPU."
}

Install-Cpu
