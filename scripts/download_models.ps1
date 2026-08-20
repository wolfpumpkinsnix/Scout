param(
    [ValidateSet("embedding", "gemma", "chat", "reranker", "all")]
    [string]$Model = "embedding",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$models = Join-Path $root "models"
New-Item -ItemType Directory -Path $models -Force | Out-Null

$downloads = @{
    embedding = @{
        Name = "Qwen3-Embedding-0.6B-Q8_0.gguf"
        Url = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true"
    }
    chat = @{
        Name = "Qwen3-1.7B-Q4_K_M.gguf"
        Url = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true"
    }
    gemma = @{
        Name = "embeddinggemma-300m-Q4_0.gguf"
        Url = "https://huggingface.co/second-state/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-Q4_0.gguf?download=true"
    }
    reranker = @{
        Name = "qwen3-reranker-0.6b-q8_0.gguf"
        Url = "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf?download=true"
    }
}

$selected = if ($Model -eq "all") {
    $downloads.Values
} else {
    @($downloads[$Model])
}

foreach ($download in $selected) {
    $destination = Join-Path $models $download.Name
    if ((Test-Path $destination) -and -not $Force) {
        Write-Host "Already exists: $($download.Name)"
        continue
    }

    Write-Host "Downloading $($download.Name)..."
    & curl.exe --fail --location --retry 3 --continue-at - `
        --output $destination $download.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($download.Name)"
    }
}

Write-Host "Models directory: $models"
