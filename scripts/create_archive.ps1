$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archiveDirectory = Join-Path $root "Archive"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archivePath = Join-Path $archiveDirectory "scout-$timestamp.zip"
$staging = Join-Path ([IO.Path]::GetTempPath()) "scout-archive-$timestamp"
$excludedDirectories = @(
    ".venv", "build", "dist", "models", "data",
    "__pycache__", ".git", ".copilot", "Archive"
)

New-Item -ItemType Directory -Path $archiveDirectory -Force | Out-Null
Set-Location $root

$files = @(Get-ChildItem -Recurse -File | Where-Object {
    $relative = $_.FullName.Substring($root.Length + 1)
    $parts = $relative -split "[\\/]"
    $excluded = $parts | Where-Object { $excludedDirectories -contains $_ }
    -not $excluded -and $_.Extension -ne ".pyc" -and $_.Name -notlike "*.egg-info"
} | ForEach-Object {
    $_.FullName.Substring($root.Length + 1)
})

try {
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    foreach ($relative in $files) {
        $destination = Join-Path $staging $relative
        New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $root $relative) -Destination $destination
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $archivePath -Force
}
finally {
    if (Test-Path $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}

Write-Output "Archive created: $archivePath"
