# 将 develop/detect640.pt 链接到 edge_jetson/weights（Windows 需管理员或开发者模式才能目录联接）
param(
    [string]$DevelopRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\develop")).Path
)
$weights = Join-Path $PSScriptRoot "..\weights"
New-Item -ItemType Directory -Force -Path $weights | Out-Null
$src = Join-Path $DevelopRoot "detect640.pt"
$dst = Join-Path $weights "detect640.pt"
if (-not (Test-Path $src)) {
    Write-Error "未找到 $src"
    exit 1
}
if (Test-Path $dst) { Remove-Item $dst -Force }
New-Item -ItemType HardLink -Path $dst -Target $src
Write-Host "已链接 $dst -> $src"
Write-Host "启动前可设置: `$env:NANO_MODEL_LOW='$dst'"
