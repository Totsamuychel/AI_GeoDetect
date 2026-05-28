# =============================================================================
# pack_for_runpod.ps1 — Пакування датасету та коду для завантаження на RunPod
#
# Створює два архіви у папці runpod_upload/:
#   diploma_dataset.tar.gz  — тільки потрібні зображення + маніфести (~2.6 GB)
#   diploma_code.zip        — код, конфіги, скрипти, requirements (~2 MB)
#
# Запуск з кореня проекту:
#   powershell -ExecutionPolicy Bypass -File scripts/pack_for_runpod.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$outDir = "$root\runpod_upload"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# ── 1. Пакування датасету (тільки mapillary + manifests) ─────────────────────
$datasetArchive = "$outDir\diploma_dataset.tar.gz"
Write-Host "Пакування датасету → $datasetArchive"
Write-Host "  (mapillary images ~2.6 GB + manifests — це займе кілька хвилин)"

# tar через git bash (є на Windows з Git)
$tarArgs = @(
    "-czf", $datasetArchive,
    "dataset/manifests",
    "dataset/raw/mapillary"
)
& tar @tarArgs
if ($LASTEXITCODE -ne 0) {
    # Fallback: спробувати через WSL або 7-Zip
    Write-Host "  tar не знайдено — спробую через 7-Zip..."
    $sevenZip = "C:\Program Files\7-Zip\7z.exe"
    if (Test-Path $sevenZip) {
        & $sevenZip a -ttar "$outDir\diploma_dataset.tar" "dataset\manifests" "dataset\raw\mapillary"
        & $sevenZip a -tgzip $datasetArchive "$outDir\diploma_dataset.tar"
        Remove-Item "$outDir\diploma_dataset.tar"
    } else {
        Write-Error "Встановіть Git for Windows або 7-Zip для створення tar.gz"
    }
}

$sizeMB = [math]::Round((Get-Item $datasetArchive).Length / 1MB)
Write-Host "  OK: diploma_dataset.tar.gz ($sizeMB MB)"

# ── 2. Пакування коду ────────────────────────────────────────────────────────
$codeArchive = "$outDir\diploma_code.zip"
Write-Host ""
Write-Host "Пакування коду → $codeArchive"

$include = @(
    "code",
    "configs",
    "scripts",
    "tests",
    "requirements.txt",
    "environment.yml"
)

if (Test-Path $codeArchive) { Remove-Item $codeArchive }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($codeArchive, 'Create')

foreach ($item in $include) {
    $fullPath = Join-Path $root $item
    if (-not (Test-Path $fullPath)) { continue }

    if ((Get-Item $fullPath).PSIsContainer) {
        Get-ChildItem $fullPath -Recurse -File | Where-Object {
            $_.Extension -notin @('.pyc', '.pyo') -and
            $_.Name -ne '.DS_Store' -and
            $_.FullName -notmatch '__pycache__'
        } | ForEach-Object {
            $relative = $_.FullName.Substring($root.Length + 1).Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $relative) | Out-Null
        }
    } else {
        $relative = $item.Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $fullPath, $relative) | Out-Null
    }
}
$zip.Dispose()

$sizeKB = [math]::Round((Get-Item $codeArchive).Length / 1KB)
Write-Host "  OK: diploma_code.zip ($sizeKB KB)"

# ── 3. Підсумок ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=============================================="
Write-Host "Готово! Файли у: $outDir"
Write-Host ""
Write-Host "Наступні кроки:"
Write-Host "  1. Завантажити обидва архіви на RunPod:"
Write-Host "     scp runpod_upload/* root@<pod-ip>:/workspace/"
Write-Host ""
Write-Host "  2. На RunPod розпакувати:"
Write-Host "     cd /workspace"
Write-Host "     unzip diploma_code.zip -d diploma"
Write-Host "     cd diploma"
Write-Host "     tar -xzf /workspace/diploma_dataset.tar.gz"
Write-Host ""
Write-Host "  3. Налаштувати і запустити:"
Write-Host "     bash scripts/runpod_setup.sh"
Write-Host "     bash scripts/runpod_train_all.sh"
Write-Host "=============================================="
