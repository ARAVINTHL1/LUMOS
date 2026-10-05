# LUMOS Pipeline - Live Progress Monitor for PowerShell
$host.UI.RawUI.WindowTitle = "LUMOS - Live Training & Segmentation Progress"
Clear-Host

$maskDir = "d:\LUMOS\processed\segmentation\pseudo_labels"
$vizDir  = "d:\LUMOS\processed\segmentation\visualizations"
$total   = 1600

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   LUMOS Pipeline - Live Segmentation Progress Monitor     " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Monitoring target: $total images (800 patients x 2 views)`n"

while ($true) {
    if (Test-Path $maskDir) {
        $masks = (Get-ChildItem -Path $maskDir -Filter "*.png" -ErrorAction SilentlyContinue).Count
    } else {
        $masks = 0
    }

    if (Test-Path $vizDir) {
        $viz = (Get-ChildItem -Path $vizDir -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    } else {
        $viz = 0
    }

    $pct = [math]::Round(($masks / $total) * 100, 2)
    $remaining = $total - $masks

    # Display progress
    Write-Progress -Activity "Generating SAM Vertebral Segmentation Masks" `
                   -Status "$masks / $total completed ($pct%)" `
                   -PercentComplete $pct

    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] Completed: $masks / $total ($pct%) | Remaining: $remaining | Visualizations: $viz" -ForegroundColor Green

    if ($masks -ge $total) {
        Write-Host "`nAll $total images successfully segmented!" -ForegroundColor Yellow
        break
    }

    Start-Sleep -Seconds 4
}
