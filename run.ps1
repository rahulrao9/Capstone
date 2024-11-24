# Redirect all output to a log file
$logFile = "C:\PES\Capstone\code\app\daily_tasks.log"
Start-Transcript -Path $logFile

Write-Host "Starting daily tasks at $(Get-Date)"

Write-Host "Running main.py..."
python main.py

Write-Host "Running downlatestimg.ps1..."
.\downlatestimg.ps1

Write-Host "Running img_proc.py..."
python img_proc.py

Write-Host "Running Weeder.py..."
python weeder.py

Write-Host "Uploading to Drive"
python driveUploader.py

Write-Host "All tasks completed at $(Get-Date)"

Stop-Transcript
