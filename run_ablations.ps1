$modes = @("full", "local_only", "global_only", "fixed_mix", "no_reg")
foreach ($mode in $modes) {
    Write-Host "Running mode: $mode"
    .\venv\Scripts\python.exe train.py --config configs/hsm_synthetic_tiny.yaml --mode $mode
}
Write-Host "Evaluating..."
.\venv\Scripts\python.exe eval.py --results_dir runs/ > final_evaluation_results.txt
Write-Host "Done! Check final_evaluation_results.txt"
