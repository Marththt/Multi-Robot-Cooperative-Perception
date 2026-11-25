# Every variant of the review, 3 seeds each, then the comparison table (Windows PowerShell). From the repository root:
#     powershell -ExecutionPolicy Bypass -File gnn_review\run_all.ps1
# A stopped run can be restarted: seeds already saved are not trained again.
$ErrorActionPreference = "Continue"   # python warnings on stderr must not stop the run
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results\review | Out-Null
foreach ($v in @("base", "k1", "k3", "nofalse", "logdeg", "world")) {
    python -m gnn_review.train --variant $v --seeds 3 2>&1 | Tee-Object -Append results\review\run_all.log
    if ($LASTEXITCODE -ne 0) { throw "variant $v failed" }
}
python -m gnn_review.summarize | Tee-Object -Append results\review\run_all.log
if ($LASTEXITCODE -ne 0) { throw "summarize failed" }
# send the results back: commit results/review (models, logs, tables) and push to the current branch
git add results/review
git commit -m "results/review: GNN review variants, 3 seeds each (gnn_review/run_all.ps1)"
git pull --rebase
git push
if ($LASTEXITCODE -ne 0) { Write-Host "push failed: run 'git pull --rebase; git push' by hand" }
