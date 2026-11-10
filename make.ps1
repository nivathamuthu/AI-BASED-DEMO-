# DocIntel Lab task runner. Every target goes through docker compose.
# Usage: .\make.ps1 up
param(
    [Parameter(Position = 0)]
    [string]$Target = "help",
    [Parameter(Position = 1)]
    [string]$Question = ""
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location $PSScriptRoot

function Ensure-EnvFile {
    if (-not (Test-Path -Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example"
    }
}

function Assert-Compose {
    param([string]$Action)
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose $Action failed (exit $LASTEXITCODE)"
    }
}

function Wait-Web {
    Write-Host "Waiting for the web container..."
    for ($i = 0; $i -lt 40; $i++) {
        docker compose exec -T web python -c "print('ready')" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 3
    }
    throw "The web container did not become ready. Check: docker compose logs web"
}

function Invoke-Web {
    param([Parameter(Mandatory = $true)][string[]]$Command)
    docker compose exec -T web @Command
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose exec web $($Command -join ' ') failed (exit $LASTEXITCODE)"
    }
}

switch ($Target) {
    "up" {
        Ensure-EnvFile
        docker compose up -d --build
        Assert-Compose "up"
        Wait-Web
        Invoke-Web @("python", "manage.py", "migrate", "--noinput")
        Write-Host ""
        Write-Host "Ready. Open http://localhost:8010/"
        Write-Host "Next: .\make.ps1 demo"
    }
    "down" {
        docker compose down
        Assert-Compose "down"
    }
    "migrate" {
        Invoke-Web @("python", "manage.py", "migrate", "--noinput")
    }
    "worker" {
        docker compose up -d worker
        Assert-Compose "up worker"
        docker compose logs -f worker
    }
    "web" {
        docker compose up -d web
        Assert-Compose "up web"
        docker compose logs -f web
    }
    "ingest" {
        Invoke-Web @("python", "scripts/seed_demo.py")
    }
    "demo" {
        Ensure-EnvFile
        docker compose up -d
        Assert-Compose "up"
        Wait-Web
        Invoke-Web @("python", "manage.py", "migrate", "--noinput")
        Invoke-Web @("python", "scripts/make_synthetic_docs.py")
        Invoke-Web @("python", "scripts/seed_demo.py")
        Write-Host ""
        Write-Host "Demo is loaded. Open http://localhost:8010/"
    }
    "demo-reset" {
        Ensure-EnvFile
        docker compose up -d
        Assert-Compose "up"
        Wait-Web
        Invoke-Web @("python", "manage.py", "migrate", "--noinput")
        Write-Host "Waiting for the worker..."
        $workerReady = $false
        for ($i = 0; $i -lt 20; $i++) {
            docker compose exec -T worker celery -A config inspect ping --timeout 5 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $workerReady = $true
                break
            }
            Start-Sleep -Seconds 3
        }
        if (-not $workerReady) {
            throw "The worker did not answer. Check: docker compose logs worker"
        }
        Invoke-Web @("python", "scripts/seed_demo.py", "--reset")
        Write-Host ""
        Write-Host "Demo reset complete. Open http://localhost:8010/"
    }
    "eval" {
        Invoke-Web @("python", "manage.py", "eval_prompts")
        Invoke-Web @("python", "manage.py", "eval_rag")
        Invoke-Web @("python", "manage.py", "build_graph", "--extract")
        Invoke-Web @("python", "manage.py", "eval_kg")
        Invoke-Web @("python", "manage.py", "eval_rules")
        Invoke-Web @("python", "manage.py", "eval_agent")
    }
    "extract-rules" {
        Invoke-Web @("python", "manage.py", "extract_rules")
    }
    "eval-rules" {
        Invoke-Web @("python", "manage.py", "eval_rules")
    }
    "eval-agent" {
        Invoke-Web @("python", "manage.py", "eval_agent")
    }
    "ask-agent" {
        if (-not $Question) {
            throw 'Usage: .\make.ps1 ask-agent "your question"'
        }
        Invoke-Web @("python", "manage.py", "ask_agent", $Question)
    }
    "test" {
        Invoke-Web @("pytest")
    }
    "help" {
        Write-Host @"
DocIntel Lab (Docker). Usage: .\make.ps1 <target>

  up        Build and start postgres, redis, neo4j, web, worker. Then migrate.
  down      Stop the containers.
  migrate   python manage.py migrate, inside web.
  web       Follow the Django log.
  worker    Follow the Celery log (prefork pool).
  ingest    Load data/corpus through the pipeline. Builds the synthetic docs first if you run demo.
  demo      up + synthetic corpus + ingest, then print the URL.
  demo-reset  Wipe documents, rules, the graph and agent runs, then rebuild the 4 corpus documents with the real providers. Eval runs are kept.
  eval      Prompt v1/v2 eval, retrieval hit@5, hybrid vs GraphRAG, rule extraction v1/v2, then the agent cases.
  extract-rules  Extract, quote-check, merge and conflict-check rules from every document.
  eval-rules     Precision and recall of rule prompts v1 and v2 against the gold set.
  eval-agent     The 6 compliance-agent cases: intent, answer, citations, conflict, loops.
  ask-agent "Q"  One question through the agent; prints the trace and stops at approval.
  test      pytest, inside web.
"@
    }
    default {
        Write-Host "Unknown target '$Target'."
        Write-Host "Run .\make.ps1 help"
        exit 1
    }
}
