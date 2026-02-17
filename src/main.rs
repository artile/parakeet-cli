use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

#[derive(Debug, Parser)]
#[command(name = "parakeet")]
#[command(about = "Fast local transcription CLI using NVIDIA Parakeet")]
struct Cli {
    #[arg(long, short = 'i')]
    input: PathBuf,

    #[arg(long, short = 'o')]
    out: Option<PathBuf>,

    #[arg(long)]
    model: Option<String>,

    #[arg(long, default_value = "auto")]
    device: String,

    #[arg(long)]
    vocab: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    #[arg(long, default_value_t = false)]
    timestamps: bool,

    #[arg(long, default_value_t = false)]
    no_fuzzy_vocab: bool,

    #[arg(long, default_value_t = false)]
    verbose: bool,

    #[arg(long, value_enum, default_value_t = EmitMode::Text)]
    emit: EmitMode,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Md,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum EmitMode {
    Text,
    Json,
}

#[derive(serde::Serialize)]
struct BackendRequest<'a> {
    input: &'a Path,
    output: Option<&'a Path>,
    model: &'a str,
    device: &'a str,
    vocab: Option<&'a Path>,
    format: &'a str,
    timestamps: bool,
    fuzzy_vocab: bool,
    verbose: bool,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct BackendResponse {
    transcript: String,
    output_path: Option<String>,
    source: String,
    model: String,
    device: String,
    format: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let root_dir = std::env::var("PARAKEET_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root/.parakeet"));
    let venv_python = root_dir.join(".venv/bin/python");
    let backend = root_dir.join("python/parakeet_backend.py");

    if !cli.input.exists() {
        bail!("input does not exist: {}", cli.input.display());
    }
    if !venv_python.exists() {
        bail!(
            "python environment missing at {}. Run: {}/scripts/install.sh",
            venv_python.display(),
            root_dir.display()
        );
    }
    if !backend.exists() {
        bail!("backend script not found: {}", backend.display());
    }

    let output_format = match cli.format {
        OutputFormat::Text => "text",
        OutputFormat::Md => "md",
    };
    let model_name = cli
        .model
        .as_deref()
        .unwrap_or("nvidia/parakeet-tdt-0.6b-v3");

    let request = BackendRequest {
        input: &cli.input,
        output: cli.out.as_deref(),
        model: model_name,
        device: &cli.device,
        vocab: cli.vocab.as_deref(),
        format: output_format,
        timestamps: cli.timestamps,
        fuzzy_vocab: !cli.no_fuzzy_vocab,
        verbose: cli.verbose,
    };
    let json = serde_json::to_string(&request).context("serialize backend request")?;

    let mut cmd = Command::new(&venv_python);
    cmd.arg(&backend)
        .arg("--json")
        .arg(json)
        .env("PARAKEET_HOME", &root_dir)
        .env("HF_HOME", root_dir.join(".cache/hf"))
        .env("TRANSFORMERS_CACHE", root_dir.join(".cache/hf"))
        .env("TORCH_HOME", root_dir.join(".cache/torch"))
        .env("NEMO_HOME", root_dir.join(".cache/nemo"))
        .env("PIP_CACHE_DIR", root_dir.join(".cache/pip"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().context("failed to launch python backend")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to capture backend stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("failed to capture backend stderr"))?;

    let stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        let mut out = Vec::new();
        while let Some(line) = reader.next_line().await? {
            out.push(line);
        }
        Ok::<Vec<String>, std::io::Error>(out)
    });

    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        let mut err = Vec::new();
        while let Some(line) = reader.next_line().await? {
            err.push(line);
        }
        Ok::<Vec<String>, std::io::Error>(err)
    });

    let status = child.wait().await.context("failed waiting for backend")?;
    let stdout_lines = stdout_task.await.context("stdout task join error")??;
    let stderr_lines = stderr_task.await.context("stderr task join error")??;

    if !status.success() {
        let stderr_text = stderr_lines.join("\n");
        bail!("transcription failed:\n{}", stderr_text.trim());
    }

    if cli.verbose {
        for line in &stderr_lines {
            eprintln!("{line}");
        }
    }

    let stdout_text = stdout_lines.join("\n");
    let json_line = stdout_text
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow!("backend did not return JSON output"))?;
    let parsed: BackendResponse =
        serde_json::from_str(json_line.trim()).context("failed to parse backend response JSON")?;

    match cli.emit {
        EmitMode::Text => {
            println!("{}", parsed.transcript);
        }
        EmitMode::Json => {
            let json = serde_json::to_string_pretty(&parsed).context("serialize output JSON")?;
            println!("{json}");
        }
    }

    Ok(())
}
