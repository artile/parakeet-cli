use std::io::{BufRead, BufReader as StdBufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use std::{collections::BTreeSet, fs};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

#[derive(Debug, Parser)]
#[command(name = "parakeet")]
#[command(about = "Fast local transcription CLI using NVIDIA Parakeet")]
struct TranscribeCli {
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

    #[arg(long, default_value_t = false)]
    no_library: bool,

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

    #[arg(long, default_value = "/root/.parakeet/tmp/parakeet.sock")]
    daemon_socket: PathBuf,

    #[arg(long, default_value_t = false)]
    no_daemon: bool,
}

#[derive(Debug, Parser)]
#[command(name = "parakeet")]
struct RootCli {
    #[command(subcommand)]
    command: RootCommand,
}

#[derive(Debug, Subcommand)]
enum RootCommand {
    Transcribe(TranscribeCli),
    Daemon(DaemonCli),
}

#[derive(Debug, Parser)]
struct DaemonCli {
    #[command(subcommand)]
    command: DaemonCommand,
}

#[derive(Debug, Subcommand)]
enum DaemonCommand {
    Start {
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeet.sock")]
        socket: PathBuf,
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeetd.pid")]
        pidfile: PathBuf,
        #[arg(long, default_value = "/root/.parakeet/output/parakeetd.log")]
        logfile: PathBuf,
    },
    Stop {
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeetd.pid")]
        pidfile: PathBuf,
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeet.sock")]
        socket: PathBuf,
    },
    Status {
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeetd.pid")]
        pidfile: PathBuf,
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeet.sock")]
        socket: PathBuf,
        #[arg(long, default_value = "/root/.parakeet/output/parakeetd.log")]
        logfile: PathBuf,
    },
    Logs {
        #[arg(long, default_value = "/root/.parakeet/output/parakeetd.log")]
        logfile: PathBuf,
        #[arg(long, default_value_t = 80)]
        lines: usize,
    },
    Serve {
        #[arg(long, default_value = "/root/.parakeet/tmp/parakeet.sock")]
        socket: PathBuf,
    },
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
    metrics: Option<BackendMetrics>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct BackendMetrics {
    model_load_sec: f64,
    inference_sec: f64,
    total_sec: f64,
    audio_sec: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if let Some(argv0) = args.first()
        && Path::new(argv0)
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|name| name == "parakeetd")
    {
        args.insert(1, "daemon".into());
    }
    if args.len() > 1 {
        if args[1] == "transcribe" {
            args.remove(1);
            let cli = TranscribeCli::parse_from(args);
            return run_transcribe(cli).await;
        }
        if args[1] == "daemon" {
            let root = RootCli::parse_from(args);
            return run_root(root).await;
        }
    }

    let cli = TranscribeCli::parse_from(args);
    run_transcribe(cli).await
}

async fn run_root(root: RootCli) -> Result<()> {
    match root.command {
        RootCommand::Transcribe(cli) => run_transcribe(cli).await,
        RootCommand::Daemon(daemon) => run_daemon(daemon).await,
    }
}

async fn run_daemon(daemon: DaemonCli) -> Result<()> {
    match daemon.command {
        DaemonCommand::Start {
            socket,
            pidfile,
            logfile,
        } => daemon_start(&socket, &pidfile, &logfile),
        DaemonCommand::Stop { pidfile, socket } => daemon_stop(&pidfile, &socket),
        DaemonCommand::Status {
            pidfile,
            socket,
            logfile,
        } => daemon_status(&pidfile, &socket, &logfile),
        DaemonCommand::Logs { logfile, lines } => daemon_logs(&logfile, lines),
        DaemonCommand::Serve { socket } => daemon_serve(&socket).await,
    }
}

async fn daemon_serve(socket: &Path) -> Result<()> {
    let root_dir = parakeet_home();
    let venv_python = root_dir.join(".venv/bin/python");
    let backend = root_dir.join("python/parakeet_backend.py");

    fs::create_dir_all(root_dir.join("tmp"))?;
    fs::create_dir_all(root_dir.join("output"))?;

    let status = std::process::Command::new(&venv_python)
        .arg(&backend)
        .arg("--serve")
        .arg("--socket-path")
        .arg(socket)
        .env("PARAKEET_HOME", &root_dir)
        .env("HF_HOME", root_dir.join(".cache/hf"))
        .env("TRANSFORMERS_CACHE", root_dir.join(".cache/hf"))
        .env("TORCH_HOME", root_dir.join(".cache/torch"))
        .env("NEMO_HOME", root_dir.join(".cache/nemo"))
        .env("PIP_CACHE_DIR", root_dir.join(".cache/pip"))
        .status()
        .context("failed launching daemon backend")?;

    if status.success() {
        Ok(())
    } else {
        bail!("daemon backend exited with status: {status}")
    }
}

fn daemon_start(socket: &Path, pidfile: &Path, logfile: &Path) -> Result<()> {
    if is_pidfile_running(pidfile)? {
        println!("parakeet daemon already running");
        return Ok(());
    }

    if let Some(parent) = socket.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = logfile.parent() {
        fs::create_dir_all(parent)?;
    }
    if socket.exists() {
        let _ = fs::remove_file(socket);
    }

    let exe = std::env::current_exe()?;
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(logfile)
        .with_context(|| format!("failed opening logfile: {}", logfile.display()))?;
    let log_err = log.try_clone()?;

    let child = std::process::Command::new(exe)
        .arg("daemon")
        .arg("serve")
        .arg("--socket")
        .arg(socket)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err))
        .spawn()
        .context("failed spawning daemon")?;

    fs::write(pidfile, child.id().to_string())
        .with_context(|| format!("failed writing pidfile: {}", pidfile.display()))?;

    for _ in 0..240 {
        if socket.exists() {
            println!("parakeet daemon started");
            println!("socket: {}", socket.display());
            println!("log: {}", logfile.display());
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    bail!("daemon start timed out")
}

fn daemon_stop(pidfile: &Path, socket: &Path) -> Result<()> {
    let pid = read_pid(pidfile)?;
    if let Some(pid) = pid {
        let status = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status()
            .context("failed to send kill")?;
        if !status.success() {
            bail!("failed stopping daemon pid={pid}");
        }
        println!("parakeet daemon stopped");
    } else {
        println!("parakeet daemon not running");
    }

    let _ = fs::remove_file(pidfile);
    let _ = fs::remove_file(socket);
    Ok(())
}

fn daemon_status(pidfile: &Path, socket: &Path, logfile: &Path) -> Result<()> {
    if is_pidfile_running(pidfile)? {
        println!("parakeet daemon running");
        println!("socket: {}", socket.display());
        println!("log: {}", logfile.display());
        Ok(())
    } else {
        println!("parakeet daemon not running");
        bail!("not running")
    }
}

fn daemon_logs(logfile: &Path, lines: usize) -> Result<()> {
    let content = fs::read_to_string(logfile)
        .with_context(|| format!("failed reading logfile: {}", logfile.display()))?;
    let all: Vec<&str> = content.lines().collect();
    let start = all.len().saturating_sub(lines);
    for line in &all[start..] {
        println!("{line}");
    }
    Ok(())
}

fn is_pidfile_running(pidfile: &Path) -> Result<bool> {
    let Some(pid) = read_pid(pidfile)? else {
        return Ok(false);
    };

    let status = std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .context("failed to probe pid")?;
    Ok(status.success())
}

fn read_pid(pidfile: &Path) -> Result<Option<u32>> {
    if !pidfile.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(pidfile)
        .with_context(|| format!("failed reading pidfile: {}", pidfile.display()))?;
    let pid = raw.trim().parse::<u32>().ok();
    Ok(pid)
}

async fn run_transcribe(cli: TranscribeCli) -> Result<()> {
    let root_dir = parakeet_home();
    let venv_python = root_dir.join(".venv/bin/python");
    let backend = root_dir.join("python/parakeet_backend.py");

    if !cli.input.exists() {
        bail!("input does not exist: {}", cli.input.display());
    }
    if !venv_python.exists() {
        bail!(
            "python environment missing at {}. Bootstrap env/tools via: /root/.parakeet/install.sh",
            venv_python.display(),
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

    let merged_vocab_path = prepare_vocab_file(&root_dir, cli.vocab.as_deref(), !cli.no_library)
        .context("failed preparing vocabulary file")?;

    let request = BackendRequest {
        input: &cli.input,
        output: cli.out.as_deref(),
        model: model_name,
        device: &cli.device,
        vocab: merged_vocab_path.as_deref(),
        format: output_format,
        timestamps: cli.timestamps,
        fuzzy_vocab: !cli.no_fuzzy_vocab,
        verbose: cli.verbose,
    };
    let json = serde_json::to_string(&request).context("serialize backend request")?;

    if !cli.no_daemon
        && let Ok(parsed) = try_daemon_request(&cli.daemon_socket, &json)
    {
        emit_response(&cli, &parsed)?;
        return Ok(());
    }

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

    emit_response(&cli, &parsed)?;

    Ok(())
}

fn parakeet_home() -> PathBuf {
    std::env::var("PARAKEET_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root/.parakeet"))
}

fn try_daemon_request(socket_path: &Path, request_json: &str) -> Result<BackendResponse> {
    let mut stream = UnixStream::connect(socket_path)
        .with_context(|| format!("daemon socket not reachable: {}", socket_path.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(180)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    stream.write_all(request_json.as_bytes())?;
    stream.write_all(b"\n")?;

    let mut reader = StdBufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim().is_empty() {
        bail!("empty daemon response");
    }
    let parsed: BackendResponse =
        serde_json::from_str(line.trim()).context("invalid daemon JSON response")?;
    Ok(parsed)
}

fn emit_response(cli: &TranscribeCli, parsed: &BackendResponse) -> Result<()> {
    match cli.emit {
        EmitMode::Text => {
            println!("{}", parsed.transcript);
            if cli.verbose
                && let Some(m) = &parsed.metrics
            {
                eprintln!(
                    "[parakeet metrics] load={:.2}s infer={:.2}s total={:.2}s audio={}",
                    m.model_load_sec,
                    m.inference_sec,
                    m.total_sec,
                    m.audio_sec
                        .map(|x| format!("{x:.2}s"))
                        .unwrap_or_else(|| "n/a".to_string())
                );
            }
        }
        EmitMode::Json => {
            let json = serde_json::to_string_pretty(&parsed).context("serialize output JSON")?;
            println!("{json}");
        }
    }
    Ok(())
}

fn prepare_vocab_file(
    root_dir: &Path,
    user_vocab: Option<&Path>,
    use_library: bool,
) -> Result<Option<PathBuf>> {
    let mut vocab_files = Vec::new();
    if use_library {
        let auto_vocab = root_dir.join("terms/vocab.txt");
        if auto_vocab.exists() {
            vocab_files.push(auto_vocab);
        }
    }
    if let Some(path) = user_vocab {
        if !path.exists() {
            bail!("vocab file does not exist: {}", path.display());
        }
        vocab_files.push(path.to_path_buf());
    }
    if vocab_files.is_empty() {
        return Ok(None);
    }

    let mut merged = BTreeSet::new();
    for file in vocab_files {
        let content = fs::read_to_string(&file)
            .with_context(|| format!("failed reading vocab: {}", file.display()))?;
        for line in content.lines() {
            let term = line.trim();
            if term.is_empty() || term.starts_with('#') {
                continue;
            }
            merged.insert(term.to_string());
        }
    }

    let tmp_dir = root_dir.join("tmp");
    fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("failed creating tmp dir: {}", tmp_dir.display()))?;
    let merged_path = tmp_dir.join("merged_vocab.txt");
    let mut out = String::new();
    for term in merged {
        out.push_str(&term);
        out.push('\n');
    }
    fs::write(&merged_path, out)
        .with_context(|| format!("failed writing merged vocab: {}", merged_path.display()))?;
    Ok(Some(merged_path))
}
