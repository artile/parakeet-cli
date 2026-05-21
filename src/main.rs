mod multitalker;

use std::io::{BufRead, BufReader as StdBufReader, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use std::{collections::BTreeSet, fs};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use multitalker::run_multitalker;
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
    work_dir: Option<PathBuf>,

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
    diarize: bool,

    #[arg(long, value_enum, default_value_t = SpeakerEngine::Pyannote)]
    speaker_engine: SpeakerEngine,

    #[arg(long, default_value_t = false)]
    identify_speakers: bool,

    #[arg(long)]
    speaker_profile_dir: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    no_fuzzy_vocab: bool,

    #[arg(long, default_value_t = false)]
    verbose: bool,

    #[arg(long, value_enum, default_value_t = EmitMode::Text)]
    emit: EmitMode,

    #[arg(long)]
    daemon_socket: Option<PathBuf>,

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
    Speaker(SpeakerCli),
}

#[derive(Debug, Parser)]
struct DaemonCli {
    #[command(subcommand)]
    command: DaemonCommand,
}

#[derive(Debug, Parser)]
struct SpeakerCli {
    #[command(subcommand)]
    command: SpeakerCommand,
}

#[derive(Debug, Subcommand)]
enum SpeakerCommand {
    Enroll {
        #[arg(long)]
        name: String,
        #[arg(long, short = 'i')]
        input: PathBuf,
        #[arg(long)]
        profile_dir: Option<PathBuf>,
        #[arg(long)]
        start_sec: Option<f64>,
        #[arg(long)]
        end_sec: Option<f64>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    Identify {
        #[arg(long, short = 'i')]
        input: PathBuf,
        #[arg(long)]
        diarization: PathBuf,
        #[arg(long)]
        profile_dir: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        render_out: Option<PathBuf>,
        #[arg(long)]
        work_dir: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
        #[arg(long, default_value_t = false)]
        timestamps: bool,
        #[arg(long, value_enum, default_value_t = EmitMode::Json)]
        emit: EmitMode,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
}

#[derive(Debug, Subcommand)]
enum DaemonCommand {
    Start {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long)]
        logfile: Option<PathBuf>,
    },
    Stop {
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    Status {
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        logfile: Option<PathBuf>,
    },
    Logs {
        #[arg(long)]
        logfile: Option<PathBuf>,
        #[arg(long, default_value_t = 80)]
        lines: usize,
    },
    Serve {
        #[arg(long)]
        socket: Option<PathBuf>,
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum SpeakerEngine {
    Pyannote,
    Multitalker,
}

#[derive(serde::Serialize)]
struct BackendRequest<'a> {
    input: &'a Path,
    output: Option<&'a Path>,
    work_dir: Option<&'a Path>,
    model: &'a str,
    device: &'a str,
    vocab: Option<&'a Path>,
    format: &'a str,
    timestamps: bool,
    diarize: bool,
    identify_speakers: bool,
    speaker_profile_dir: Option<&'a Path>,
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
    diarization: Option<BackendDiarization>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct BackendMetrics {
    model_load_sec: f64,
    inference_sec: f64,
    total_sec: f64,
    audio_sec: Option<f64>,
    diarization_sec: Option<f64>,
    speaker_identification_sec: Option<f64>,
    speaker_segments: Option<usize>,
    identified_speakers: Option<usize>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct BackendDiarization {
    model: String,
    speaker_count: usize,
    segment_count: usize,
}

#[derive(serde::Serialize)]
struct SpeakerEnrollRequest<'a> {
    input: &'a Path,
    name: &'a str,
    profile_dir: &'a Path,
    start_sec: Option<f64>,
    end_sec: Option<f64>,
    device: &'a str,
    verbose: bool,
}

#[derive(serde::Deserialize)]
struct SpeakerEnrollResponse {
    name: String,
    profile_path: String,
    sample_count: usize,
    duration_sec: f64,
    embedding_model: String,
}

#[derive(serde::Serialize)]
struct SpeakerIdentifyRequest<'a> {
    input: &'a Path,
    diarization: &'a Path,
    profile_dir: &'a Path,
    output: Option<&'a Path>,
    render_output: Option<&'a Path>,
    work_dir: Option<&'a Path>,
    format: &'a str,
    timestamps: bool,
    device: &'a str,
    verbose: bool,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct SpeakerIdentifyResponse {
    input: String,
    diarization_source: String,
    output_path: Option<String>,
    rendered_path: Option<String>,
    transcript: Option<String>,
    embedding_model: String,
    metrics: SpeakerIdentifyMetrics,
    assignments: Vec<serde_json::Value>,
    segments: Vec<SpeakerIdentifiedSegment>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct SpeakerIdentifyMetrics {
    identification_sec: f64,
    audio_sec: Option<f64>,
    cluster_count: usize,
    matched_speakers: usize,
    segment_count: usize,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct SpeakerIdentifiedSegment {
    speaker: String,
    speaker_original: Option<String>,
    start_sec: f64,
    end_sec: f64,
    text: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if let Some(argv0) = args.first()
        && Path::new(argv0)
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|name| matches!(name, "parakeetd" | "paraketd"))
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
        if args[1] == "speaker" {
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
        RootCommand::Speaker(speaker) => run_speaker(speaker).await,
    }
}

async fn run_daemon(daemon: DaemonCli) -> Result<()> {
    match daemon.command {
        DaemonCommand::Start {
            socket,
            pidfile,
            logfile,
        } => daemon_start(
            &socket.unwrap_or_else(default_socket_path),
            &pidfile.unwrap_or_else(default_pid_path),
            &logfile.unwrap_or_else(default_log_path),
        ),
        DaemonCommand::Stop { pidfile, socket } => daemon_stop(
            &pidfile.unwrap_or_else(default_pid_path),
            &socket.unwrap_or_else(default_socket_path),
        ),
        DaemonCommand::Status {
            pidfile,
            socket,
            logfile,
        } => daemon_status(
            &pidfile.unwrap_or_else(default_pid_path),
            &socket.unwrap_or_else(default_socket_path),
            &logfile.unwrap_or_else(default_log_path),
        ),
        DaemonCommand::Logs { logfile, lines } => {
            daemon_logs(&logfile.unwrap_or_else(default_log_path), lines)
        }
        DaemonCommand::Serve { socket } => {
            daemon_serve(&socket.unwrap_or_else(default_socket_path)).await
        }
    }
}

async fn daemon_serve(socket: &Path) -> Result<()> {
    let root_dir = parakeet_home();
    let venv_python = root_dir.join(".venv/bin/python");
    let backend = root_dir.join("python/parakeet_backend.py");

    fs::create_dir_all(root_dir.join("tmp"))?;
    fs::create_dir_all(root_dir.join("output"))?;

    let err = std::process::Command::new(&venv_python)
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
        .exec();

    Err(anyhow!(err)).context("failed launching daemon backend")
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
    if is_pidfile_running(pidfile)? || is_socket_reachable(socket) {
        println!("parakeet daemon running");
        println!("socket: {}", socket.display());
        println!("log: {}", logfile.display());
        Ok(())
    } else {
        let _ = fs::remove_file(pidfile);
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

    if is_zombie_pid(pid)? {
        return Ok(false);
    }

    let status = std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .context("failed to probe pid")?;
    Ok(status.success())
}

fn is_zombie_pid(pid: u32) -> Result<bool> {
    let proc_stat = format!("/proc/{pid}/stat");
    let raw = match fs::read_to_string(&proc_stat) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err).with_context(|| format!("failed reading {proc_stat}")),
    };

    let Some(close_paren) = raw.rfind(')') else {
        return Ok(false);
    };
    let rest = raw[close_paren + 1..].trim_start();
    let mut parts = rest.split_whitespace();
    let state = parts.next().unwrap_or_default();
    Ok(state == "Z")
}

fn is_socket_reachable(socket: &Path) -> bool {
    if !socket.exists() {
        return false;
    }
    UnixStream::connect(socket).is_ok()
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

    if cli.identify_speakers && !cli.diarize {
        bail!("--identify-speakers requires --diarize");
    }
    if cli.identify_speakers && cli.speaker_engine != SpeakerEngine::Pyannote {
        bail!("--identify-speakers is currently supported only with --speaker-engine pyannote");
    }

    if !cli.input.exists() {
        bail!("input does not exist: {}", cli.input.display());
    }
    if !venv_python.exists() {
        bail!(
            "python environment missing at {}. Bootstrap env/tools via: {}/install.sh",
            venv_python.display(),
            root_dir.display(),
        );
    }
    if !backend.exists() {
        bail!("backend script not found: {}", backend.display());
    }

    let output_format = match cli.format {
        OutputFormat::Text => "text",
        OutputFormat::Md => "md",
    };
    let speaker_profile_dir = if cli.identify_speakers {
        Some(
            cli.speaker_profile_dir
                .clone()
                .unwrap_or_else(default_speaker_profile_dir),
        )
    } else {
        cli.speaker_profile_dir.clone()
    };
    let model_name = cli
        .model
        .as_deref()
        .unwrap_or("nvidia/parakeet-tdt-0.6b-v3");

    let merged_vocab_path = prepare_vocab_file(
        &root_dir,
        cli.vocab.as_deref(),
        !cli.no_library,
        cli.work_dir.as_deref(),
    )
    .context("failed preparing vocabulary file")?;

    let request = BackendRequest {
        input: &cli.input,
        output: cli.out.as_deref(),
        work_dir: cli.work_dir.as_deref(),
        model: model_name,
        device: &cli.device,
        vocab: merged_vocab_path.as_deref(),
        format: output_format,
        timestamps: cli.timestamps,
        diarize: cli.diarize,
        identify_speakers: cli.identify_speakers,
        speaker_profile_dir: speaker_profile_dir.as_deref(),
        fuzzy_vocab: !cli.no_fuzzy_vocab,
        verbose: cli.verbose,
    };

    if cli.diarize && cli.speaker_engine == SpeakerEngine::Multitalker {
        let parsed = run_multitalker(&cli, &root_dir)?;
        emit_response(&cli, &parsed)?;
        return Ok(());
    }

    let json = serde_json::to_string(&request).context("serialize backend request")?;

    let daemon_socket = cli
        .daemon_socket
        .as_deref()
        .map_or_else(default_socket_path, PathBuf::from);
    if !cli.no_daemon
        && !cli.diarize
        && let Ok(parsed) = try_daemon_request(&daemon_socket, &json)
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

async fn run_speaker(cli: SpeakerCli) -> Result<()> {
    let root_dir = parakeet_home();
    let venv_python = root_dir.join(".venv/bin/python");
    let backend = root_dir.join("python/parakeet_backend.py");

    match cli.command {
        SpeakerCommand::Enroll {
            name,
            input,
            profile_dir,
            start_sec,
            end_sec,
            device,
            verbose,
        } => {
            if !input.exists() {
                bail!("input does not exist: {}", input.display());
            }
            if !venv_python.exists() {
                bail!(
                    "python environment missing at {}. Bootstrap env/tools via: {}/install.sh",
                    venv_python.display(),
                    root_dir.display(),
                );
            }
            if !backend.exists() {
                bail!("backend script not found: {}", backend.display());
            }

            let profile_dir = profile_dir.unwrap_or_else(default_speaker_profile_dir);
            let request = SpeakerEnrollRequest {
                input: &input,
                name: &name,
                profile_dir: &profile_dir,
                start_sec,
                end_sec,
                device: &device,
                verbose,
            };
            let json =
                serde_json::to_string(&request).context("serialize speaker enroll request")?;

            let output = Command::new(&venv_python)
                .arg(&backend)
                .arg("--speaker-enroll-json")
                .arg(json)
                .env("PARAKEET_HOME", &root_dir)
                .env("HF_HOME", root_dir.join(".cache/hf"))
                .env("TRANSFORMERS_CACHE", root_dir.join(".cache/hf"))
                .env("TORCH_HOME", root_dir.join(".cache/torch"))
                .env("NEMO_HOME", root_dir.join(".cache/nemo"))
                .env("PIP_CACHE_DIR", root_dir.join(".cache/pip"))
                .output()
                .await
                .context("failed to launch speaker enrollment backend")?;

            if !output.status.success() {
                bail!(
                    "speaker enrollment failed:\n{}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }

            let stdout = String::from_utf8(output.stdout)
                .context("invalid UTF-8 from speaker enrollment")?;
            let json_line = stdout
                .lines()
                .rev()
                .find(|line| line.trim_start().starts_with('{'))
                .ok_or_else(|| anyhow!("speaker enrollment backend did not return JSON output"))?;
            let response: SpeakerEnrollResponse = serde_json::from_str(json_line.trim())
                .context("failed to parse speaker enrollment response")?;

            println!("speaker enrolled");
            println!("name: {}", response.name);
            println!("profile: {}", response.profile_path);
            println!("samples: {}", response.sample_count);
            println!("duration_sec: {:.2}", response.duration_sec);
            println!("embedding_model: {}", response.embedding_model);
            Ok(())
        }
        SpeakerCommand::Identify {
            input,
            diarization,
            profile_dir,
            out,
            render_out,
            work_dir,
            format,
            timestamps,
            emit,
            device,
            verbose,
        } => {
            if !input.exists() {
                bail!("input does not exist: {}", input.display());
            }
            if !diarization.exists() {
                bail!(
                    "diarization input does not exist: {}",
                    diarization.display()
                );
            }
            if !venv_python.exists() {
                bail!(
                    "python environment missing at {}. Bootstrap env/tools via: {}/install.sh",
                    venv_python.display(),
                    root_dir.display(),
                );
            }
            if !backend.exists() {
                bail!("backend script not found: {}", backend.display());
            }

            let profile_dir = profile_dir.unwrap_or_else(default_speaker_profile_dir);
            let output_format = match format {
                OutputFormat::Text => "text",
                OutputFormat::Md => "md",
            };
            let request = SpeakerIdentifyRequest {
                input: &input,
                diarization: &diarization,
                profile_dir: &profile_dir,
                output: out.as_deref(),
                render_output: render_out.as_deref(),
                work_dir: work_dir.as_deref(),
                format: output_format,
                timestamps,
                device: &device,
                verbose,
            };
            let json =
                serde_json::to_string(&request).context("serialize speaker identify request")?;

            let output = Command::new(&venv_python)
                .arg(&backend)
                .arg("--speaker-identify-json")
                .arg(json)
                .env("PARAKEET_HOME", &root_dir)
                .env("HF_HOME", root_dir.join(".cache/hf"))
                .env("TRANSFORMERS_CACHE", root_dir.join(".cache/hf"))
                .env("TORCH_HOME", root_dir.join(".cache/torch"))
                .env("NEMO_HOME", root_dir.join(".cache/nemo"))
                .env("PIP_CACHE_DIR", root_dir.join(".cache/pip"))
                .output()
                .await
                .context("failed to launch speaker identify backend")?;

            if !output.status.success() {
                bail!(
                    "speaker identification failed:\n{}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }

            let stdout =
                String::from_utf8(output.stdout).context("invalid UTF-8 from speaker identify")?;
            let json_line = stdout
                .lines()
                .rev()
                .find(|line| line.trim_start().starts_with('{'))
                .ok_or_else(|| anyhow!("speaker identify backend did not return JSON output"))?;
            let response: SpeakerIdentifyResponse = serde_json::from_str(json_line.trim())
                .context("failed to parse speaker identify response")?;

            emit_speaker_identify_response(emit, verbose, &response)
        }
    }
}

fn parakeet_home() -> PathBuf {
    std::env::var("PARAKEET_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_parakeet_home())
}

fn default_parakeet_home() -> PathBuf {
    std::env::current_exe()
        .ok()
        .map(|path| {
            let parent = path.parent().map(PathBuf::from);
            let is_target_bin = path
                .parent()
                .and_then(|dir| dir.file_name())
                .and_then(|name| name.to_str())
                .is_some_and(|name| matches!(name, "debug" | "release"))
                && path
                    .parent()
                    .and_then(|dir| dir.parent())
                    .and_then(|dir| dir.file_name())
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name == "target");
            if is_target_bin {
                path.parent()
                    .and_then(|dir| dir.parent())
                    .and_then(|dir| dir.parent())
                    .map(PathBuf::from)
                    .or(parent)
            } else {
                parent
            }
        })
        .flatten()
        .unwrap_or_else(|| PathBuf::from("/root/TAO/Tools/parakeet"))
}

fn default_socket_path() -> PathBuf {
    parakeet_home().join("tmp/parakeet.sock")
}

fn default_pid_path() -> PathBuf {
    parakeet_home().join("tmp/parakeetd.pid")
}

fn default_log_path() -> PathBuf {
    parakeet_home().join("output/parakeetd.log")
}

fn default_speaker_profile_dir() -> PathBuf {
    parakeet_home().join("profiles/speakers")
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

fn emit_speaker_identify_response(
    emit: EmitMode,
    verbose: bool,
    response: &SpeakerIdentifyResponse,
) -> Result<()> {
    match emit {
        EmitMode::Json => {
            let json = serde_json::to_string_pretty(response)
                .context("serialize speaker identify JSON")?;
            println!("{json}");
        }
        EmitMode::Text => {
            if let Some(transcript) = &response.transcript {
                println!("{transcript}");
            } else {
                println!("speaker identification completed");
                println!("clusters: {}", response.metrics.cluster_count);
                println!("matched: {}", response.metrics.matched_speakers);
                println!("segments: {}", response.metrics.segment_count);
                if let Some(path) = &response.output_path {
                    println!("output: {path}");
                }
            }
            if verbose {
                eprintln!(
                    "[speaker identify metrics] identify={:.2}s audio={} clusters={} matched={} segments={}",
                    response.metrics.identification_sec,
                    response
                        .metrics
                        .audio_sec
                        .map(|x| format!("{x:.2}s"))
                        .unwrap_or_else(|| "n/a".to_string()),
                    response.metrics.cluster_count,
                    response.metrics.matched_speakers,
                    response.metrics.segment_count,
                );
            }
        }
    }
    Ok(())
}

fn prepare_vocab_file(
    root_dir: &Path,
    user_vocab: Option<&Path>,
    use_library: bool,
    work_dir: Option<&Path>,
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

    let tmp_dir = work_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| root_dir.join("tmp"));
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
