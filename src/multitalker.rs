use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, process::Stdio};

use anyhow::{Context, Result, bail};
use parakeet_rs::{MultitalkerASR, SpeakerTranscript};
use tempfile::TempDir;

use crate::{BackendDiarization, BackendMetrics, BackendResponse, OutputFormat, TranscribeCli};

const DEFAULT_MULTITALKER_DIR: &str = ".cache/models/multitalker";
const DEFAULT_SORTFORMER_PATH: &str =
    ".cache/models/sortformer/diar_streaming_sortformer_4spk-v2.onnx";
const SEGMENT_GAP_SECS: f32 = 0.9;

#[derive(Debug, Clone)]
struct WordSpan {
    speaker_id: usize,
    start_secs: f32,
    end_secs: f32,
    text: String,
}

#[derive(Debug, Clone)]
struct SpeakerSegment {
    speaker_id: usize,
    start_secs: f32,
    end_secs: f32,
    text: String,
}

pub(crate) fn run_multitalker(cli: &TranscribeCli, root_dir: &Path) -> Result<BackendResponse> {
    if !matches!(cli.device.as_str(), "auto" | "cpu") {
        bail!("multitalker backend currently supports only --device auto|cpu");
    }

    let started = Instant::now();
    let temp_dir = runtime_dir(root_dir, cli.work_dir.as_deref())?;
    let runtime_path = temp_dir.path();
    let normalized = normalize_audio(&cli.input, runtime_path)?;
    let audio_sec = probe_audio_duration(&normalized).ok();

    let multitalker_dir = root_dir.join(DEFAULT_MULTITALKER_DIR);
    let sortformer_model = root_dir.join(DEFAULT_SORTFORMER_PATH);
    ensure_model_artifacts(&multitalker_dir, &sortformer_model)?;

    let model_name = "smcleod/multitalker-parakeet-streaming-0.6b-v1-onnx-int8";
    let diarization_name = "altunenes/parakeet-rs Sortformer v2";

    let model_load_start = Instant::now();
    let mut model = MultitalkerASR::from_pretrained(&multitalker_dir, &sortformer_model, None)
        .context("failed to initialize MultitalkerASR")?;
    let model_load_sec = model_load_start.elapsed().as_secs_f64();

    let infer_start = Instant::now();
    let transcripts = model
        .transcribe_file_multitalker(&normalized)
        .context("multitalker transcription failed")?;
    let inference_sec = infer_start.elapsed().as_secs_f64();

    let segments = build_segments(&transcripts);
    if segments.is_empty() {
        bail!("multitalker produced no speaker-attributed transcript");
    }

    let transcript = match cli.format {
        OutputFormat::Text => render_text(&segments, cli.timestamps),
        OutputFormat::Md => render_markdown(&segments, &cli.input, model_name, diarization_name),
    };

    if let Some(output) = &cli.out {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating output dir: {}", parent.display()))?;
        }
        fs::write(output, &transcript)
            .with_context(|| format!("failed writing transcript: {}", output.display()))?;
    }

    Ok(BackendResponse {
        transcript,
        output_path: cli.out.as_ref().map(|p| p.display().to_string()),
        source: cli.input.display().to_string(),
        model: model_name.to_string(),
        device: "cpu".to_string(),
        format: match cli.format {
            OutputFormat::Text => "text".to_string(),
            OutputFormat::Md => "md".to_string(),
        },
        metrics: Some(BackendMetrics {
            model_load_sec,
            inference_sec,
            total_sec: started.elapsed().as_secs_f64(),
            audio_sec,
            diarization_sec: None,
            speaker_identification_sec: None,
            speaker_segments: Some(segments.len()),
            identified_speakers: None,
        }),
        diarization: Some(BackendDiarization {
            model: diarization_name.to_string(),
            speaker_count: unique_speaker_count(&segments),
            segment_count: segments.len(),
        }),
    })
}

struct RuntimeDir {
    path: PathBuf,
    _temp: Option<TempDir>,
}

impl RuntimeDir {
    fn path(&self) -> &Path {
        &self.path
    }
}

fn runtime_dir(root_dir: &Path, work_dir: Option<&Path>) -> Result<RuntimeDir> {
    if let Some(dir) = work_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed creating work dir: {}", dir.display()))?;
        Ok(RuntimeDir {
            path: dir.to_path_buf(),
            _temp: None,
        })
    } else {
        let temp = TempDir::new_in(root_dir.join("tmp")).context("failed creating temp dir")?;
        Ok(RuntimeDir {
            path: temp.path().to_path_buf(),
            _temp: Some(temp),
        })
    }
}

fn ensure_model_artifacts(multitalker_dir: &Path, sortformer_model: &Path) -> Result<()> {
    for required in [
        multitalker_dir.join("encoder.int8.onnx"),
        multitalker_dir.join("decoder_joint.int8.onnx"),
        multitalker_dir.join("tokenizer.model"),
        sortformer_model.to_path_buf(),
    ] {
        if !required.exists() {
            bail!("missing Multitalker model artifact: {}", required.display());
        }
    }
    Ok(())
}

fn normalize_audio(input: &Path, runtime_dir: &Path) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("audio");
    let normalized = runtime_dir.join(format!("normalized_{stem}.wav"));

    let status = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(&normalized)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to launch ffmpeg for audio normalization")?;

    if !status.success() {
        bail!("ffmpeg failed to normalize audio");
    }

    Ok(normalized)
}

fn probe_audio_duration(audio_path: &Path) -> Result<f64> {
    let output = std::process::Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(audio_path)
        .output()
        .context("failed to launch ffprobe")?;

    if !output.status.success() {
        bail!("ffprobe failed for {}", audio_path.display());
    }

    let value = String::from_utf8(output.stdout).context("invalid ffprobe UTF-8 output")?;
    let duration = value
        .trim()
        .parse::<f64>()
        .with_context(|| format!("invalid duration from ffprobe for {}", audio_path.display()))?;
    Ok(duration)
}

fn build_segments(transcripts: &[SpeakerTranscript]) -> Vec<SpeakerSegment> {
    let mut segments = Vec::new();

    for transcript in transcripts {
        let mut current: Vec<WordSpan> = Vec::new();
        for word in &transcript.words {
            let trimmed = word.word.trim();
            if trimmed.is_empty() {
                continue;
            }
            let span = WordSpan {
                speaker_id: transcript.speaker_id,
                start_secs: word.start_secs,
                end_secs: word.end_secs,
                text: trimmed.to_string(),
            };

            let split = current.last().is_some_and(|prev| {
                span.start_secs - prev.end_secs > SEGMENT_GAP_SECS || ends_sentence(&prev.text)
            });
            if split {
                if let Some(segment) = collapse_words(&current) {
                    segments.push(segment);
                }
                current.clear();
            }
            current.push(span);
        }

        if let Some(segment) = collapse_words(&current) {
            segments.push(segment);
        }
    }

    segments.sort_by(|a, b| {
        a.start_secs
            .partial_cmp(&b.start_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    segments
}

fn collapse_words(words: &[WordSpan]) -> Option<SpeakerSegment> {
    let first = words.first()?;
    let last = words.last()?;
    Some(SpeakerSegment {
        speaker_id: first.speaker_id,
        start_secs: first.start_secs,
        end_secs: last.end_secs,
        text: join_words(words.iter().map(|word| word.text.as_str())),
    })
}

fn join_words<'a>(words: impl Iterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for word in words {
        if out.is_empty() || is_punctuation(word) {
            out.push_str(word);
        } else {
            out.push(' ');
            out.push_str(word);
        }
    }
    out
}

fn is_punctuation(word: &str) -> bool {
    matches!(word, "." | "," | "!" | "?" | ":" | ";")
}

fn ends_sentence(word: &str) -> bool {
    word.ends_with('.') || word.ends_with('!') || word.ends_with('?')
}

fn unique_speaker_count(segments: &[SpeakerSegment]) -> usize {
    let mut speakers = std::collections::BTreeSet::new();
    for segment in segments {
        speakers.insert(segment.speaker_id);
    }
    speakers.len()
}

fn render_text(segments: &[SpeakerSegment], timestamps: bool) -> String {
    let mut out = String::new();
    for (idx, segment) in segments.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        if timestamps {
            out.push('[');
            out.push_str(&format_timestamp(segment.start_secs));
            out.push_str("] ");
        }
        out.push_str(&format!(
            "SPEAKER_{:02}: {}",
            segment.speaker_id, segment.text
        ));
    }
    out
}

fn render_markdown(
    segments: &[SpeakerSegment],
    input: &Path,
    model_name: &str,
    diarization_name: &str,
) -> String {
    let mut out = String::new();
    out.push_str("# Speaker Transcript\n\n");
    out.push_str(&format!("- Source: `{}`\n", input.display()));
    out.push_str(&format!("- ASR model: `{model_name}`\n"));
    out.push_str(&format!("- Speaker backend: `{diarization_name}`\n\n"));
    for segment in segments {
        out.push_str(&format!(
            "- [{} - {}] `SPEAKER_{:02}`: {}\n",
            format_timestamp(segment.start_secs),
            format_timestamp(segment.end_secs),
            segment.speaker_id,
            segment.text
        ));
    }
    out
}

fn format_timestamp(secs: f32) -> String {
    let total = secs.max(0.0).floor() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}
