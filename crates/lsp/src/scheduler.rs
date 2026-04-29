// Copyright (C) 2019-2026 Provable Inc.
// This file is part of the Leo library.

// The Leo library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The Leo library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the Leo library. If not, see <https://www.gnu.org/licenses/>.

#![allow(clippy::mutable_key_type)]

use crate::{
    compiler_bridge::{PackageAnalysisCache, PackageWorkerAnalysis, analyze_package_snapshot, build_document_view},
    document_store::{AnalysisBucket, DocumentSnapshot, DocumentViewKey, DocumentViewSnapshot, PackageAnalysisKey},
    panic_boundary::{PanicReport, catch_unwind},
    semantics::{CachedDocumentView, CachedPackageAnalysis},
};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, unbounded};
use lsp_types::Uri;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, atomic::AtomicU64},
    thread::{self, JoinHandle},
};

const WORKER_CHANNEL_BOUND: usize = 1;
const MAX_PENDING_PACKAGE_JOBS: usize = 16;
const MAX_PENDING_VIEW_JOBS: usize = 64;

type PendingPackageJobs = HashMap<PackageAnalysisKey, PendingPackageJob>;
type PendingViewJobs = HashMap<DocumentViewKey, PendingViewJob>;

/// Commands sent from the main thread to the background worker.
#[derive(Debug)]
pub enum WorkerCommand {
    AnalyzePackage(DocumentSnapshot),
    BuildDocumentView { snapshot: DocumentViewSnapshot, package: Arc<CachedPackageAnalysis> },
    SetOpenBuckets(HashSet<AnalysisBucket>),
    Shutdown,
}

/// Events sent from the background worker back to the main thread.
#[derive(Debug)]
pub enum WorkerEvent {
    PackageAnalyzed(PackageAnalysis),
    DocumentViewBuilt(CachedDocumentView),
    PackageCancelled { key: PackageAnalysisKey, uri: Uri, generation: u64 },
    DocumentViewCancelled { key: DocumentViewKey },
    PackagePanicked { key: PackageAnalysisKey, uri: Uri, generation: u64, report: PanicReport },
    DocumentViewPanicked { key: DocumentViewKey, report: PanicReport },
}

/// Worker-produced semantic state for one document generation.
#[derive(Debug, Clone)]
pub struct PackageAnalysis {
    pub uri: Uri,
    pub generation: u64,
    pub key: PackageAnalysisKey,
    pub result: PackageWorkerAnalysis,
}

/// A coalesced worker job paired with its arrival order.
///
/// The worker keeps at most one pending snapshot per URI, and `sequence`
/// preserves global recency so the most recently updated document runs first.
#[derive(Debug)]
struct PendingPackageJob {
    sequence: u64,
    snapshot: DocumentSnapshot,
}

#[derive(Debug)]
struct PendingViewJob {
    sequence: u64,
    snapshot: DocumentViewSnapshot,
    package: Arc<CachedPackageAnalysis>,
}

/// Background worker owner and communication channels.
#[derive(Debug)]
pub struct Scheduler {
    command_tx: Sender<WorkerCommand>,
    event_rx: Receiver<WorkerEvent>,
    worker: Option<JoinHandle<()>>,
}

impl Scheduler {
    /// Spawn the dedicated analysis worker thread.
    pub fn new(panic_on_worker_job: bool) -> Self {
        let (command_tx, command_rx) = bounded(WORKER_CHANNEL_BOUND);
        let (event_tx, event_rx) = unbounded();

        let worker = thread::Builder::new()
            .name("leo-lsp-worker".to_owned())
            .spawn(move || worker_loop(command_rx, event_tx, panic_on_worker_job))
            .expect("failed to spawn leo-lsp worker");

        Self { command_tx, event_rx, worker: Some(worker) }
    }

    /// Enqueue a document snapshot for background analysis.
    pub fn enqueue_package(&self, snapshot: DocumentSnapshot) {
        let _ = self.command_tx.send(WorkerCommand::AnalyzePackage(snapshot));
    }

    /// Enqueue one document-view rebuild against an already cached package analysis.
    pub fn enqueue_document_view(&self, snapshot: DocumentViewSnapshot, package: Arc<CachedPackageAnalysis>) {
        let _ = self.command_tx.send(WorkerCommand::BuildDocumentView { snapshot, package });
    }

    /// Inform the worker which package buckets still have open documents.
    pub fn set_open_buckets(&self, buckets: HashSet<AnalysisBucket>) {
        let _ = self.command_tx.send(WorkerCommand::SetOpenBuckets(buckets));
    }

    /// Return the receiver used to observe worker events.
    pub fn events(&self) -> &Receiver<WorkerEvent> {
        &self.event_rx
    }

    /// Shut down the worker thread and wait for it to exit.
    pub fn shutdown(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Shutdown);

        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(command_rx: Receiver<WorkerCommand>, event_tx: Sender<WorkerEvent>, panic_on_worker_job: bool) {
    let mut pending_packages = PendingPackageJobs::new();
    let mut pending_views = PendingViewJobs::new();
    let mut sequence = 0_u64;
    let mut package_cache = PackageAnalysisCache::default();

    loop {
        // Block only when there is nothing queued locally; otherwise keep
        // draining and coalescing messages before choosing the next job.
        if pending_packages.is_empty() && pending_views.is_empty() {
            match command_rx.recv() {
                Ok(command) => {
                    if absorb_command(
                        command,
                        &mut pending_packages,
                        &mut pending_views,
                        &mut sequence,
                        &mut package_cache,
                        &event_tx,
                    ) {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        if drain_commands(
            &command_rx,
            &mut pending_packages,
            &mut pending_views,
            &mut sequence,
            &mut package_cache,
            &event_tx,
        ) {
            break;
        }

        if let Some(snapshot) = take_latest_package(&mut pending_packages) {
            run_package_job(snapshot, &event_tx, panic_on_worker_job, &mut package_cache);
        } else if let Some((snapshot, package)) = take_latest_view(&mut pending_views) {
            run_view_job(snapshot, package, &event_tx, panic_on_worker_job);
        }
    }
}

fn absorb_command(
    command: WorkerCommand,
    pending_packages: &mut PendingPackageJobs,
    pending_views: &mut PendingViewJobs,
    sequence: &mut u64,
    package_cache: &mut PackageAnalysisCache,
    event_tx: &Sender<WorkerEvent>,
) -> bool {
    match command {
        WorkerCommand::AnalyzePackage(snapshot) => {
            *sequence += 1;
            if pending_packages.len() >= MAX_PENDING_PACKAGE_JOBS
                && !pending_packages.contains_key(&snapshot.package_key)
                && let Some(dropped) = drop_oldest_package(pending_packages)
            {
                let _ = event_tx.send(WorkerEvent::PackageCancelled {
                    key: dropped.snapshot.package_key,
                    uri: dropped.snapshot.uri,
                    generation: dropped.snapshot.generation,
                });
            }
            // Replacing by package key coalesces edit storms down to the latest
            // heavy overlay snapshot for that package generation.
            pending_packages.insert(snapshot.package_key.clone(), PendingPackageJob { sequence: *sequence, snapshot });
            false
        }
        WorkerCommand::BuildDocumentView { snapshot, package } => {
            *sequence += 1;
            if pending_views.len() >= MAX_PENDING_VIEW_JOBS
                && !pending_views.contains_key(&snapshot.key)
                && let Some(dropped) = drop_oldest_view(pending_views)
            {
                let _ = event_tx.send(WorkerEvent::DocumentViewCancelled { key: dropped.snapshot.key });
            }
            pending_views.insert(snapshot.key.clone(), PendingViewJob { sequence: *sequence, snapshot, package });
            false
        }
        WorkerCommand::SetOpenBuckets(open_buckets) => {
            package_cache.retain_open_buckets(&open_buckets);
            false
        }
        WorkerCommand::Shutdown => true,
    }
}

fn drain_commands(
    command_rx: &Receiver<WorkerCommand>,
    pending_packages: &mut PendingPackageJobs,
    pending_views: &mut PendingViewJobs,
    sequence: &mut u64,
    package_cache: &mut PackageAnalysisCache,
    event_tx: &Sender<WorkerEvent>,
) -> bool {
    // Collapse any burst of pending commands before work starts so the worker
    // spends time on the freshest snapshots rather than intermediate states.
    loop {
        match command_rx.try_recv() {
            Ok(command) => {
                if absorb_command(command, pending_packages, pending_views, sequence, package_cache, event_tx) {
                    return true;
                }
            }
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => return true,
        }
    }
}

fn take_latest_package(pending: &mut PendingPackageJobs) -> Option<DocumentSnapshot> {
    let next_sequence = pending.values().max_by_key(|job| job.sequence)?.sequence;
    pending.extract_if(|_, job| job.sequence == next_sequence).next().map(|(_, job)| job.snapshot)
}

fn take_latest_view(pending: &mut PendingViewJobs) -> Option<(DocumentViewSnapshot, Arc<CachedPackageAnalysis>)> {
    let next_sequence = pending.values().max_by_key(|job| job.sequence)?.sequence;
    pending.extract_if(|_, job| job.sequence == next_sequence).next().map(|(_, job)| (job.snapshot, job.package))
}

fn drop_oldest_package(pending: &mut PendingPackageJobs) -> Option<PendingPackageJob> {
    if let Some(oldest) = pending.iter().min_by_key(|(_, job)| job.sequence).map(|(key, _)| key.clone()) {
        return pending.remove(&oldest);
    }
    None
}

fn drop_oldest_view(pending: &mut PendingViewJobs) -> Option<PendingViewJob> {
    if let Some(oldest) = pending.iter().min_by_key(|(_, job)| job.sequence).map(|(key, _)| key.clone()) {
        return pending.remove(&oldest);
    }
    None
}

fn run_package_job(
    snapshot: DocumentSnapshot,
    event_tx: &Sender<WorkerEvent>,
    panic_on_worker_job: bool,
    package_cache: &mut PackageAnalysisCache,
) {
    let uri = snapshot.uri.clone();
    let generation = snapshot.generation;
    let key = snapshot.package_key.clone();
    let cancel_token = snapshot.cancel_token.clone();

    if is_cancelled(cancel_token.as_ref(), generation) {
        let _ = event_tx.send(WorkerEvent::PackageCancelled { key, uri, generation });
        return;
    }

    // Worker analysis runs off the main thread, so this is the task boundary
    // where we can contain an internal panic, report it as a bug, and keep
    // the server alive for future requests and newer document generations.
    let result = catch_unwind("worker_analyze", Some(&uri), Some(generation), || {
        if panic_on_worker_job {
            panic!("injected worker panic");
        }

        PackageAnalysis {
            uri: uri.clone(),
            generation,
            key: key.clone(),
            result: analyze_package_snapshot(&snapshot, package_cache),
        }
    });

    match result {
        Ok(analysis) => {
            // Check cancellation again after analysis because a newer commit can
            // arrive while this job is already in flight.
            let event = if is_cancelled(cancel_token.as_ref(), generation) {
                WorkerEvent::PackageCancelled { key, uri, generation }
            } else {
                WorkerEvent::PackageAnalyzed(analysis)
            };

            let _ = event_tx.send(event);
        }
        Err(report) => {
            let _ = event_tx.send(WorkerEvent::PackagePanicked { key, uri, generation, report });
        }
    }
}

fn run_view_job(
    snapshot: DocumentViewSnapshot,
    package: Arc<CachedPackageAnalysis>,
    event_tx: &Sender<WorkerEvent>,
    panic_on_worker_job: bool,
) {
    let key = snapshot.key.clone();
    if is_cancelled(snapshot.cancel_token.as_ref(), key.document_generation) {
        let _ = event_tx.send(WorkerEvent::DocumentViewCancelled { key });
        return;
    }

    let result = catch_unwind("worker_document_view", Some(&snapshot.uri), Some(key.document_generation), || {
        if panic_on_worker_job {
            panic!("injected worker panic");
        }

        build_document_view(&snapshot, package)
    });

    match result {
        Ok(view) => {
            let event = if is_cancelled(snapshot.cancel_token.as_ref(), key.document_generation) {
                WorkerEvent::DocumentViewCancelled { key }
            } else {
                WorkerEvent::DocumentViewBuilt(view)
            };
            let _ = event_tx.send(event);
        }
        Err(report) => {
            let _ = event_tx.send(WorkerEvent::DocumentViewPanicked { key, report });
        }
    }
}

fn is_cancelled(cancel_token: &AtomicU64, generation: u64) -> bool {
    // The token stores the latest committed generation for a URI, so any
    // mismatch means newer document state has superseded this snapshot.
    cancel_token.load(std::sync::atomic::Ordering::SeqCst) != generation
}

#[cfg(test)]
mod tests {
    use super::{
        PendingPackageJobs,
        PendingViewJobs,
        Scheduler,
        WorkerCommand,
        WorkerEvent,
        absorb_command,
        take_latest_package,
    };
    use crate::{
        compiler_bridge::PackageAnalysisCache,
        document_store::{AnalysisBucket, DocumentSnapshot, DocumentViewKey, PackageAnalysisKey},
    };
    use crossbeam_channel::unbounded;
    use line_index::LineIndex;
    use lsp_types::Uri;
    use std::{
        path::PathBuf,
        sync::{Arc, atomic::AtomicU64},
        time::Duration,
    };

    fn snapshot(uri: &str, generation: u64) -> DocumentSnapshot {
        snapshot_with_token(uri, generation, Arc::new(AtomicU64::new(generation)))
    }

    fn snapshot_with_token(uri: &str, generation: u64, cancel_token: Arc<AtomicU64>) -> DocumentSnapshot {
        let uri = uri.parse::<Uri>().expect("valid uri");
        let bucket = AnalysisBucket::UnmanagedDocument { uri: uri.clone() };
        let package_key = PackageAnalysisKey { bucket, bucket_generation: generation };
        let view_key =
            DocumentViewKey { uri: uri.clone(), document_generation: generation, package: package_key.clone() };
        DocumentSnapshot {
            uri,
            text: Arc::from("program test.aleo {}"),
            line_index: Arc::new(LineIndex::new("program test.aleo {}")),
            version: generation as i32,
            generation,
            file_path: Some(Arc::new(PathBuf::from("/tmp/main.leo"))),
            project: None,
            package_key,
            view_key,
            open_overlays: Arc::from([]),
            cancel_token,
        }
    }

    fn event_sink() -> crossbeam_channel::Sender<WorkerEvent> {
        let (tx, _rx) = unbounded();
        tx
    }

    #[test]
    fn coalescing_keeps_latest_snapshot_per_uri() {
        let mut pending = PendingPackageJobs::new();
        let mut pending_views = PendingViewJobs::new();
        let mut sequence = 0;
        let mut package_cache = PackageAnalysisCache::default();
        let event_tx = event_sink();
        let uri = "file:///tmp/main.leo";

        assert!(!absorb_command(
            WorkerCommand::AnalyzePackage(snapshot(uri, 1)),
            &mut pending,
            &mut pending_views,
            &mut sequence,
            &mut package_cache,
            &event_tx
        ));
        assert!(!absorb_command(
            WorkerCommand::AnalyzePackage(snapshot(uri, 2)),
            &mut pending,
            &mut pending_views,
            &mut sequence,
            &mut package_cache,
            &event_tx
        ));

        let next = take_latest_package(&mut pending).expect("pending snapshot");
        assert_eq!(next.generation, 2);
    }

    #[test]
    fn latest_updated_document_runs_first() {
        let mut pending = PendingPackageJobs::new();
        let mut pending_views = PendingViewJobs::new();
        let mut sequence = 0;
        let mut package_cache = PackageAnalysisCache::default();
        let event_tx = event_sink();

        assert!(!absorb_command(
            WorkerCommand::AnalyzePackage(snapshot("file:///tmp/a.leo", 1)),
            &mut pending,
            &mut pending_views,
            &mut sequence,
            &mut package_cache,
            &event_tx
        ));
        assert!(!absorb_command(
            WorkerCommand::AnalyzePackage(snapshot("file:///tmp/b.leo", 1)),
            &mut pending,
            &mut pending_views,
            &mut sequence,
            &mut package_cache,
            &event_tx
        ));
        assert!(!absorb_command(
            WorkerCommand::AnalyzePackage(snapshot("file:///tmp/a.leo", 2)),
            &mut pending,
            &mut pending_views,
            &mut sequence,
            &mut package_cache,
            &event_tx
        ));

        let next = take_latest_package(&mut pending).expect("pending snapshot");
        assert_eq!(next.uri.as_str(), "file:///tmp/a.leo");
        assert_eq!(next.generation, 2);
    }

    #[test]
    fn stale_snapshot_is_cancelled_before_work_starts() {
        let mut scheduler = Scheduler::new(false);
        let cancel_token = Arc::new(AtomicU64::new(2));

        scheduler.enqueue_package(snapshot_with_token("file:///tmp/main.leo", 1, cancel_token));

        let event = scheduler.events().recv_timeout(Duration::from_secs(1)).expect("worker event");

        match event {
            WorkerEvent::PackageCancelled { uri, generation, .. } => {
                assert_eq!(uri.as_str(), "file:///tmp/main.leo");
                assert_eq!(generation, 1);
            }
            other => panic!("expected cancelled event, got {other:?}"),
        }

        scheduler.shutdown();
    }
}
