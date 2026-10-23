# Background ingestion design

Status: accepted for the V3 local/controlled-demo architecture, 2026-10-16.

## Decision

Use PostgreSQL as the durable ingestion queue and run one separately supervised
Python worker service. Jobs are claimed with `FOR UPDATE SKIP LOCKED`, have a
bounded lease, and are retried by setting `available_at` with exponential
backoff. The worker and API share versioned schemas and the existing document
repository, parser, embedding adapter, authorization model, and database.

The API stores an upload in a tenant-scoped job row and returns immediately.
The opaque extension and bounded upload bytes are retained only while work can
still be attempted. Payload bytes are set to `NULL` on completion or terminal
failure. Parsing, embedding, document/version/chunk persistence, job completion,
and payload purge commit as one transaction. A process crash before that commit
therefore leaves no partial document version; an expired lease makes the job
claimable again.

## Why not Redis + Celery

Celery provides mature scheduling, concurrency controls, and broker adapters,
but Redis would create another stateful service, credential, backup target, and
failure/reconciliation boundary. The authoritative result still belongs in
PostgreSQL, so a Celery acknowledgement can race the database commit. This
repository currently has one API instance, batches are limited to 20 files,
uploads are limited to 10 MB, and no measured load requires distributed fanout.

A PostgreSQL queue reuses the existing operational dependency, can atomically
publish status with document data, and is adequate for the measured workload.
Reconsider Celery or another broker only after measurements show PostgreSQL
claim contention or worker throughput cannot meet an approved service target.
Kafka, Spark, Airflow, and Kubernetes are not justified for this workload.

## Contracts and limits

- States: `queued`, `processing`, `completed`, `failed`.
- Job types: `ingest` and `reindex`.
- Idempotency key: unique per tenant; reuse with a different request hash is a
  conflict. Content hashes also preserve document-version idempotency.
- Default maximum attempts: 3. Retry delay is exponential and capped at one hour.
- A processing lease permits crash recovery. Claims never cross tenant data;
  public reads are filtered by the authenticated tenant.
- Parser/data-quality failures are terminal. Transient embedding or database
  failures are retryable and only expose fixed error categories/messages.
- Reindexing targets an existing tenant-owned immutable document version and
  updates all of its vectors and embedding metadata transactionally.
- Embedding name and implementation/package version are stored on every
  document version. The fixed pgvector dimension remains 384.
- Operational metrics are derived from durable job rows and contain counts and
  timings only—never filenames, text, source keys, tokens, or embeddings.

## Deliberate limits

The worker polls PostgreSQL; there is no push notification or autoscaling. One
worker process and configurable low concurrency are the supported Compose
topology. Upload payloads are temporarily stored in PostgreSQL rather than an
object store, which is acceptable only under the existing size/batch limits.
There is no claim that this design supports high-volume production ingestion.
