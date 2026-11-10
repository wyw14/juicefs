/*
 * JuiceFS, Copyright 2020 Juicedata, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package chunk

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/juicedata/juicefs/pkg/object"
	"github.com/juicedata/juicefs/pkg/utils"
	"github.com/prometheus/client_golang/prometheus"
)

// defaultHedgeReadThreshold is applied when hedged reads are enabled without an
// explicit threshold.
const defaultHedgeReadThreshold = 200 * time.Millisecond

// Sentinel errors keep the stable failure-priority ordering independent of the
// wording of wrapped error messages.
var (
	errHedgeShortRead  = errors.New("short read")
	errHedgeDecompress = errors.New("decompress failed")
)

const (
	hedgeTagPrimary = "primary"
	hedgeTagHedge   = "hedge"

	hedgeOutcomeSuccess    = "success"
	hedgeOutcomeCanceled   = "canceled"
	hedgeOutcomeTimeout    = "timeout"
	hedgeOutcomeShortRead  = "short_read"
	hedgeOutcomeDecompress = "decompress_error"
	hedgeOutcomeError      = "error"

	hedgeAbandonNoDownloadToken = "no_download_token"
	hedgeAbandonNoHedgeToken    = "no_hedge_token"

	hedgeCancelSuperseded = "superseded"
	hedgeCancelParent     = "parent_canceled"
)

// liveHedgePages counts off-heap pages owned by in-flight hedged load
// attempts. Every load must leave it at zero; tests use it to prove that
// failed and canceled attempts release their pages instead of leaking them.
var liveHedgePages atomic.Int64

// hedgeMetrics describes every GET attempt, not only the logically successful
// one, so the extra traffic and latency introduced by hedging stay visible.
type hedgeMetrics struct {
	launched  prometheus.Counter
	wins      prometheus.Counter
	abandoned *prometheus.CounterVec
	cancels   *prometheus.CounterVec
	attempts  *prometheus.CounterVec
	bytes     *prometheus.CounterVec
	latency   *prometheus.HistogramVec
}

func newHedgeMetrics() *hedgeMetrics {
	return &hedgeMetrics{
		launched: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "blockcache_hedge_launched",
			Help: "number of hedged GET requests started for slow full-block downloads",
		}),
		wins: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "blockcache_hedge_wins",
			Help: "number of blocks whose valid result came from the hedged GET instead of the primary one",
		}),
		abandoned: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "blockcache_hedge_abandoned",
			Help: "number of hedged GETs given up because a token was not available immediately",
		}, []string{"reason"}),
		cancels: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "blockcache_hedge_cancel",
			Help: "number of GET attempts canceled before completion",
		}, []string{"reason"}),
		attempts: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "blockcache_hedge_attempts",
			Help: "number of GET attempts started by hedged block loads, by attempt role and outcome",
		}, []string{"attempt", "outcome"}),
		bytes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "blockcache_hedge_attempt_bytes",
			Help: "bytes actually read by each hedged-load GET attempt, including failed or canceled ones",
		}, []string{"attempt", "outcome"}),
		latency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "blockcache_hedge_attempt_hist_seconds",
			Help:    "latency of each hedged-load GET attempt",
			Buckets: prometheus.ExponentialBuckets(0.01, 1.5, 25),
		}, []string{"attempt", "outcome"}),
	}
}

func (m *hedgeMetrics) register(reg prometheus.Registerer) {
	if reg == nil {
		return
	}
	reg.MustRegister(m.launched, m.wins, m.abandoned, m.cancels, m.attempts, m.bytes, m.latency)
}

// hedgeBody guards the io.ReadCloser owned by one attempt. Closing the guard
// while a Get or Read is still in flight unblocks it through the body's own
// Close, and a body arriving after cancellation is closed immediately.
type hedgeBody struct {
	mu     sync.Mutex
	body   io.ReadCloser
	closed bool
}

func (b *hedgeBody) set(rc io.ReadCloser) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		_ = rc.Close()
		return
	}
	b.body = rc
	b.mu.Unlock()
}

func (b *hedgeBody) close() {
	b.mu.Lock()
	rc := b.body
	b.body = nil
	b.closed = true
	b.mu.Unlock()
	if rc != nil {
		_ = rc.Close()
	}
}

// hedgeAttempt owns everything two concurrent GETs must not share: its own
// read page, its own decompression page and its own request metadata. It never
// writes the caller's target page; the orchestrator publishes the winner once.
type hedgeAttempt struct {
	tag        string
	compressed bool
	size       int

	buf  *Page // raw bytes returned by GET
	out  *Page // decompressed bytes, only allocated for compressed blocks
	body *hedgeBody

	res   getResult
	start time.Time
}

func newHedgeAttempt(tag string, compressed bool, size, bound int) *hedgeAttempt {
	a := &hedgeAttempt{
		tag:        tag,
		compressed: compressed,
		size:       size,
		body:       &hedgeBody{},
	}
	if compressed {
		a.buf = NewOffPage(bound)
		a.out = NewOffPage(size)
		liveHedgePages.Add(2)
	} else {
		a.buf = NewOffPage(size)
		liveHedgePages.Add(1)
	}
	return a
}

func (a *hedgeAttempt) release() {
	a.buf.Release()
	liveHedgePages.Add(-1)
	if a.out != nil {
		a.out.Release()
		liveHedgePages.Add(-1)
	}
}

// page returns the validated plain bytes of the attempt.
func (a *hedgeAttempt) page() *Page {
	if a.compressed {
		return a.out
	}
	return a.buf
}

type hedgeAttemptResult struct {
	tag string
	err error
}

// loadHedged loads one immutable, complete block. The primary GET starts at
// once; if it has not finished after HedgeReadThreshold, at most one hedged GET
// is added. The first attempt that passes the full read and decompression
// length checks wins; an earlier short read or decompression error can never
// preempt a later valid result.
func (store *cachedStore) loadHedged(ctx context.Context, key string, page *Page, cache bool, forceCache bool) (err error) {
	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("recovered from %s", e)
		}
	}()

	bound := store.compressor.CompressBound(len(page.Data))
	compressed := bound > len(page.Data)

	// The primary attempt occupies a real download slot, exactly like load().
	store.currentDownload <- struct{}{}
	defer func() { <-store.currentDownload }()

	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make(chan hedgeAttemptResult, 2)
	primary := newHedgeAttempt(hedgeTagPrimary, compressed, len(page.Data), bound)
	go store.runHedgeAttempt(attemptCtx, ctx, key, primary, false, results)

	timer := time.NewTimer(store.conf.HedgeReadThreshold)
	defer timer.Stop()

	var (
		hedge        *hedgeAttempt
		hedgeRunning bool
		hedgeSlot    bool // extra currentDownload token held by the hedge
		hedgeDecided bool // the hedge was launched or permanently abandoned
	)
	defer func() {
		if hedgeSlot {
			<-store.currentDownload
		}
		if hedgeRunning {
			<-store.hedgeDownload
		}
	}()

	var (
		errs     [2]error
		failures int
		reported [2]bool // results already consumed from each attempt slot
	)

	for {
		select {
		case r := <-results:
			idx := 0
			if r.tag == hedgeTagHedge {
				idx = 1
			}
			reported[idx] = true
			if r.err == nil {
				// Stop the other attempt before touching the caller's page, so
				// the winning result is published exactly once.
				cancel()
				var loser *hedgeAttempt
				loserIdx := -1
				winner := primary
				if r.tag == hedgeTagHedge {
					loser = primary
					loserIdx = 0
					winner = hedge
					store.hedge.wins.Add(1)
				} else if hedge != nil {
					loser = hedge
					loserIdx = 1
				}
				if loser != nil {
					loser.body.close()
					if !reported[loserIdx] {
						<-results // wait for the still-running attempt to release its resources
					}
				}
				copy(page.Data, winner.page().Data[:len(page.Data)])
				primary.release()
				if hedge != nil {
					hedge.release()
				}
				// Only the validated winner is allowed to populate the cache.
				if cache {
					store.bcache.cache(key, page, forceCache, !store.conf.OSCache)
				}
				return nil
			}

			errs[idx] = r.err
			failures++
			// A failure before the threshold never preempts a possible rescue:
			// keep waiting for the timer to launch the hedge.
			if failures == 1 && !hedgeRunning && !hedgeDecided {
				continue
			}
			expected := 1
			if hedgeRunning {
				expected = 2
			}
			if failures >= expected {
				cancel()
				primary.release()
				if hedge != nil {
					hedge.release()
				}
				return fmt.Errorf("get %s: %w", key, pickHedgeErr(errs[:expected]))
			}
			// A failed early return never wins: keep waiting for the other attempt.

		case <-timer.C:
			if hedgeDecided || ctx.Err() != nil {
				continue
			}
			if a := store.startHedge(attemptCtx, ctx, key, compressed, len(page.Data), bound, results); a != nil {
				hedge = a
				hedgeRunning = true
				hedgeSlot = true
				store.hedge.launched.Add(1)
			}
			hedgeDecided = true
			// The hedge was abandoned while the primary has already failed:
			// no further attempt can change the result.
			if !hedgeRunning && failures > 0 {
				cancel()
				primary.release()
				return fmt.Errorf("get %s: %w", key, errs[0])
			}

		case <-ctx.Done():
			cancel()
			primary.body.close()
			if hedge != nil {
				hedge.body.close()
			}
			expected := 1
			if hedgeRunning {
				expected = 2
			}
			for i := 0; i < expected; i++ {
				if !reported[i] {
					<-results
				}
			}
			primary.release()
			if hedge != nil {
				hedge.release()
			}
			return ctx.Err()
		}
	}
}

// startHedge acquires both budgets without blocking: the primary attempt is
// already holding a download slot, so waiting here could deadlock when
// MaxDownload is fully occupied. Failure to get either token immediately means
// the hedge is abandoned and the primary attempt continues alone.
func (store *cachedStore) startHedge(attemptCtx, parent context.Context, key string, compressed bool, size, bound int,
	results chan hedgeAttemptResult) *hedgeAttempt {
	select {
	case store.currentDownload <- struct{}{}:
	default:
		store.hedge.abandoned.WithLabelValues(hedgeAbandonNoDownloadToken).Add(1)
		return nil
	}
	select {
	case store.hedgeDownload <- struct{}{}:
	default:
		<-store.currentDownload
		store.hedge.abandoned.WithLabelValues(hedgeAbandonNoHedgeToken).Add(1)
		return nil
	}
	a := newHedgeAttempt(hedgeTagHedge, compressed, size, bound)
	go store.runHedgeAttempt(attemptCtx, parent, key, a, true, results)
	return a
}

// runHedgedGet runs fn under the configured GET timeout. It only returns after
// fn has fully exited, so the private pages of the attempt are never freed
// while the GET goroutine still touches them.
func (store *cachedStore) runHedgedGet(ctx context.Context, body *hedgeBody, fn func(context.Context) error) error {
	getCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan error, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		done <- fn(getCtx)
	}()
	timer := time.NewTimer(store.conf.GetTimeout)
	defer timer.Stop()
	var err error
	select {
	case <-getCtx.Done():
		err = getCtx.Err()
	case err = <-done:
	case <-timer.C:
		err = fmt.Errorf("timeout after %s: %w", store.conf.GetTimeout, utils.ErrFuncTimeout)
	}
	cancel()
	// Unblock a read still in flight and guarantee every opened body is
	// closed; close() is idempotent with the in-fn Close.
	body.close()
	wg.Wait()
	return err
}

// runHedgeAttempt performs one GET with private buffers and request metadata,
// validates the result and reports its own traffic, latency and cancel reason.
func (store *cachedStore) runHedgeAttempt(attemptCtx, parent context.Context, key string, a *hedgeAttempt, hedged bool,
	results chan<- hedgeAttemptResult) {
	limit := store.downLimit
	if hedged {
		limit = store.hedgeLimit
	}
	// The compressed size is unknown until the body arrives, so compressed
	// blocks are charged after the read, mirroring the classic load path.
	if limit != nil && !a.compressed {
		limit.Wait(int64(a.size))
	}

	a.start = time.Now()
	err := store.runHedgedGet(attemptCtx, a.body, func(cCtx context.Context) error {
		in, getErr := store.storage.Get(cCtx, key, 0, -1,
			object.WithRequestID(&a.res.reqID), object.WithStorageClass(&a.res.sc))
		if getErr != nil {
			return getErr
		}
		a.body.set(in)
		n, readErr := io.ReadFull(in, a.buf.Data)
		a.res.n = n
		if readErr != nil && !(a.compressed && readErr == io.ErrUnexpectedEOF) {
			_ = in.Close()
			if errors.Is(readErr, io.EOF) || errors.Is(readErr, io.ErrUnexpectedEOF) {
				return fmt.Errorf("read %s fully: %d < %d: %w", key, n, a.size, errHedgeShortRead)
			}
			return readErr
		}
		_ = in.Close()
		if a.compressed {
			dn, derr := store.compressor.Decompress(a.out.Data, a.buf.Data[:n])
			if derr != nil {
				return fmt.Errorf("decompress %s: %w", key, errHedgeDecompress)
			}
			if dn < a.size {
				return fmt.Errorf("read %s fully: %d < %d: %w", key, dn, a.size, errHedgeShortRead)
			}
		} else if n < a.size {
			return fmt.Errorf("read %s fully: %d < %d: %w", key, n, a.size, errHedgeShortRead)
		}
		return nil
	})
	// Defensive: runHedgedGet already closed the body on every exit path.
	a.body.close()

	latency := time.Since(a.start)
	if limit != nil && a.compressed {
		limit.Wait(int64(a.res.n))
	}

	// Per-attempt object metrics are recorded for every outcome.
	res := getResult{sc: object.DefaultStorageClass}
	if err == nil {
		res = a.res
	}
	logRequest("GET", key, "HEDGE("+a.tag+") ", res.reqID, err, latency)
	store.objectDataBytes.WithLabelValues("GET", res.sc).Add(float64(res.n))
	store.objectReqsHistogram.WithLabelValues("GET", res.sc).Observe(latency.Seconds())
	if err != nil {
		store.objectReqErrors.Add(1)
	}

	outcome := hedgeOutcome(err)
	store.hedge.attempts.WithLabelValues(a.tag, outcome).Add(1)
	store.hedge.bytes.WithLabelValues(a.tag, outcome).Add(float64(a.res.n))
	store.hedge.latency.WithLabelValues(a.tag, outcome).Observe(latency.Seconds())
	if errors.Is(err, context.Canceled) {
		reason := hedgeCancelSuperseded
		if parent.Err() != nil {
			reason = hedgeCancelParent
		}
		store.hedge.cancels.WithLabelValues(reason).Add(1)
	}
	results <- hedgeAttemptResult{tag: a.tag, err: err}
}

func hedgeOutcome(err error) string {
	switch {
	case err == nil:
		return hedgeOutcomeSuccess
	case errors.Is(err, context.Canceled):
		return hedgeOutcomeCanceled
	case errors.Is(err, utils.ErrFuncTimeout):
		return hedgeOutcomeTimeout
	case errors.Is(err, errHedgeShortRead):
		return hedgeOutcomeShortRead
	case errors.Is(err, errHedgeDecompress):
		return hedgeOutcomeDecompress
	default:
		return hedgeOutcomeError
	}
}

// hedgeErrRank defines a stable, scheduling-independent priority for failures.
// When every attempt fails, the lowest-ranked reason is returned; ties keep
// the primary attempt's error.
func hedgeErrRank(err error) int {
	switch {
	case errors.Is(err, context.Canceled):
		return 0
	case errors.Is(err, utils.ErrFuncTimeout):
		return 1
	case errors.Is(err, errHedgeShortRead):
		return 2
	case errors.Is(err, errHedgeDecompress):
		return 3
	default:
		return 4
	}
}

func pickHedgeErr(errs []error) error {
	chosen := errs[0]
	for _, e := range errs[1:] {
		if e != nil && hedgeErrRank(e) < hedgeErrRank(chosen) {
			chosen = e
		}
	}
	return chosen
}
