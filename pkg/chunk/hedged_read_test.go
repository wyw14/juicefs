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

//nolint:errcheck
package chunk

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/juicedata/juicefs/pkg/compress"
	"github.com/juicedata/juicefs/pkg/object"
	"github.com/juicedata/juicefs/pkg/utils"
	"github.com/stretchr/testify/require"
)

var errHedgeBodyClosed = errors.New("hedge mock body closed")

// scriptedBody serves one GET response; its first Read can be delayed and is
// unblocked by Close so cancellation tests never hang.
type scriptedBody struct {
	store     *hedgeMockStore
	data      []byte
	readDelay time.Duration
	off       int
	first     bool
	closeOnce sync.Once
	closed    chan struct{}
}

func (b *scriptedBody) Read(p []byte) (int, error) {
	if b.first && b.readDelay > 0 {
		b.first = false
		select {
		case <-time.After(b.readDelay):
		case <-b.closed:
			return 0, errHedgeBodyClosed
		}
	}
	if b.off >= len(b.data) {
		return 0, io.EOF
	}
	n := copy(p, b.data[b.off:])
	b.off += n
	return n, nil
}

func (b *scriptedBody) Close() error {
	b.closeOnce.Do(func() {
		b.store.openBodies.Add(-1)
		b.store.closedBodies.Add(1)
		close(b.closed)
	})
	return nil
}

type scriptedGet struct {
	delay     time.Duration // block before Get returns a body
	readDelay time.Duration // block inside the first body Read
	data      []byte        // nil together with err simulates a failed Get
	err       error
}

// hedgeMockStore is a controllable ObjectStorage: each GET follows a scripted
// scenario selected by call sequence, and the last scenario repeats.
type hedgeMockStore struct {
	object.ObjectStorage
	mu           sync.Mutex
	script       []scriptedGet
	gets         atomic.Int64
	openedBodies atomic.Int64
	openBodies   atomic.Int64
	closedBodies atomic.Int64
}

func (s *hedgeMockStore) Get(ctx context.Context, key string, off, limit int64, getters ...object.AttrGetter) (io.ReadCloser, error) {
	seq := int(s.gets.Add(1)) - 1
	s.mu.Lock()
	var g scriptedGet
	if seq < len(s.script) {
		g = s.script[seq]
	} else {
		g = s.script[len(s.script)-1]
	}
	s.mu.Unlock()
	if g.err != nil {
		return nil, g.err
	}
	if g.delay > 0 {
		select {
		case <-time.After(g.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	s.openedBodies.Add(1)
	s.openBodies.Add(1)
	return &scriptedBody{
		store:     s,
		data:      g.data,
		readDelay: g.readDelay,
		first:     true,
		closed:    make(chan struct{}),
	}, nil
}

func (s *hedgeMockStore) assertAllBodiesClosed(t *testing.T) {
	require.Zero(t, s.openBodies.Load(), "an opened body was not closed")
	require.Equal(t, s.openedBodies.Load(), s.closedBodies.Load(), "every opened body must be closed exactly once")
}

func newHedgeMock(script ...scriptedGet) *hedgeMockStore {
	mem, _ := object.CreateStorage("mem", "", "", "", "")
	return &hedgeMockStore{ObjectStorage: mem, script: script}
}

func newHedgeTestStore(t *testing.T, mock *hedgeMockStore, conf Config) *cachedStore {
	conf.CacheDir = "memory"
	conf.CacheSize = 100 << 20
	conf.CacheEviction = EvictionNone
	conf.HedgeRead = true
	if conf.HedgeReadThreshold == 0 {
		conf.HedgeReadThreshold = 20 * time.Millisecond
	}
	if conf.MaxDownload == 0 {
		conf.MaxDownload = 4
	}
	if conf.MaxHedgeDownloads == 0 {
		conf.MaxHedgeDownloads = 2
	}
	if conf.BlockSize == 0 {
		conf.BlockSize = 64 << 10
	}
	if conf.GetTimeout == 0 {
		conf.GetTimeout = time.Second
	}
	conf.AutoCreate = false
	store := NewCachedStore(mock, conf, nil).(*cachedStore)
	require.NotNil(t, store.hedge)
	require.Equal(t, conf.MaxHedgeDownloads, cap(store.hedgeDownload))
	return store
}

func hedgeCounter(s *cachedStore, attempt, outcome string) float64 {
	return toFloat64(s.hedge.attempts.WithLabelValues(attempt, outcome))
}

// Slow primary GET, fast hedged copy: the hedge produces the first valid
// result and the primary is canceled; every opened body is closed.
func TestHedgeReadSlowPrimaryFastCopy(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	mock := newHedgeMock(
		scriptedGet{delay: 300 * time.Millisecond, data: want},
		scriptedGet{delay: time.Millisecond, data: want},
	)
	store := newHedgeTestStore(t, mock, Config{})

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "slow-primary", page, false, false))
	require.Equal(t, int64(2), mock.gets.Load(), "exactly one hedged GET may be added")
	require.Equal(t, want, page.Data)
	require.Zero(t, liveHedgePages.Load(), "attempt pages must be released")
	mock.assertAllBodiesClosed(t)
	require.Equal(t, 1.0, toFloat64(store.hedge.launched))
	require.Equal(t, 1.0, toFloat64(store.hedge.wins))
	require.Equal(t, 1.0, toFloat64(store.hedge.cancels.WithLabelValues(hedgeCancelSuperseded)))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagHedge, hedgeOutcomeSuccess))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagPrimary, hedgeOutcomeCanceled))
	require.Equal(t, float64(bs), toFloat64(store.hedge.bytes.WithLabelValues(hedgeTagHedge, hedgeOutcomeSuccess)))
	page.Release()
}

// A fast primary finishes before the threshold: no hedge is started.
func TestHedgeReadFastPrimaryNoHedge(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'p'}, bs)
	mock := newHedgeMock(scriptedGet{data: want})
	store := newHedgeTestStore(t, mock, Config{})

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "fast-primary", page, false, false))
	require.Equal(t, int64(1), mock.gets.Load())
	require.Equal(t, want, page.Data)
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	require.Zero(t, toFloat64(store.hedge.launched))
	require.Zero(t, toFloat64(store.hedge.wins))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagPrimary, hedgeOutcomeSuccess))
	page.Release()
}

// The hedge starts after the threshold but the primary still returns a valid
// result first: the hedge is canceled and the primary win publishes once.
func TestHedgeReadPrimaryWinsAfterHedgeStarted(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'p'}, bs)
	mock := newHedgeMock(
		scriptedGet{delay: 35 * time.Millisecond, data: want},
		scriptedGet{delay: 300 * time.Millisecond, data: want},
	)
	store := newHedgeTestStore(t, mock, Config{})

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "primary-comeback", page, false, false))
	require.Equal(t, int64(2), mock.gets.Load())
	require.Equal(t, want, page.Data)
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	require.Equal(t, 1.0, toFloat64(store.hedge.launched))
	require.Zero(t, toFloat64(store.hedge.wins))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagPrimary, hedgeOutcomeSuccess))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagHedge, hedgeOutcomeCanceled))
	require.Equal(t, 1.0, toFloat64(store.hedge.cancels.WithLabelValues(hedgeCancelSuperseded)))
	page.Release()
}

// Fast short read followed by a slow but valid hedge: the early failure must
// not preempt the later valid result, and only the winner writes the cache.
func TestHedgeReadFastShortSlowGood(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	short := bytes.Repeat([]byte{'x'}, bs/2)
	mock := newHedgeMock(
		scriptedGet{data: short},
		scriptedGet{delay: 60 * time.Millisecond, data: want},
	)
	store := newHedgeTestStore(t, mock, Config{})

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "fast-bad", page, true, false))
	require.Equal(t, int64(2), mock.gets.Load())
	require.Equal(t, want, page.Data)
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	cnt, _ := store.bcache.stats()
	require.Equal(t, int64(1), cnt, "only the winning attempt may populate the cache")
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagPrimary, hedgeOutcomeShortRead))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagHedge, hedgeOutcomeSuccess))
	require.Equal(t, 1.0, toFloat64(store.hedge.wins))
	page.Release()
}

// Compressed block: the primary returns full-length garbage that fails
// decompression; each attempt uses its own compressed and plain pages, and the
// slow valid hedge still wins.
func TestHedgeReadCompressedBadPrimary(t *testing.T) {
	const bs = 64 << 10
	comp := compress.NewCompressor("lz4")
	want := bytes.Repeat([]byte{'z'}, bs)
	bound := comp.CompressBound(bs)
	cbuf := make([]byte, bound)
	n, err := comp.Compress(cbuf, want)
	require.NoError(t, err)
	good := append([]byte{}, cbuf[:n]...)
	bad := bytes.Repeat([]byte{0xff}, len(good))

	mock := newHedgeMock(
		scriptedGet{data: bad},
		scriptedGet{delay: 60 * time.Millisecond, data: good},
	)
	store := newHedgeTestStore(t, mock, Config{Compress: "lz4"})

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "compressed-bad", page, false, false))
	require.Equal(t, int64(2), mock.gets.Load())
	require.Equal(t, want, page.Data)
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagPrimary, hedgeOutcomeDecompress))
	require.Equal(t, 1.0, hedgeCounter(store, hedgeTagHedge, hedgeOutcomeSuccess))
	page.Release()
}

// Both attempts fail: the wrapped error keeps the underlying reason, nothing
// is cached, the target page is untouched and all pages/bodies are released.
func TestHedgeReadBothFail(t *testing.T) {
	const bs = 64 << 10
	boom := errors.New("object store boom")
	mock := newHedgeMock(
		scriptedGet{err: boom},
		scriptedGet{err: boom},
	)
	store := newHedgeTestStore(t, mock, Config{})

	residue := bytes.Repeat([]byte{'r'}, bs)
	page := NewOffPage(bs)
	copy(page.Data, residue)
	err := store.load(context.Background(), "both-fail", page, true, false)
	require.Error(t, err)
	require.ErrorIs(t, err, boom)
	require.Contains(t, err.Error(), "both-fail")
	require.Equal(t, int64(2), mock.gets.Load())
	require.Equal(t, residue, page.Data, "a failed load must never write the caller's page")
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	cnt, _ := store.bcache.stats()
	require.Zero(t, cnt)
	require.Equal(t, 2.0, hedgeCounter(store, hedgeTagPrimary, hedgeOutcomeError)+
		hedgeCounter(store, hedgeTagHedge, hedgeOutcomeError))
	page.Release()
}

// Stable failure priority: a primary GET timeout outranks the hedge's short
// read, so the timeout reason is returned regardless of completion ordering.
func TestHedgeReadErrorPriority(t *testing.T) {
	const bs = 64 << 10
	short := bytes.Repeat([]byte{'x'}, bs/2)
	mock := newHedgeMock(
		scriptedGet{delay: time.Second, data: bytes.Repeat([]byte{'p'}, bs)},
		scriptedGet{data: short},
	)
	conf := Config{GetTimeout: 40 * time.Millisecond}
	store := newHedgeTestStore(t, mock, conf)

	page := NewOffPage(bs)
	err := store.load(context.Background(), "priority", page, false, false)
	require.Error(t, err)
	require.ErrorIs(t, err, utils.ErrFuncTimeout)
	require.Equal(t, int64(2), mock.gets.Load())
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	page.Release()
}

// Parent cancellation while both GETs are in flight: context.Canceled is
// returned, both bodies close and no page leaks.
func TestHedgeReadParentCancel(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	mock := newHedgeMock(
		scriptedGet{readDelay: 5 * time.Second, data: want},
		scriptedGet{readDelay: 5 * time.Second, data: want},
	)
	conf := Config{GetTimeout: 10 * time.Second}
	store := newHedgeTestStore(t, mock, conf)

	cctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	page := NewOffPage(bs)
	go func() { done <- store.load(cctx, "cancel-me", page, false, false) }()

	require.Eventually(t, func() bool {
		return mock.gets.Load() == 2 && mock.openBodies.Load() == 2
	}, time.Second, 5*time.Millisecond)
	cancel()
	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("canceled load did not return")
	}
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	require.Equal(t, 2.0, toFloat64(store.hedge.cancels.WithLabelValues(hedgeCancelParent)))
	page.Release()
}

// All MaxDownload slots are occupied: the hedge must give up immediately
// instead of blocking while holding the primary's download token.
func TestHedgeReadNoDownloadToken(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	mock := newHedgeMock(
		scriptedGet{delay: 100 * time.Millisecond, data: want},
		scriptedGet{data: want},
	)
	store := newHedgeTestStore(t, mock, Config{MaxDownload: 2, MaxHedgeDownloads: 2})

	// Occupy one slot; the primary takes the last one, leaving none for hedge.
	store.currentDownload <- struct{}{}
	defer func() { <-store.currentDownload }()

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "no-download-token", page, false, false))
	require.Equal(t, int64(1), mock.gets.Load())
	require.Equal(t, want, page.Data)
	require.Zero(t, toFloat64(store.hedge.launched))
	require.Equal(t, 1.0, toFloat64(store.hedge.abandoned.WithLabelValues(hedgeAbandonNoDownloadToken)))
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	page.Release()
}

// The separate global hedge concurrency budget is exhausted: the hedge is
// abandoned and the acquired download token is returned immediately.
func TestHedgeReadNoHedgeToken(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	mock := newHedgeMock(
		scriptedGet{delay: 100 * time.Millisecond, data: want},
		scriptedGet{data: want},
	)
	store := newHedgeTestStore(t, mock, Config{MaxDownload: 4, MaxHedgeDownloads: 1})

	store.hedgeDownload <- struct{}{} // exhaust the single hedge slot
	defer func() { <-store.hedgeDownload }()

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "no-hedge-token", page, false, false))
	require.Equal(t, int64(1), mock.gets.Load())
	require.Equal(t, want, page.Data)
	require.Zero(t, toFloat64(store.hedge.launched))
	require.Equal(t, 1.0, toFloat64(store.hedge.abandoned.WithLabelValues(hedgeAbandonNoHedgeToken)))
	require.Equal(t, len(store.currentDownload), 0, "download token acquired for the abandoned hedge must be returned")
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
	page.Release()
}

// Default configuration keeps hedging disabled: a slow GET never gets a peer.
func TestHedgeReadDisabledByDefault(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	mock := newHedgeMock(scriptedGet{delay: 50 * time.Millisecond, data: want})
	conf := defaultConf
	conf.CacheDir = "memory"
	conf.BlockSize = bs
	conf.HedgeRead = false
	store := NewCachedStore(mock, conf, nil).(*cachedStore)
	require.Nil(t, store.hedge)

	page := NewOffPage(bs)
	require.NoError(t, store.load(context.Background(), "disabled", page, false, false))
	require.Equal(t, int64(1), mock.gets.Load())
	require.Equal(t, want, page.Data)
	mock.assertAllBodiesClosed(t)
	page.Release()
}

// singleflight: one block with many concurrent waiters triggers only one
// group of attempts (at most two GETs), and every waiter gets the bytes.
func TestHedgeReadSingleflightDedup(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'s'}, bs)
	mock := newHedgeMock(
		scriptedGet{delay: 200 * time.Millisecond, data: want},
		scriptedGet{delay: time.Millisecond, data: want},
	)
	store := newHedgeTestStore(t, mock, Config{})

	const waiters = 8
	r := sliceForRead(1, bs, store)
	key := r.key(0)
	require.NoError(t, mock.ObjectStorage.Put(context.Background(), key, bytes.NewReader(want)))

	var wg sync.WaitGroup
	errs := make(chan error, waiters)
	for i := 0; i < waiters; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			page := NewOffPage(bs)
			defer page.Release()
			n, err := store.NewReader(1, bs).ReadAt(context.Background(), page, 0)
			if err != nil {
				errs <- err
				return
			}
			if n != bs || !bytes.Equal(page.Data, want) {
				errs <- fmt.Errorf("unexpected read result n=%d", n)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	require.Equal(t, int64(2), mock.gets.Load(), "singleflight waiters must share one attempt group")
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
}

// Upper-layer retries cannot multiply hedging: every load adds at most one
// extra GET even when every attempt fails.
func TestHedgeReadRetryBounded(t *testing.T) {
	const bs = 64 << 10
	want := bytes.Repeat([]byte{'g'}, bs)
	boom := errors.New("boom")
	mock := newHedgeMock(
		scriptedGet{err: boom},
		scriptedGet{err: boom},
		scriptedGet{err: boom},
		scriptedGet{err: boom},
		scriptedGet{data: want},
	)
	store := newHedgeTestStore(t, mock, Config{})

	calls := 0
	var page *Page
	for {
		page = NewOffPage(bs)
		err := store.load(context.Background(), "retry-bounded", page, false, false)
		calls++
		if err == nil {
			break
		}
		page.Release()
		require.Less(t, calls, 10)
	}
	require.Equal(t, want, page.Data)
	page.Release()
	require.LessOrEqual(t, mock.gets.Load(), int64(2*calls), "each retried load may add at most one hedged GET")
	require.Zero(t, liveHedgePages.Load())
	mock.assertAllBodiesClosed(t)
}
