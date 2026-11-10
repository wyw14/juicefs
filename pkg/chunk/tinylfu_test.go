/*
 * JuiceFS, Copyright 2026 Juicedata, Inc.
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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/juicedata/juicefs/pkg/object"
	"github.com/stretchr/testify/require"
)

const tlBlockSize = 1024

func tlTestConf(t *testing.T) Config {
	conf := defaultConf
	conf.CacheEviction = EvictionTinyLFU
	conf.CacheScanInterval = -1 // disable periodic scan, tests trigger scans explicitly
	conf.FreeSpace = 0.00001
	conf.CacheDir = t.TempDir()
	conf.BlockSize = 1 << 20
	return conf
}

func newTLMetrics() *cacheManagerMetrics {
	m := new(cacheManagerMetrics)
	m.initMetrics()
	return m
}

func newTLDiskCache(t *testing.T, conf *Config, capacity, maxItems int64) *diskCache {
	m := newTLMetrics()
	dir := filepath.Join(conf.CacheDir, "diskcache") + string(filepath.Separator)
	c := newDiskCache(m, dir, capacity, maxItems, 16, conf, nil)
	require.NotNil(t, c)
	require.NotNil(t, c.admit)
	// Platforms without an inode counter (e.g. Windows) reset maxItems to
	// unlimited during construction; pin it explicitly for deterministic tests.
	c.maxItems = maxItems
	t.Cleanup(func() { _ = os.RemoveAll(conf.CacheDir) })
	return c
}

func tlKey(id uint64, indx uint32, size uint32) string {
	return fmt.Sprintf("chunks/%d/%d/%d_%d_%d", id/1000/1000, id/1000, id, indx, size)
}

func tlPage(size int) *Page {
	return NewPage(make([]byte, size))
}

func waitCacheIdle(t *testing.T, c *diskCache) {
	require.Eventually(t, func() bool {
		c.Lock()
		n := len(c.pages)
		c.Unlock()
		return n == 0
	}, 5*time.Second, 5*time.Millisecond)
}

func TestTinyLFUSketchSaturateAndDecay(t *testing.T) {
	s := newFrequencySketch(64)
	require.Equal(t, 64, int(s.width))
	k := cacheKey{id: 1, indx: 0, size: 10}

	// 4-bit counters saturate at 15 and never overflow.
	for i := 0; i < 100; i++ {
		s.add(k)
	}
	require.Equal(t, uint8(sketchCounterMax), s.estimate(k))

	// Decay halves counters; the triggering add leaves the saturated
	// counter at floor(15/2) = 7.
	decays := s.decays
	for s.decays == decays {
		s.add(k)
	}
	require.Equal(t, uint8(7), s.estimate(k))
	require.GreaterOrEqual(t, s.decays, uint64(1))

	// Every packed nibble always stays inside [0, 15].
	for _, row := range s.table {
		for _, word := range row {
			for i := 0; i < 16; i++ {
				require.LessOrEqual(t, uint8((word>>(i*4))&0xf), uint8(15))
			}
		}
	}

	// Repeated decay/observe cycles keep every counter bounded and never
	// overflow across many generations of halving.
	decaysBefore := s.decays
	for i := 0; i < 10; i++ {
		s.add(k)
		for s.additions != 0 {
			s.add(k)
		}
	}
	require.Greater(t, s.decays, decaysBefore)
	require.LessOrEqual(t, s.estimate(k), uint8(15))
}

func TestTinyLFUSketchDecayOddValue(t *testing.T) {
	s := newFrequencySketch(64)
	k := cacheKey{id: 7, indx: 1, size: 3}
	for i := 0; i < 3; i++ {
		s.add(k)
	}
	require.Equal(t, uint8(3), s.estimate(k))
	s.mu.Lock()
	s.decayLocked()
	s.mu.Unlock()
	require.Equal(t, uint8(1), s.estimate(k))
}

func TestTinyLFUSketchHardMemoryCap(t *testing.T) {
	// An arbitrarily large cache must still get a bounded sketch.
	big := newTinyLFUPolicy(1<<40, 1<<20)
	require.Equal(t, maxSketchWidth, int(big.sketch.width))
	require.LessOrEqual(t, big.sketch.memoryBytes(), 2<<20)
	require.LessOrEqual(t, big.windowCap, maxWindowItems)

	small := newTinyLFUPolicy(1<<20, 1<<20)
	require.Equal(t, minSketchWidth, int(small.sketch.width))
	require.Equal(t, minWindowItems, small.windowCap)

	// A huge number of distinct keys cannot grow the allocation.
	before := big.sketch.memoryBytes()
	for i := uint64(0); i < 200000; i++ {
		big.observe(cacheKey{id: i, indx: 0, size: 1})
	}
	require.Equal(t, before, big.sketch.memoryBytes())
}

func TestTinyLFUIndexBasics(t *testing.T) {
	conf := tlTestConf(t)
	idx, err := NewKeyIndex(&conf)
	require.NoError(t, err)
	tl, ok := idx.(*tinylfuEviction)
	require.True(t, ok)
	require.Equal(t, EvictionTinyLFU, tl.name())

	k := cacheKey{id: 1, indx: 0, size: 10}
	tl.add(k, cacheItem{size: 1024, atime: 1})
	require.NotNil(t, tl.get(k))
	require.NotZero(t, tl.peekAtime(k))

	// Staging blocks coexist but are not read blocks.
	ks := cacheKey{id: 2, indx: 0, size: 9}
	tl.add(ks, cacheItem{size: -9, atime: 1})
	require.Equal(t, 2, tl.len())
	require.Equal(t, 1, tl.readLen())

	// Staging entries are protected from ordinary removal.
	require.Nil(t, tl.remove(ks, false))
	require.Equal(t, 2, tl.len())
	require.NotNil(t, tl.remove(ks, true))
	require.Equal(t, 1, tl.len())

	// Scan reset keeps a read-only snapshot with the shared policy.
	snap := tl.reset()
	require.Equal(t, 1, snap.len())
	require.Equal(t, 0, tl.len())
	require.Panics(t, func() {
		snap.evictionIter()(func(cacheKey, cacheItem) bool { return true })
	})
}

func TestTinyLFUEvictionIterSkipsStaging(t *testing.T) {
	conf := tlTestConf(t)
	tl := newTinyLFUEviction(&conf)
	for i := uint64(1); i <= 4; i++ {
		tl.add(cacheKey{id: i, indx: 0, size: 9}, cacheItem{size: -9, atime: 1})
	}
	tl.add(cacheKey{id: 100, indx: 0, size: 10}, cacheItem{size: 1024, atime: 1})
	var yielded int
	tl.evictionIter()(func(k cacheKey, item cacheItem) bool {
		require.Greater(t, item.size, int32(0))
		yielded++
		return true
	})
	require.Equal(t, 1, yielded)
	require.Equal(t, 4, tl.len(), "staging entries must remain")
}

func TestTinyLFUAdmissionProtectsHotSet(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(6 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 6)

	// Build a hot set: each key is logically read many times before caching.
	var hotKeys []string
	for id := uint64(1); id <= 6; id++ {
		key := tlKey(id, 0, tlBlockSize)
		hotKeys = append(hotKeys, key)
		for i := 0; i < 20; i++ {
			c.touch(key)
		}
		c.cache(key, tlPage(tlBlockSize), true, false)
	}
	waitCacheIdle(t, c)

	// A one-off sequential scan of cold blocks, each observed exactly once,
	// mirroring a miss followed by a successful load.
	for id := uint64(100); id < 120; id++ {
		key := tlKey(id, 0, tlBlockSize)
		c.touch(key)
		c.cache(key, tlPage(tlBlockSize), true, false)
	}
	waitCacheIdle(t, c)

	for _, key := range hotKeys {
		r, err := c.load(key)
		require.NoError(t, err, "hot block %s must stay cached", key)
		require.NoError(t, r.Close())
	}
	require.Greater(t, toFloat64(c.m.cacheAdmissionReject), 0.0)

	cnt, used := c.stats()
	require.LessOrEqual(t, cnt, int64(6))
	require.LessOrEqual(t, used, capacity)

	// No physical file was created for rejected blocks.
	files := countRawFiles(t, c)
	require.Equal(t, 6, files)
}

func TestTinyLFUAdmissionOversizedBypass(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(4 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 4)

	// Even a very hot block is bypassed when it can never fit.
	key := tlKey(999, 0, 64*tlBlockSize)
	for i := 0; i < 30; i++ {
		c.touch(key)
	}
	c.cache(key, tlPage(64*tlBlockSize), true, false)
	waitCacheIdle(t, c)

	require.Equal(t, 1.0, toFloat64(c.m.cacheAdmissionReject))
	cnt, used := c.stats()
	require.Equal(t, int64(0), cnt)
	require.Equal(t, int64(0), used)
	require.Equal(t, 0, countRawFiles(t, c))
}

func TestTinyLFUAdmissionMultiVictimBytesAndInode(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(4 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 4)

	// Four cold read blocks fill the main partition.
	for id := uint64(1); id <= 4; id++ {
		key := tlKey(id, 0, tlBlockSize)
		c.add(key, tlBlockSize, uint32(time.Now().Unix()))
	}
	require.Equal(t, int64(4*(tlBlockSize+blockFixedOverhead)), c.used)

	// A 6000-byte candidate costs 10096 bytes: two victims are required on
	// the byte constraint and one on the inode constraint.
	cand := tlKey(50, 0, 6000)
	for i := 0; i < 10; i++ {
		c.touch(cand)
	}
	c.cache(cand, tlPage(6000), true, false)
	waitCacheIdle(t, c)

	tl := c.keys.(*tinylfuEviction)
	require.Equal(t, 3, tl.readLen(), "4 - 2 victims + 1 candidate")
	require.NotNil(t, tl.get(c.getCacheKey(cand)))
	require.LessOrEqual(t, c.used, capacity)
	require.LessOrEqual(t, int64(tl.len()), c.maxItems)
	require.Equal(t, 2.0, toFloat64(c.m.cacheEvicts))
	require.FileExists(t, c.cachePath(c.getPathFromKey(c.getCacheKey(cand))))

	// Inode-only pressure: tight maxItems, ample bytes.
	c.maxItems = 3
	c.Lock()
	need := int64(tl.readLen()) + 1 - c.maxItems
	c.Unlock()
	require.Equal(t, int64(1), need)
	cand2 := tlKey(51, 0, 512)
	for i := 0; i < 10; i++ {
		c.touch(cand2)
	}
	c.cache(cand2, tlPage(512), true, false)
	waitCacheIdle(t, c)
	require.LessOrEqual(t, int64(tl.readLen()), c.maxItems)
}

func TestTinyLFUWindowProbationAdmitsColdKey(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(2 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 2)
	tl := c.keys.(*tinylfuEviction)

	// Two hot victims occupy the whole partition.
	for id := uint64(1); id <= 2; id++ {
		key := tlKey(id, 0, tlBlockSize)
		c.add(key, tlBlockSize, 1)
		for i := 0; i < 20; i++ {
			c.touch(key)
		}
	}
	cold := c.getCacheKey(tlKey(80, 0, tlBlockSize))
	c.Lock()
	c.admit.pushWindow(cold, tl.present) // window probation grants a free pass
	c.Unlock()

	victims, bypass := c.admit.decide(c, cold, tlBlockSize)
	require.False(t, bypass)
	require.Len(t, victims, 1)
}

func TestTinyLFUStagingBlocksAreNeverEvicted(t *testing.T) {
	conf := tlTestConf(t)
	conf.Writeback = true
	capacity := int64(3 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 3)
	// Staging entries share the item budget but are never evictable, so give
	// enough inode room for the five staged blocks plus read blocks.
	c.maxItems = 10

	var stageKeys []string
	for id := uint64(1); id <= 5; id++ {
		key := tlKey(id, 0, 9)
		stageKeys = append(stageKeys, key)
		_, err := c.stage(key, []byte("stage-data"), 0)
		require.NoError(t, err)
	}
	tl := c.keys.(*tinylfuEviction)
	require.Equal(t, 5, tl.len())
	require.Equal(t, 0, tl.readLen())

	// Simulate a full read partition while only staging entries exist in the
	// index: no read victim can be assembled, so admission must be bypassed
	// instead of touching staging blocks.
	c.Lock()
	c.used = c.capacity
	c.Unlock()
	for id := uint64(100); id < 105; id++ {
		key := tlKey(id, 0, tlBlockSize)
		c.touch(key)
		c.cache(key, tlPage(tlBlockSize), true, false)
	}
	require.Equal(t, 5.0, toFloat64(c.m.cacheAdmissionReject))
	require.Equal(t, 5, tl.len(), "all staging entries remain")
	require.Equal(t, 0, tl.readLen())
	for _, key := range stageKeys {
		require.FileExists(t, c.stagePath(key))
		require.FileExists(t, c.cachePath(key), "staging hardlink must remain")
	}

	// Once uploaded, the block becomes a normal read block and may be evicted.
	c.Lock()
	c.used = 0 // the uploaded read block now owns the accounting again
	c.Unlock()
	uploaded := stageKeys[0]
	c.uploaded(uploaded, 9)
	require.Equal(t, 1, tl.readLen())
	cand := tlKey(200, 0, 512)
	for i := 0; i < 10; i++ {
		c.touch(cand)
	}
	// Fill the read partition first so the candidate needs a victim.
	c.Lock()
	c.used = c.capacity
	c.Unlock()
	c.cache(cand, tlPage(512), true, false)
	waitCacheIdle(t, c)
	// The four pure staging copies are still untouched.
	for _, key := range stageKeys[1:] {
		require.FileExists(t, c.stagePath(key))
	}
}

func TestTinyLFUScanRebuildDoesNotInflateHeat(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(8 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 8)

	hot := tlKey(1, 0, tlBlockSize)
	cold := tlKey(2, 0, tlBlockSize)
	c.cache(hot, tlPage(tlBlockSize), true, false)
	c.cache(cold, tlPage(tlBlockSize), true, false)
	waitCacheIdle(t, c)
	for i := 0; i < 30; i++ {
		c.touch(hot)
	}
	hotBefore := c.admit.frequency(c.getCacheKey(hot))
	coldBefore := c.admit.frequency(c.getCacheKey(cold))
	require.Greater(t, hotBefore, coldBefore)

	// Rebuilding the index from disk must not record any observation.
	c.scanCached(false)
	require.Equal(t, hotBefore, c.admit.frequency(c.getCacheKey(hot)))
	require.Equal(t, coldBefore, c.admit.frequency(c.getCacheKey(cold)))

	assertIndexFileConsistency(t, c)
}

func TestTinyLFUGenerationIsolationAndFallback(t *testing.T) {
	confA := tlTestConf(t)
	confB := tlTestConf(t)
	capacity := int64(4 * (tlBlockSize + blockFixedOverhead))
	confA.CacheSize = uint64(capacity)
	confB.CacheSize = uint64(capacity)
	a := newTLDiskCache(t, &confA, capacity, 4)
	b := newTLDiskCache(t, &confB, capacity, 4)
	require.NotEqual(t, a.admit.instance, b.admit.instance)

	key := tlKey(42, 0, tlBlockSize)
	for i := 0; i < 20; i++ {
		a.touch(key)
	}
	require.Greater(t, a.admit.frequency(a.getCacheKey(key)), uint8(0))
	// The same key in another volume/instance has no inherited heat.
	require.Equal(t, uint8(0), b.admit.frequency(b.getCacheKey(key)))

	// Memory fallback starts a fresh store with no frequency state.
	memConf := defaultConf
	memConf.CacheDir = "memory"
	memConf.CacheEviction = EvictionTinyLFU
	memConf.SelfCheck("uuid")
	require.Equal(t, Eviction2Random, memConf.CacheEviction)
	mem := newMemStore(&memConf, newTLMetrics())
	mem.touch(key) // must be a no-op
	require.IsType(t, &memcache{}, mem)
}

func TestTinyLFUTwoPhaseGenerationGuard(t *testing.T) {
	conf := tlTestConf(t)
	tl := newTinyLFUEviction(&conf)
	policy := tl.policy
	k := cacheKey{id: 5, indx: 0, size: 10}
	full := tl.keyString(k)

	tl.add(k, cacheItem{size: tlBlockSize, atime: 1})
	require.NotNil(t, tl.remove(k, false))
	gen := policy.tombstones[k]
	require.NotZero(t, gen)
	require.True(t, policy.readyForDeletion(k, full, tl, map[string]*Page{}))

	// A concurrent replacement indexes a new generation: deletion is blocked.
	tl.add(k, cacheItem{size: tlBlockSize, atime: 2})
	require.False(t, policy.readyForDeletion(k, full, tl, map[string]*Page{}))

	// An in-flight page write also blocks deletion of the new generation.
	require.NotNil(t, tl.remove(k, false))
	pages := map[string]*Page{full: NewPage(make([]byte, 1))}
	require.False(t, policy.readyForDeletion(k, full, tl, pages))
	require.True(t, policy.readyForDeletion(k, full, tl, map[string]*Page{}))
	policy.clearTombstone(k)
	require.False(t, policy.readyForDeletion(k, full, tl, map[string]*Page{}))
}

func TestTinyLFUSafeRemoveSkipsReusedGeneration(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(4 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 4)
	k := c.getCacheKey(tlKey(9, 0, tlBlockSize))
	path := c.cachePath(c.getPathFromKey(k))
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
	require.NoError(t, os.WriteFile(path, []byte("v1"), 0600))
	c.add(tlKey(9, 0, tlBlockSize), tlBlockSize, 1)

	// Simulate an eviction selecting the key, then a concurrent re-cache that
	// starts the next generation before file deletion.
	c.Lock()
	for key, item := range c.keys.evictionIter() {
		require.Equal(t, k, key)
		_ = item
	}
	require.NotEmpty(t, c.admit.tombstones)
	c.keys.add(k, cacheItem{size: tlBlockSize, atime: 2}) // new generation
	c.Unlock()

	c.safeRemoveCacheFile(k)
	require.FileExists(t, path, "new generation file must not be deleted")

	// After the new generation is evicted in turn, deletion proceeds.
	c.Lock()
	c.keys.remove(k, false)
	c.Unlock()
	c.safeRemoveCacheFile(k)
	_, err := os.Stat(path)
	require.True(t, os.IsNotExist(err))
}

func TestTinyLFUConcurrentLoadCacheRemove(t *testing.T) {
	conf := tlTestConf(t)
	capacity := int64(8 * (tlBlockSize + blockFixedOverhead))
	conf.CacheSize = uint64(capacity)
	c := newTLDiskCache(t, &conf, capacity, 8)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			i := 0
			for {
				select {
				case <-stop:
					return
				default:
				}
				id := uint64(g*50) + uint64(i%40)
				key := tlKey(id, uint32(i%2), tlBlockSize)
				switch i % 4 {
				case 0:
					c.touch(key)
					c.cache(key, tlPage(tlBlockSize), true, false)
				case 1:
					if r, err := c.load(key); err == nil {
						_ = r.Close()
					}
				case 2:
					c.touch(key)
				case 3:
					c.remove(key, false)
				}
				i++
			}
		}(uint64(g))
	}
	time.Sleep(2 * time.Second)
	close(stop)
	wg.Wait()
	waitCacheIdle(t, c)

	// Capacity and inode limits are respected.
	cnt, used := c.stats()
	require.LessOrEqual(t, used, capacity)
	require.LessOrEqual(t, cnt, c.maxItems)
	assertIndexFileConsistency(t, c)
}

func assertIndexFileConsistency(t *testing.T, c *diskCache) {
	c.scanCached(false) // reconcile any orphan file with the index
	c.Lock()
	tl := c.keys.(*tinylfuEviction)
	for k, it := range tl.keys {
		if it.size > 0 {
			require.FileExists(t, c.cachePath(c.getPathFromKey(k)))
		}
	}
	c.Unlock()

	rawRoot := filepath.Join(c.dir, cacheDir)
	require.NoError(t, filepath.WalkDir(rawRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || strings.HasSuffix(path, ".tmp") {
			return nil
		}
		rel, err := filepath.Rel(filepath.Join(c.dir, cacheDir), path)
		require.NoError(t, err)
		rel = strings.ReplaceAll(rel, string(filepath.Separator), "/")
		k := c.getCacheKey(rel)
		c.Lock()
		_, ok := tl.keys[k]
		c.Unlock()
		require.True(t, ok, "file %s has no index entry", rel)
		return nil
	}))
}

func countRawFiles(t *testing.T, c *diskCache) int {
	n := 0
	root := filepath.Join(c.dir, cacheDir)
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && !strings.HasSuffix(path, ".tmp") {
			n++
		}
		return nil
	})
	return n
}

// TestTinyLFUEndToEndTrace runs the same deterministic hot-set + sequential
// scan + mixed-size access trace against tinylfu and 2-random and checks data
// correctness, cache limits and hot-set hit-rate improvement.
func TestTinyLFUEndToEndTrace(t *testing.T) {
	storage, _ := object.CreateStorage("mem", "", "", "", "")
	const (
		blockSize = 64 << 10
		hotNum    = 6
		fullNum   = 40 // large cold scan relative to the 8-block cache
	)
	capacity := int64(8 * (blockSize + blockFixedOverhead))

	writeObjects := func(store ChunkStore) {
		for id := uint64(1); id <= fullNum; id++ {
			writeTraceBlock(t, store, id, blockSize)
		}
	}

	runTrace := func(eviction string) (hotRetained, hotHits int, maxUsed, maxCnt int64) {
		conf := defaultConf
		conf.CacheDir = t.TempDir()
		conf.CacheEviction = eviction
		conf.CacheScanInterval = -1
		conf.FreeSpace = 0.00001
		conf.BlockSize = blockSize
		conf.CacheSize = uint64(capacity)
		conf.CacheItems = 8
		// Full-block reads enter the read cache, while the write path is kept
		// from caching blocks so the trace exercises admission only.
		conf.CacheFullBlock = true
		conf.CacheLargeWrite = false
		store := NewCachedStore(storage, conf, nil).(*cachedStore)
		for _, s := range store.bcache.(*cacheManager).stores {
			s.maxItems = int64(conf.CacheItems)
		}
		writeObjects(store)

		readFull := func(id uint64) {
			r := store.NewReader(id, blockSize)
			p := NewOffPage(blockSize)
			n, err := r.ReadAt(context.Background(), p, 0)
			require.Equal(t, blockSize, n)
			require.NoError(t, err)
			verifyTraceData(t, id, p.Data, blockSize)
			p.Release()
		}
		// Hot set: five repeated rounds.
		for round := 0; round < 5; round++ {
			for id := uint64(1); id <= hotNum; id++ {
				readFull(id)
			}
		}
		// One-off sequential scan.
		for id := uint64(hotNum + 1); id <= fullNum; id++ {
			readFull(id)
		}
		// Mixed-size accesses: partial reads of varied lengths on the hot set.
		partialSizes := []int{4 << 10, 16 << 10, 20 << 10}
		for idx, id := uint64(0), uint64(1); id <= hotNum; id, idx = id+1, idx+1 {
			n := partialSizes[int(idx)%len(partialSizes)]
			r := store.NewReader(id, blockSize)
			p := NewOffPage(n)
			got, err := r.ReadAt(context.Background(), p, 0)
			require.Equal(t, n, got)
			require.NoError(t, err)
			verifyTraceData(t, id, p.Data, n)
			p.Release()
		}

		require.Eventually(t, func() bool {
			cnt, _ := store.bcache.stats()
			return cnt <= conf.CacheItems
		}, 5*time.Second, 10*time.Millisecond)
		cnt, used := store.bcache.stats()
		maxUsed, maxCnt = used, cnt
		require.LessOrEqual(t, maxUsed, capacity)
		require.LessOrEqual(t, maxCnt, int64(conf.CacheItems))

		// Final verification round: count retained hot blocks and hits.
		for id := uint64(1); id <= hotNum; id++ {
			key := fmt.Sprintf("chunks/0/0/%d_0_%d", id, blockSize)
			if _, ok := store.bcache.exist(key); ok {
				hotRetained++
			}
			before := toFloat64(store.cacheHits)
			readFull(id)
			if toFloat64(store.cacheHits) > before {
				hotHits++
			}
		}
		return
	}

	// TinyLFU is deterministic here (the index is smaller than one eviction
	// sample), while 2-random depends on map iteration order: average several
	// runs to make the comparison statistically meaningful.
	tlRetained, tlHits, tlUsed, tlCnt := runTrace(EvictionTinyLFU)
	const randomRuns = 5
	var rndRetainedSum, rndHitsSum int
	for i := 0; i < randomRuns; i++ {
		retained, hits, _, _ := runTrace(Eviction2Random)
		rndRetainedSum += retained
		rndHitsSum += hits
	}
	rndRetainedAvg := float64(rndRetainedSum) / randomRuns
	rndHitsAvg := float64(rndHitsSum) / randomRuns

	require.Equal(t, hotNum, tlRetained, "tinylfu must deterministically keep the whole hot set")
	require.Greater(t, float64(tlRetained), rndRetainedAvg, "tinylfu retains more hot blocks than 2-random on average")
	require.Greater(t, float64(tlHits), rndHitsAvg, "tinylfu serves more hot-set hits than 2-random on average")
	require.Less(t, rndRetainedAvg, float64(hotNum), "the scan must evict hot blocks under 2-random")
	require.LessOrEqual(t, tlUsed, capacity)
	require.LessOrEqual(t, tlCnt, int64(8))
}

func writeTraceBlock(t *testing.T, store ChunkStore, id uint64, size int) {
	w := store.NewWriter(id, 0)
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(id) ^ byte(i)
	}
	n, err := w.WriteAt(data, 0)
	require.Equal(t, size, n)
	require.NoError(t, err)
	require.NoError(t, w.Finish(size))
}

func verifyTraceData(t *testing.T, id uint64, data []byte, size int) {
	require.Equal(t, size, len(data))
	for i, b := range data {
		require.Equal(t, byte(id)^byte(i), b, "data mismatch for block %d at %d", id, i)
	}
}
