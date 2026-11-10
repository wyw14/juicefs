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
	"encoding/binary"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/twmb/murmur3"
)

// EvictionTinyLFU is a byte-aware TinyLFU read-cache admission policy. It keeps
// a bounded frequency sketch of real logical accesses and only lets a newly
// loaded block enter the main cache partition when its access frequency beats
// the blocks that must be evicted to make room for it.
const EvictionTinyLFU = "tinylfu"

const (
	sketchDepth      = 4
	sketchBits       = 4
	sketchCounterMax = 1<<sketchBits - 1 // 4-bit counters saturate at 15
	minSketchWidth   = 64
	// maxSketchWidth bounds the sketch memory: sketchDepth rows of width/16
	// 64-bit words, i.e. at most 2 MiB regardless of the number of distinct keys.
	maxSketchWidth = 1 << 20
	// decayEvery halves every counter after decayEvery*width observations.
	decayEvery = 10

	minWindowItems   = 8
	maxWindowItems   = 1024
	victimSampleSize = 64
	// blockFixedOverhead mirrors the per-file overhead accounted in diskCache.
	blockFixedOverhead = 4096
)

// policyInstanceSeq uniquely identifies every admission instance. A new
// instance (new disk cache, device replacement or memory fallback) starts with
// an empty sketch and never inherits heat of another volume/instance.
var policyInstanceSeq atomic.Uint64

// frequencySketch is a fixed-memory Count-Min sketch with packed 4-bit
// counters. Counters saturate at sketchCounterMax and are periodically halved,
// so neither saturation nor decay can overflow.
type frequencySketch struct {
	mu        sync.Mutex
	table     [sketchDepth][]uint64
	width     uint64 // power of two
	mask      uint64
	additions uint64
	decays    uint64
}

func newFrequencySketch(width int) *frequencySketch {
	w := uint64(minSketchWidth)
	for w < uint64(width) && w < maxSketchWidth {
		w <<= 1
	}
	s := &frequencySketch{width: w, mask: w - 1}
	words := int(w) / 16
	for d := range s.table {
		s.table[d] = make([]uint64, words)
	}
	return s
}

// memoryBytes reports the fixed allocation of the sketch tables.
func (s *frequencySketch) memoryBytes() int {
	return sketchDepth * len(s.table[0]) * 8
}

func keyHashes(k cacheKey) (uint64, uint64) {
	var b [16]byte
	binary.BigEndian.PutUint64(b[:8], k.id)
	binary.BigEndian.PutUint32(b[8:12], k.indx)
	binary.BigEndian.PutUint32(b[12:16], k.size)
	return murmur3.Sum128(b[:])
}

// add records one observation. It only touches sketch memory and never takes
// the disk cache lock or performs any I/O.
func (s *frequencySketch) add(k cacheKey) {
	h1, h2 := keyHashes(k)
	s.mu.Lock()
	for d := 0; d < sketchDepth; d++ {
		idx := (h1 + uint64(d)*h2) & s.mask
		word, pos := idx>>4, (idx&15)*sketchBits
		if (s.table[d][word]>>pos)&sketchCounterMax < sketchCounterMax {
			s.table[d][word] += 1 << pos
		}
	}
	s.additions++
	if s.additions >= s.width*decayEvery {
		s.decayLocked()
	}
	s.mu.Unlock()
}

// decayLocked halves every counter. Shifting first and masking the low three
// bits of each nibble drops the carried bit, so each 4-bit counter becomes
// floor(v/2) and stays inside [0, 15].
func (s *frequencySketch) decayLocked() {
	const lowThreeBits = 0x7777777777777777
	for d := range s.table {
		for i, w := range s.table[d] {
			s.table[d][i] = (w >> 1) & lowThreeBits
		}
	}
	s.additions = 0
	s.decays++
}

func (s *frequencySketch) estimate(k cacheKey) uint8 {
	h1, h2 := keyHashes(k)
	s.mu.Lock()
	defer s.mu.Unlock()
	hot := uint8(sketchCounterMax)
	for d := 0; d < sketchDepth; d++ {
		idx := (h1 + uint64(d)*h2) & s.mask
		v := uint8((s.table[d][idx>>4] >> ((idx & 15) * sketchBits)) & sketchCounterMax)
		if v < hot {
			hot = v
		}
	}
	return hot
}

// victimCandidate is a read block selected as a potential eviction victim.
type victimCandidate struct {
	key   cacheKey
	full  string
	gen   uint64
	size  int32
	atime uint32
	freq  uint8
}

// victimKey is a victim whose index entry has already been removed; the
// physical file can only be deleted after validating its generation.
type victimKey struct {
	key  cacheKey
	full string
	gen  uint64
	size int32
}

// tinyLFUPolicy holds the frequency history and the small window partition.
// All frequency state is in-memory and bounded: the sketch has a fixed size
// and the window holds at most maxWindowItems entries.
type tinyLFUPolicy struct {
	instance uint64
	sketch   *frequencySketch

	// window is a FIFO probation partition; all accesses to it happen while
	// the disk cache lock is held.
	window    []cacheKey
	windowSet map[cacheKey]struct{}
	windowCap int

	// itemGen assigns a generation to every indexed entry; tombstones records
	// the generation of entries whose file deletion is still pending.
	itemGen    uint64
	tombstones map[cacheKey]uint64
}

func newTinyLFUPolicy(cacheSize uint64, blockSize int) *tinyLFUPolicy {
	if blockSize <= 0 {
		blockSize = 1 << 20
	}
	expectedBlocks := int(cacheSize / uint64(blockSize))
	width := expectedBlocks * 4
	if width < minSketchWidth {
		width = minSketchWidth
	}
	if width > maxSketchWidth {
		width = maxSketchWidth
	}
	windowCap := expectedBlocks / 100
	if windowCap < minWindowItems {
		windowCap = minWindowItems
	}
	if windowCap > maxWindowItems {
		windowCap = maxWindowItems
	}
	return &tinyLFUPolicy{
		instance:   policyInstanceSeq.Add(1),
		sketch:     newFrequencySketch(width),
		window:     make([]cacheKey, 0, windowCap),
		windowSet:  make(map[cacheKey]struct{}, windowCap),
		windowCap:  windowCap,
		tombstones: make(map[cacheKey]uint64),
	}
}

// observe records a real logical access. It never takes the disk cache lock.
func (p *tinyLFUPolicy) observe(k cacheKey) {
	p.sketch.add(k)
}

func (p *tinyLFUPolicy) frequency(k cacheKey) uint8 {
	return p.sketch.estimate(k)
}

// purgeWindow drops window entries that no longer exist in the key index.
// Caller holds the disk cache lock.
func (p *tinyLFUPolicy) purgeWindow(present func(cacheKey) bool) {
	kept := p.window[:0]
	for _, k := range p.window {
		if present(k) {
			kept = append(kept, k)
		} else {
			delete(p.windowSet, k)
		}
	}
	p.window = kept
}

// pushWindow admits k into the bounded window partition, evicting its oldest
// entry when full. Caller holds the disk cache lock.
func (p *tinyLFUPolicy) pushWindow(k cacheKey, present func(cacheKey) bool) {
	if _, ok := p.windowSet[k]; ok {
		return
	}
	p.purgeWindow(present)
	if len(p.window) >= p.windowCap {
		oldest := p.window[0]
		p.window = p.window[1:]
		delete(p.windowSet, oldest)
	}
	p.window = append(p.window, k)
	p.windowSet[k] = struct{}{}
}

func (p *tinyLFUPolicy) inWindow(k cacheKey, present func(cacheKey) bool) bool {
	// The candidate itself is not indexed yet, so it must survive the purge.
	kept := p.window[:0]
	for _, wk := range p.window {
		if wk == k || present(wk) {
			kept = append(kept, wk)
		} else {
			delete(p.windowSet, wk)
		}
	}
	p.window = kept
	_, ok := p.windowSet[k]
	return ok
}

// decide applies the byte-aware admission rule for a newly loaded read block.
// Caller holds the disk cache lock; this function does no I/O. When admission
// is granted and evictions are required, the victim entries are removed from
// the index by commitVictims. A bypass decision never touches staging blocks.
func (p *tinyLFUPolicy) decide(c *diskCache, k cacheKey, blockSize int32) (victims []victimKey, bypass bool) {
	cost := int64(blockSize) + blockFixedOverhead
	// A candidate that can never fit, even into an empty main partition, is
	// bypassed instead of overflowing the capacity.
	if cost > c.capacity {
		return nil, true
	}
	idx := c.keys.(*tinylfuEviction)

	var needBytes, needItems int64
	if c.used+cost > c.capacity {
		needBytes = c.used + cost - c.capacity
	}
	if c.maxItems != 0 && int64(idx.readLen())+1 > c.maxItems {
		needItems = int64(idx.readLen()) + 1 - c.maxItems
	}
	if needBytes == 0 && needItems == 0 {
		p.pushWindow(k, idx.present)
		return nil, false
	}

	candidates, ok := idx.peekVictims(needBytes, needItems, k, p)
	if !ok {
		// Only staging blocks remain; the unique un-uploaded copies are never
		// evicted for a read block.
		return nil, true
	}
	// Keys serving their window probation period are protected from the
	// frequency gate; every other candidate must be hotter than the hottest
	// block of the whole victim set required to make room.
	if !p.inWindow(k, idx.present) {
		candidateFreq := p.frequency(k)
		var hottestVictim uint8
		for _, cand := range candidates {
			if cand.freq > hottestVictim {
				hottestVictim = cand.freq
			}
		}
		if candidateFreq <= hottestVictim {
			return nil, true
		}
	}
	victims = idx.commitVictims(candidates)
	p.pushWindow(k, idx.present)
	return victims, false
}

// readyForDeletion revalidates a victim under the disk cache lock before its
// file is removed: the tombstone must still match and no new generation may
// have been indexed or be in flight in the pending pages map.
func (p *tinyLFUPolicy) readyForDeletion(k cacheKey, fullKey string, idx *tinylfuEviction, pages map[string]*Page) bool {
	if _, ok := p.tombstones[k]; !ok {
		return false
	}
	if _, ok := idx.keys[k]; ok {
		return false // a new generation was indexed concurrently
	}
	if _, ok := pages[fullKey]; ok {
		return false // a replacement file is being written
	}
	return true
}

func (p *tinyLFUPolicy) clearTombstone(k cacheKey) {
	delete(p.tombstones, k)
}

// tlItem is an indexed cache entry with its insertion generation.
type tlItem struct {
	cacheItem
	gen uint64
}

// tinylfuEviction is the KeyIndex implementation backing the TinyLFU policy.
// Read blocks (size > 0) and staging blocks (size < 0) coexist in the same map,
// but staging blocks are never selected as eviction victims.
type tinylfuEviction struct {
	keys       map[cacheKey]*tlItem
	policy     *tinyLFUPolicy
	hashPrefix bool
	snapshot   bool
}

func newTinyLFUEviction(config *Config) *tinylfuEviction {
	policy := newTinyLFUPolicy(config.CacheSize, config.BlockSize)
	return &tinylfuEviction{
		keys:       make(map[cacheKey]*tlItem),
		policy:     policy,
		hashPrefix: config.HashPrefix,
	}
}

func (e *tinylfuEviction) name() string { return EvictionTinyLFU }

func (e *tinylfuEviction) add(key cacheKey, item cacheItem) {
	if it, ok := e.keys[key]; ok {
		// Re-adding (e.g. a staged block finishes uploading) starts a new
		// generation so pending deletions cannot remove the new file.
		e.policy.itemGen++
		it.cacheItem = item
		it.gen = e.policy.itemGen
		return
	}
	e.policy.itemGen++
	e.keys[key] = &tlItem{cacheItem: item, gen: e.policy.itemGen}
}

func (e *tinylfuEviction) remove(key cacheKey, staging bool) *cacheItem {
	it, ok := e.keys[key]
	if !ok {
		return nil
	}
	if it.size < 0 && !staging {
		return nil
	}
	delete(e.keys, key)
	if it.size > 0 {
		// Remember the removed generation until the physical file is deleted.
		e.policy.tombstones[key] = it.gen
	}
	cp := it.cacheItem
	return &cp
}

func (e *tinylfuEviction) get(key cacheKey) *cacheItem {
	if it, ok := e.keys[key]; ok {
		it.atime = uint32(time.Now().Unix())
		cp := it.cacheItem
		return &cp
	}
	return nil
}

func (e *tinylfuEviction) peekAtime(key cacheKey) uint32 {
	if it, ok := e.keys[key]; ok {
		return it.atime
	}
	return 0
}

func (e *tinylfuEviction) len() int { return len(e.keys) }

// readLen counts read blocks; staging blocks do not consume read capacity.
func (e *tinylfuEviction) readLen() int {
	n := 0
	for _, it := range e.keys {
		if it.size > 0 {
			n++
		}
	}
	return n
}

func (e *tinylfuEviction) present(k cacheKey) bool {
	_, ok := e.keys[k]
	return ok
}

func (e *tinylfuEviction) reset() KeyIndex {
	snap := &tinylfuEviction{keys: e.keys, policy: e.policy, hashPrefix: e.hashPrefix, snapshot: true}
	e.keys = make(map[cacheKey]*tlItem, len(e.keys))
	return snap
}

func (e *tinylfuEviction) randomIter() func(yield func(key cacheKey, item cacheItem) bool) {
	return func(yield func(key cacheKey, item cacheItem) bool) {
		for k, v := range e.keys {
			if !yield(k, v.cacheItem) {
				return
			}
		}
	}
}

// sampleRead collects up to n read blocks not present in visited. Go map
// iteration is randomized, so repeated calls sample different blocks; when the
// whole index fits in one sample (the common case in tests) it is exhaustive,
// which keeps eviction deterministic.
func (e *tinylfuEviction) sampleRead(n int, visited map[cacheKey]struct{}, exclude cacheKey) []cacheKey {
	batch := make([]cacheKey, 0, n)
	for k, it := range e.keys {
		if it.size <= 0 {
			continue // staging blocks are protected
		}
		if k == exclude {
			continue
		}
		if _, ok := visited[k]; ok {
			continue
		}
		visited[k] = struct{}{}
		batch = append(batch, k)
		if len(batch) >= n {
			break
		}
	}
	return batch
}

func (e *tinylfuEviction) sortByVictimOrder(keys []cacheKey, p *tinyLFUPolicy) {
	sort.Slice(keys, func(i, j int) bool {
		a, b := e.keys[keys[i]], e.keys[keys[j]]
		fa, fb := p.frequency(keys[i]), p.frequency(keys[j])
		if fa != fb {
			return fa < fb // coldest first
		}
		if a.atime != b.atime {
			return a.atime < b.atime // older first
		}
		if a.size != b.size {
			return a.size > b.size // prefer freeing larger blocks
		}
		if keys[i].id != keys[j].id {
			return keys[i].id < keys[j].id
		}
		return keys[i].indx < keys[j].indx
	})
}

// peekVictims selects the read blocks required to free needBytes and
// needItems, without mutating the index. Both the byte and inode constraints
// must be satisfied before a set is returned.
func (e *tinylfuEviction) peekVictims(needBytes, needItems int64, exclude cacheKey, p *tinyLFUPolicy) ([]victimCandidate, bool) {
	var picked []victimCandidate
	var freedBytes, freedItems int64
	visited := make(map[cacheKey]struct{})
	for {
		batch := e.sampleRead(victimSampleSize, visited, exclude)
		if len(batch) == 0 {
			return nil, false
		}
		e.sortByVictimOrder(batch, p)
		for _, k := range batch {
			it := e.keys[k]
			picked = append(picked, victimCandidate{
				key:   k,
				full:  e.keyString(k),
				gen:   it.gen,
				size:  it.size,
				atime: it.atime,
				freq:  p.frequency(k),
			})
			freedBytes += int64(it.size) + blockFixedOverhead
			freedItems++
			if freedBytes >= needBytes && freedItems >= needItems {
				return picked, true
			}
		}
		if len(visited) >= e.readLen() {
			return nil, false
		}
	}
}

// commitVictims removes the selected entries from the index and records their
// tombstones for the two-phase file deletion.
func (e *tinylfuEviction) commitVictims(candidates []victimCandidate) []victimKey {
	victims := make([]victimKey, 0, len(candidates))
	for _, cand := range candidates {
		it, ok := e.keys[cand.key]
		if !ok || it.size <= 0 || it.gen != cand.gen {
			continue // changed concurrently, skip it
		}
		delete(e.keys, cand.key)
		e.policy.tombstones[cand.key] = cand.gen
		victims = append(victims, victimKey{
			key:  cand.key,
			full: cand.full,
			gen:  cand.gen,
			size: cand.size,
		})
	}
	return victims
}

// evictionIter is the backstop used by generic capacity/free-space cleanup.
// It yields read blocks in frequency order and never yields staging blocks.
func (e *tinylfuEviction) evictionIter() func(yield func(key cacheKey, item cacheItem) bool) {
	return func(yield func(key cacheKey, item cacheItem) bool) {
		if e.snapshot {
			panic("eviction is not supported on a snapshot index")
		}
		visited := make(map[cacheKey]struct{})
		for {
			batch := e.sampleRead(victimSampleSize, visited, cacheKey{})
			if len(batch) == 0 {
				return
			}
			e.sortByVictimOrder(batch, e.policy)
			for _, k := range batch {
				it, ok := e.keys[k]
				if !ok || it.size <= 0 {
					continue
				}
				gen := it.gen
				item := it.cacheItem
				delete(e.keys, k)
				e.policy.tombstones[k] = gen
				if !yield(k, item) {
					return
				}
			}
		}
	}
}

// keyString reconstructs the on-disk key ("chunks/.../id_indx_size"), which is
// also how keys are stored in diskCache.pages.
func (e *tinylfuEviction) keyString(k cacheKey) string {
	if e.hashPrefix {
		return "chunks/" + hex2(k.id%256) + "/" + itoa(k.id/1000/1000) + "/" + k.String()
	}
	return "chunks/" + itoa(k.id/1000/1000) + "/" + itoa(k.id/1000) + "/" + k.String()
}

const hexDigits = "0123456789ABCDEF"

func hex2(v uint64) string {
	return string([]byte{hexDigits[(v>>4)&0xf], hexDigits[v&0xf]})
}

func itoa(v uint64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
