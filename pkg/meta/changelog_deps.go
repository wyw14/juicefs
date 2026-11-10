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

package meta

import (
	"fmt"
	"sort"
)

// ResourceClass identifies the metadata resource an edge conflicts on.
type ResourceClass uint8

const (
	ResUnknown ResourceClass = iota
	// ResInode is the inode lifecycle record: existence, parent and nlink.
	ResInode
	// ResDentry is a directory entry addressed by (parent, name).
	ResDentry
	// ResChunk is a single chunk addressed by (inode, chunk index).
	ResChunk
	// ResChunkAll stands for every chunk of an inode, used by TRUNCATE.
	ResChunkAll
	// ResAttrSet is the attribute record of an inode (mode, owner, times, ...).
	ResAttrSet
	// ResXattr is one extended attribute addressed by (inode, xattr name).
	ResXattr
	// ResTransaction orders records sharing the same (session ID, txn ID).
	ResTransaction
)

func (c ResourceClass) String() string {
	switch c {
	case ResInode:
		return "inode"
	case ResDentry:
		return "dentry"
	case ResChunk:
		return "chunk"
	case ResChunkAll:
		return "chunkall"
	case ResAttrSet:
		return "attr"
	case ResXattr:
		return "xattr"
	case ResTransaction:
		return "txn"
	default:
		return "unknown"
	}
}

// DependencyResource is a typed, fully resolved conflict key. The struct is
// comparable and can be used as a map key.
type DependencyResource struct {
	Class  ResourceClass
	Ino    Ino
	Parent Ino
	Name   string
	Index  uint32
	Xattr  string
	Sid    uint64
	TxnId  uint64
}

func (r DependencyResource) String() string {
	switch r.Class {
	case ResInode:
		return fmt.Sprintf("inode:%d", r.Ino)
	case ResDentry:
		return fmt.Sprintf("dentry:%d/%q", r.Parent, r.Name)
	case ResChunk:
		return fmt.Sprintf("chunk:%d#%d", r.Ino, r.Index)
	case ResChunkAll:
		return fmt.Sprintf("chunkall:%d", r.Ino)
	case ResAttrSet:
		return fmt.Sprintf("attr:%d", r.Ino)
	case ResXattr:
		return fmt.Sprintf("xattr:%d/%q", r.Ino, r.Xattr)
	case ResTransaction:
		return fmt.Sprintf("txn:%d#%d", r.Sid, r.TxnId)
	default:
		return "unknown"
	}
}

// Edge conflict kinds. An edge always runs from an earlier entry to a later
// entry in the original changelog order.
const (
	// ConflictWAW: both endpoints write the resource.
	ConflictWAW = "WAW"
	// ConflictWAR: the source reads and the destination writes the resource.
	ConflictWAR = "WAR"
	// ConflictRAW: the source writes and the destination reads the resource.
	ConflictRAW = "RAW"
	// ConflictTxn: the endpoints belong to the same (sid, txnId).
	ConflictTxn = "TXN"
)

// DependencyEdge is a necessary happens-before edge between two entries.
type DependencyEdge struct {
	From     int
	To       int
	Resource DependencyResource
	Conflict string
	FromVer  int64
	ToVer    int64
}

// PlanNode carries the typed footprint of one analyzed entry.
type PlanNode struct {
	Index  int
	Ver    int64
	Op     string
	Reads  []DependencyResource
	Writes []DependencyResource
}

// DependencyBatch is a topological level: its entries conflict on no resource
// and are therefore suggested as parallelizable.
type DependencyBatch struct {
	Indices []int
}

// DependencyBarrier separates the plan into independently ordered segments.
// Every entry with index below Index must finish before any entry at or above
// Index starts.
type DependencyBarrier struct {
	Index  int
	Reason string
}

// DependencyPlan is an offline, advisory parallelism plan. It never executes
// replay and never touches the destination metadata. Membership in the same
// batch is only a parallelism suggestion: Replayer remains sequential, and no
// backend cross-entry transaction atomicity is implied.
type DependencyPlan struct {
	Nodes    []PlanNode
	Edges    []DependencyEdge
	Batches  []DependencyBatch
	Barriers []DependencyBarrier
	// MaxWindow bounds the number of entries analyzed without a forced barrier.
	MaxWindow int
}

// AnalyzeOption configures AnalyzeChangelogDependencies.
type AnalyzeOption func(*analyzeConfig)

type analyzeConfig struct {
	maxWindow int
}

// WithMaxWindow limits how many entries are held in the dependency window.
// When the window fills, a barrier is emitted and the indexes are reset, so
// analysis memory stays O(window) instead of growing with the whole stream.
func WithMaxWindow(n int) AnalyzeOption {
	return func(c *analyzeConfig) { c.maxWindow = n }
}

// DefaultAnalyzeMaxWindow is used when WithMaxWindow is not supplied.
const DefaultAnalyzeMaxWindow = 4096

// AnalyzeChangelogDependencies builds a causal batch plan over a stream of
// already parsed, version-ordered ChangeEntry values. It is purely offline:
// entries are not replayed and no metadata client is involved.
//
// CREATE, LINK, UNLINK, MOVE, SETATTR, WRITE, TRUNCATE and SETXATTR have typed
// read/write footprints on inode, directory entry, chunk and attribute-set
// resources. Every other operation is a conservative barrier, including
// variadic operations whose arguments cannot be fully explained offline and
// operations with global quota, ACL or session side effects.
//
// Duplicate versions, out-of-order versions and version gaps are hard errors.
func AnalyzeChangelogDependencies(entries []*ChangeEntry, opts ...AnalyzeOption) (*DependencyPlan, error) {
	cfg := analyzeConfig{maxWindow: DefaultAnalyzeMaxWindow}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.maxWindow < 1 {
		return nil, fmt.Errorf("max window must be positive, got %d", cfg.maxWindow)
	}
	plan := &DependencyPlan{MaxWindow: cfg.maxWindow}
	a := &depAnalyzer{cfg: cfg, plan: plan, entries: entries}
	a.reset()

	var prevVer int64
	for i, e := range entries {
		if e == nil {
			return nil, fmt.Errorf("entry %d is nil", i)
		}
		if e.Ver <= 0 {
			return nil, fmt.Errorf("entry %d has non-positive version %d", i, e.Ver)
		}
		if i > 0 {
			switch {
			case e.Ver == prevVer:
				return nil, fmt.Errorf("duplicate changelog version %d at entries %d and %d", e.Ver, i-1, i)
			case e.Ver < prevVer:
				return nil, fmt.Errorf("changelog version %d at entry %d is older than previous version %d", e.Ver, i, prevVer)
			case e.Ver != prevVer+1:
				return nil, fmt.Errorf("changelog version gap: entry %d has version %d, expected %d", i, e.Ver, prevVer+1)
			}
		}
		prevVer = e.Ver
		if err := a.ingest(i, e); err != nil {
			return nil, err
		}
	}
	if err := a.flush(); err != nil {
		return nil, err
	}
	sort.Slice(plan.Edges, func(i, j int) bool {
		a, b := plan.Edges[i], plan.Edges[j]
		if a.From != b.From {
			return a.From < b.From
		}
		if a.To != b.To {
			return a.To < b.To
		}
		return a.Resource.String() < b.Resource.String()
	})
	return plan, nil
}

type txnKey struct {
	sid   uint64
	txnId uint64
}

type depAnalyzer struct {
	cfg      analyzeConfig
	plan     *DependencyPlan
	entries  []*ChangeEntry
	segNodes []int
	segEdges []DependencyEdge
	// writer holds the last writer of each exact resource.
	writer map[DependencyResource]int
	// readers holds readers since the last writer of each exact resource.
	readers map[DependencyResource][]int
	// chunkAll records the last TRUNCATE per inode.
	chunkAll map[Ino]int
	// chunkSet lists precise chunk indexes observed since the last TRUNCATE.
	chunkSet map[Ino]map[uint32]struct{}
	txnLast  map[txnKey]int
	edgeSeen map[string]struct{}
}

func (a *depAnalyzer) reset() {
	a.segNodes = nil
	a.segEdges = nil
	a.writer = make(map[DependencyResource]int)
	a.readers = make(map[DependencyResource][]int)
	a.chunkAll = make(map[Ino]int)
	a.chunkSet = make(map[Ino]map[uint32]struct{})
	a.txnLast = make(map[txnKey]int)
	a.edgeSeen = make(map[string]struct{})
}

func (a *depAnalyzer) ingest(i int, e *ChangeEntry) error {
	reads, writes, barrier, err := entryFootprint(e)
	if err != nil {
		return err
	}
	if barrier {
		return a.barrier(i, barrierReason(e))
	}
	if len(a.segNodes) >= a.cfg.maxWindow {
		if err := a.cut(i, fmt.Sprintf("max analysis window %d reached", a.cfg.maxWindow)); err != nil {
			return err
		}
	}

	key := txnKey{e.Sid, e.TxnId}
	if last, ok := a.txnLast[key]; ok {
		// Records of one (sid, txnId) keep their original order. The txnId
		// counter is per session, so the session ID is part of the key: the
		// same txnId in different sessions must never be merged.
		a.addEdge(last, i, DependencyResource{Class: ResTransaction, Sid: e.Sid, TxnId: e.TxnId}, ConflictTxn)
	}
	a.txnLast[key] = i

	for _, r := range reads {
		a.observeRead(r, i)
	}
	for _, r := range writes {
		a.observeWrite(r, i)
	}
	a.plan.Nodes = append(a.plan.Nodes, PlanNode{
		Index: i, Ver: e.Ver, Op: e.Op, Reads: reads, Writes: writes,
	})
	a.segNodes = append(a.segNodes, i)
	return nil
}

func barrierReason(e *ChangeEntry) string {
	switch e.Op {
	case OpUnlinkBatch, OpCloneBatch:
		return fmt.Sprintf("variadic operation %s cannot be explained offline", e.Op)
	default:
		return fmt.Sprintf("operation %s is outside the typed footprint set "+
			"(CREATE, LINK, UNLINK, MOVE, SETATTR, WRITE, TRUNCATE, SETXATTR) or has global side effects", e.Op)
	}
}

func (a *depAnalyzer) barrier(i int, reason string) error {
	if len(a.segNodes) > 0 {
		return a.cut(i, reason)
	}
	a.plan.Barriers = append(a.plan.Barriers, DependencyBarrier{Index: i, Reason: reason})
	a.reset()
	return nil
}

// cut finalizes the current segment, records a barrier and drops every index.
// The barrier total-orders the segments, so dependencies crossing the cut are
// preserved without retaining any per-resource state.
func (a *depAnalyzer) cut(i int, reason string) error {
	if err := a.groupSegment(); err != nil {
		return err
	}
	a.plan.Barriers = append(a.plan.Barriers, DependencyBarrier{Index: i, Reason: reason})
	a.reset()
	return nil
}

func (a *depAnalyzer) flush() error {
	if len(a.segNodes) == 0 {
		return nil
	}
	return a.groupSegment()
}

// groupSegment turns the segment edges into stable topological batches.
func (a *depAnalyzer) groupSegment() error {
	indeg := make(map[int]int, len(a.segNodes))
	adj := make(map[int][]int)
	for _, n := range a.segNodes {
		indeg[n] = 0
	}
	for _, ed := range a.segEdges {
		adj[ed.From] = append(adj[ed.From], ed.To)
		indeg[ed.To]++
	}
	remaining := len(a.segNodes)
	done := make(map[int]bool, len(a.segNodes))
	for remaining > 0 {
		var ready []int
		// segNodes is in stream order, which gives stable, deterministic batches.
		for _, n := range a.segNodes {
			if !done[n] && indeg[n] == 0 {
				ready = append(ready, n)
			}
		}
		if len(ready) == 0 {
			return fmt.Errorf("dependency cycle detected among %d entries", remaining)
		}
		a.plan.Batches = append(a.plan.Batches, DependencyBatch{Indices: ready})
		for _, n := range ready {
			done[n] = true
			remaining--
			for _, to := range adj[n] {
				indeg[to]--
			}
		}
	}
	return nil
}

func (a *depAnalyzer) addEdge(from, to int, res DependencyResource, kind string) {
	if from == to {
		return
	}
	key := fmt.Sprintf("%d|%d|%s|%s", from, to, res.String(), kind)
	if _, ok := a.edgeSeen[key]; ok {
		return
	}
	a.edgeSeen[key] = struct{}{}
	ed := DependencyEdge{
		From: from, To: to, Resource: res, Conflict: kind,
		FromVer: a.entries[from].Ver, ToVer: a.entries[to].Ver,
	}
	a.plan.Edges = append(a.plan.Edges, ed)
	a.segEdges = append(a.segEdges, ed)
}

func (a *depAnalyzer) observeRead(r DependencyResource, n int) {
	if w, ok := a.writer[r]; ok && w != n {
		a.addEdge(w, n, r, ConflictRAW)
	}
	a.readers[r] = append(a.readers[r], n)
	if r.Class == ResChunk {
		if w, ok := a.chunkAll[r.Ino]; ok && w != n {
			a.addEdge(w, n, DependencyResource{Class: ResChunkAll, Ino: r.Ino}, ConflictRAW)
		}
	}
}

func (a *depAnalyzer) observeWrite(r DependencyResource, n int) {
	if r.Class == ResChunkAll {
		// TRUNCATE conflicts with every precise chunk of the inode. Precise
		// entries are consumed here; later writes also see this wildcard via
		// chunkAll, so no pair of entries is ever compared directly.
		if w, ok := a.writer[r]; ok && w != n {
			a.addEdge(w, n, r, ConflictWAW)
		}
		for indx := range a.chunkSet[r.Ino] {
			precise := DependencyResource{Class: ResChunk, Ino: r.Ino, Index: indx}
			if w, ok := a.writer[precise]; ok && w != n {
				a.addEdge(w, n, r, ConflictWAW)
			}
			for _, rd := range a.readers[precise] {
				if rd != n {
					a.addEdge(rd, n, r, ConflictWAR)
				}
			}
			delete(a.writer, precise)
			delete(a.readers, precise)
		}
		delete(a.chunkSet, r.Ino)
		a.writer[r] = n
		a.chunkAll[r.Ino] = n
		return
	}

	if w, ok := a.writer[r]; ok && w != n {
		a.addEdge(w, n, r, ConflictWAW)
	}
	for _, rd := range a.readers[r] {
		if rd != n {
			a.addEdge(rd, n, r, ConflictWAR)
		}
	}
	delete(a.readers, r)
	a.writer[r] = n

	if r.Class == ResChunk {
		if w, ok := a.chunkAll[r.Ino]; ok && w != n {
			a.addEdge(w, n, DependencyResource{Class: ResChunkAll, Ino: r.Ino}, ConflictWAW)
		}
		if a.chunkSet[r.Ino] == nil {
			a.chunkSet[r.Ino] = make(map[uint32]struct{})
		}
		a.chunkSet[r.Ino][r.Index] = struct{}{}
	}
}

func inodeRes(ino Ino) DependencyResource {
	return DependencyResource{Class: ResInode, Ino: ino}
}

func dentryRes(parent Ino, name string) DependencyResource {
	return DependencyResource{Class: ResDentry, Parent: parent, Name: name}
}

func chunkRes(ino Ino, index uint32) DependencyResource {
	return DependencyResource{Class: ResChunk, Ino: ino, Index: index}
}

func attrRes(ino Ino) DependencyResource {
	return DependencyResource{Class: ResAttrSet, Ino: ino}
}

func xattrRes(ino Ino, name string) DependencyResource {
	return DependencyResource{Class: ResXattr, Ino: ino, Xattr: name}
}

// footprintBuilder deduplicates resources within each set while preserving
// first-seen order. A resource may legitimately be both read and written by
// the same entry (e.g. UNLINK on a directory entry), so the two sets have
// independent de-duplication.
type footprintBuilder struct {
	reads     []DependencyResource
	writes    []DependencyResource
	seenRead  map[DependencyResource]struct{}
	seenWrite map[DependencyResource]struct{}
}

func newFootprintBuilder() *footprintBuilder {
	return &footprintBuilder{
		seenRead:  make(map[DependencyResource]struct{}),
		seenWrite: make(map[DependencyResource]struct{}),
	}
}

func (b *footprintBuilder) read(r DependencyResource) {
	if _, ok := b.seenRead[r]; ok {
		return
	}
	b.seenRead[r] = struct{}{}
	b.reads = append(b.reads, r)
}

func (b *footprintBuilder) write(r DependencyResource) {
	if _, ok := b.seenWrite[r]; ok {
		return
	}
	b.seenWrite[r] = struct{}{}
	b.writes = append(b.writes, r)
}

// entryFootprint resolves the typed read/write footprint of one entry. A
// barrier result means the entry must conservatively split the plan; a
// non-nil error means a typed entry carries arguments that cannot be parsed.
func entryFootprint(e *ChangeEntry) (reads, writes []DependencyResource, barrier bool, err error) {
	b := newFootprintBuilder()
	switch e.Op {
	case OpCreate:
		// CREATE(parent, name, ..., updateParent, rdev, srcMode): newInode
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		parent, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		name, err := e.Str(1)
		if err != nil {
			return nil, nil, false, err
		}
		updateParent, err := e.Bool(9)
		if err != nil {
			return nil, nil, false, err
		}
		newInode, err := e.ResultIno(0)
		if err != nil {
			return nil, nil, false, err
		}
		b.write(dentryRes(parent, name))
		b.write(inodeRes(newInode))
		if updateParent {
			b.write(inodeRes(parent))
		}
	case OpLink:
		// LINK(inode, parent, name, updateParent): nlink
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		ino, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		parent, err := e.Ino(1)
		if err != nil {
			return nil, nil, false, err
		}
		name, err := e.Str(2)
		if err != nil {
			return nil, nil, false, err
		}
		updateParent, err := e.Bool(3)
		if err != nil {
			return nil, nil, false, err
		}
		b.read(inodeRes(ino))
		b.write(inodeRes(ino))
		b.write(dentryRes(parent, name))
		if updateParent {
			b.write(inodeRes(parent))
		}
	case OpUnlink:
		// UNLINK(parent, name, trash, opened, updateParent): inode
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		parent, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		name, err := e.Str(1)
		if err != nil {
			return nil, nil, false, err
		}
		trash, err := e.Uint64(2)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Bool(3); err != nil {
			return nil, nil, false, err
		}
		updateParent, err := e.Bool(4)
		if err != nil {
			return nil, nil, false, err
		}
		ino, err := e.ResultIno(0)
		if err != nil {
			return nil, nil, false, err
		}
		// The old name must be resolved before the entry is removed.
		b.read(dentryRes(parent, name))
		b.write(dentryRes(parent, name))
		b.write(inodeRes(ino))
		if updateParent {
			b.write(inodeRes(parent))
		}
		if trash != 0 {
			// The detached node is moved into the trash directory.
			b.write(inodeRes(Ino(trash)))
		}
	case OpMove:
		// MOVE(ps, nameSrc, pd, nameDst, flags, dino, trash, opened,
		//      updateSrc, updateDst): inode
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		parentSrc, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		nameSrc, err := e.Str(1)
		if err != nil {
			return nil, nil, false, err
		}
		parentDst, err := e.Ino(2)
		if err != nil {
			return nil, nil, false, err
		}
		nameDst, err := e.Str(3)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint32(4); err != nil {
			return nil, nil, false, err
		}
		overwritten, err := e.Ino(5)
		if err != nil {
			return nil, nil, false, err
		}
		trash, err := e.Ino(6)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Bool(7); err != nil {
			return nil, nil, false, err
		}
		updateSrc, err := e.Bool(8)
		if err != nil {
			return nil, nil, false, err
		}
		updateDst, err := e.Bool(9)
		if err != nil {
			return nil, nil, false, err
		}
		inode, err := e.ResultIno(0)
		if err != nil {
			return nil, nil, false, err
		}
		// The old name is read before both directory entries are rewritten,
		// connecting src and dst directories even for cross-directory moves.
		b.read(dentryRes(parentSrc, nameSrc))
		b.write(dentryRes(parentSrc, nameSrc))
		b.write(dentryRes(parentDst, nameDst))
		b.write(inodeRes(inode))
		if overwritten != 0 {
			b.write(inodeRes(overwritten))
		}
		if updateSrc {
			b.write(inodeRes(parentSrc))
		}
		if updateDst {
			b.write(inodeRes(parentDst))
		}
		if trash != 0 {
			b.write(inodeRes(trash))
		}
	case OpSetAttr:
		// SETATTR(inode, set, sugidClear, attr...)
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		ino, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint16(1); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint8(2); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.AttrFields(3); err != nil {
			return nil, nil, false, err
		}
		b.read(inodeRes(ino))
		b.write(attrRes(ino))
	case OpWrite:
		// WRITE(inode, indx, off, sliceId, sliceLen, mtime, mtimensec): nSlices
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		ino, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		indx, err := e.Uint32(1)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint32(2); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint64(3); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint32(4); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Int64(5); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint32(6); err != nil {
			return nil, nil, false, err
		}
		b.read(inodeRes(ino))
		// Only the chunk is a conflict domain here: the recorded mtime is
		// carried by the entry itself, so writes to different chunks of one
		// inode stay independent. Explicit attribute changes go through
		// SETATTR/TRUNCATE and the ResAttrSet resource.
		b.write(chunkRes(ino, indx))
	case OpTruncate:
		// TRUNCATE(inode, oldLength, length, flags)
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		ino, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint64(1); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint64(2); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint8(3); err != nil {
			return nil, nil, false, err
		}
		b.read(inodeRes(ino))
		// Every chunk past the new length disappears: conflict on all chunks.
		b.write(DependencyResource{Class: ResChunkAll, Ino: ino})
		b.write(attrRes(ino))
	case OpSetXattr:
		// SETXATTR(inode, name, value, flags)
		if err = e.Validate(); err != nil {
			return nil, nil, false, err
		}
		ino, err := e.Ino(0)
		if err != nil {
			return nil, nil, false, err
		}
		name, err := e.Str(1)
		if err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Str(2); err != nil {
			return nil, nil, false, err
		}
		if _, err := e.Uint32(3); err != nil {
			return nil, nil, false, err
		}
		b.read(inodeRes(ino))
		b.write(xattrRes(ino, name))
	default:
		// Unknown operations, variadic batches and global quota/ACL/session
		// operations: never guess independence, force a conservative barrier.
		return nil, nil, true, nil
	}
	return b.reads, b.writes, false, nil
}
