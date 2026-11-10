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
	"reflect"
	"strings"
	"testing"
)

// ---- changelog fixture builders ------------------------------------------------

const depsAttrTail = "1000,1001,420,0,100,200,1,2,300,3,7,1"

func depsEntry(t *testing.T, ver int64, body string, sid, txn uint64) *ChangeEntry {
	t.Helper()
	line := fmt.Sprintf("%d.0|%s|(%d,%d)", ver, body, sid, txn)
	e, err := ParseChangeEntry(ver, line)
	if err != nil {
		t.Fatalf("parse %q: %s", line, err)
	}
	if err := e.Validate(); err != nil {
		t.Fatalf("validate %q: %s", line, err)
	}
	return e
}

// depsStream turns bodies into version-ordered entries, one txn per entry
// unless a body carries an explicit (sid, txn) override.
func depsStream(t *testing.T, bodies []string) []*ChangeEntry {
	entries := make([]*ChangeEntry, len(bodies))
	for i, body := range bodies {
		entries[i] = depsEntry(t, int64(i+1), body, 1, uint64(i+1))
	}
	return entries
}

func createBody(parent Ino, name string, ino Ino, updateParent bool) string {
	return fmt.Sprintf("CREATE(%d,%s,1000,1000,1,420,18,,Keep,%t,0,420):%d", parent, name, updateParent, ino)
}

func linkBody(ino, parent Ino, name string) string {
	return fmt.Sprintf("LINK(%d,%d,%s,true):2", ino, parent, name)
}

func unlinkBody(parent Ino, name string, ino Ino) string {
	return fmt.Sprintf("UNLINK(%d,%s,0,false,true):%d", parent, name, ino)
}

func moveBody(ps Ino, ns string, pd Ino, nd string, dino, ino Ino) string {
	return fmt.Sprintf("MOVE(%d,%s,%d,%s,0,%d,0,false,true,true):%d", ps, ns, pd, nd, dino, ino)
}

func writeBody(ino Ino, indx, sliceId uint32) string {
	return fmt.Sprintf("WRITE(%d,%d,0,%d,100,100,0):1", ino, indx, sliceId)
}

func truncateBody(ino Ino) string {
	return fmt.Sprintf("TRUNCATE(%d,100,0,0)", ino)
}

func setAttrBody(ino Ino) string {
	return fmt.Sprintf("SETATTR(%d,2,0,%s)", ino, depsAttrTail)
}

func setXattrBody(ino Ino, name string) string {
	return fmt.Sprintf("SETXATTR(%d,%s,value,0)", ino, name)
}

// ---- independent simplified namespace / slice model ----------------------------

type modelDentry struct {
	parent Ino
	name   string
}

type modelInode struct {
	nlink  int
	attr   int
	xattrs map[string]int
	chunks map[uint32][]uint64
}

type simNamespace struct {
	dentries map[modelDentry]Ino
	inodes   map[Ino]*modelInode
}

func newSimNamespace() *simNamespace {
	return &simNamespace{
		dentries: make(map[modelDentry]Ino),
		inodes:   make(map[Ino]*modelInode),
	}
}

func (m *simNamespace) apply(e *ChangeEntry) error {
	switch e.Op {
	case OpCreate:
		parent, _ := e.Ino(0)
		name := e.Args[1]
		ino, err := e.ResultIno(0)
		if err != nil {
			return err
		}
		key := modelDentry{parent, name}
		if _, ok := m.dentries[key]; ok {
			return fmt.Errorf("CREATE: dentry %s already exists", key.name)
		}
		if _, ok := m.inodes[ino]; ok {
			return fmt.Errorf("CREATE: inode %d already exists", ino)
		}
		m.dentries[key] = ino
		m.inodes[ino] = &modelInode{nlink: 1, xattrs: map[string]int{}, chunks: map[uint32][]uint64{}}
	case OpLink:
		ino, _ := e.Ino(0)
		parent, _ := e.Ino(1)
		name := e.Args[2]
		node, ok := m.inodes[ino]
		if !ok {
			return fmt.Errorf("LINK: inode %d does not exist", ino)
		}
		key := modelDentry{parent, name}
		if _, ok := m.dentries[key]; ok {
			return fmt.Errorf("LINK: dentry %s already exists", name)
		}
		node.nlink++
		m.dentries[key] = ino
	case OpUnlink:
		parent, _ := e.Ino(0)
		name := e.Args[1]
		ino, err := e.ResultIno(0)
		if err != nil {
			return err
		}
		key := modelDentry{parent, name}
		got, ok := m.dentries[key]
		if !ok {
			return fmt.Errorf("UNLINK: dentry %s does not exist", name)
		}
		if got != ino {
			return fmt.Errorf("UNLINK: dentry %s resolves to %d, want %d", name, got, ino)
		}
		delete(m.dentries, key)
		node := m.inodes[ino]
		node.nlink--
		if node.nlink == 0 {
			delete(m.inodes, ino)
		}
	case OpMove:
		ps, _ := e.Ino(0)
		ns := e.Args[1]
		pd, _ := e.Ino(2)
		nd := e.Args[3]
		dino, _ := e.Ino(5)
		ino, err := e.ResultIno(0)
		if err != nil {
			return err
		}
		src := modelDentry{ps, ns}
		if got, ok := m.dentries[src]; !ok || got != ino {
			return fmt.Errorf("MOVE: source dentry %s missing or wrong inode", ns)
		}
		dst := modelDentry{pd, nd}
		if existing, ok := m.dentries[dst]; ok {
			if dino == 0 || existing != dino {
				return fmt.Errorf("MOVE: unexpected overwrite of inode %d", existing)
			}
			node := m.inodes[dino]
			node.nlink--
			if node.nlink == 0 {
				delete(m.inodes, dino)
			}
		} else if dino != 0 {
			return fmt.Errorf("MOVE: expected overwritten inode %d not found", dino)
		}
		delete(m.dentries, src)
		m.dentries[dst] = ino
	case OpSetAttr:
		ino, _ := e.Ino(0)
		node, ok := m.inodes[ino]
		if !ok {
			return fmt.Errorf("SETATTR: inode %d does not exist", ino)
		}
		node.attr++
	case OpWrite:
		ino, _ := e.Ino(0)
		indx, _ := e.Uint32(1)
		sliceId, _ := e.Uint64(3)
		node, ok := m.inodes[ino]
		if !ok {
			return fmt.Errorf("WRITE: inode %d does not exist", ino)
		}
		node.chunks[indx] = append(node.chunks[indx], sliceId)
	case OpTruncate:
		ino, _ := e.Ino(0)
		node, ok := m.inodes[ino]
		if !ok {
			return fmt.Errorf("TRUNCATE: inode %d does not exist", ino)
		}
		node.chunks = map[uint32][]uint64{}
	case OpSetXattr:
		ino, _ := e.Ino(0)
		name := e.Args[1]
		node, ok := m.inodes[ino]
		if !ok {
			return fmt.Errorf("SETXATTR: inode %d does not exist", ino)
		}
		node.xattrs[name]++
	default:
		// Barrier operations have no modeled state of their own.
	}
	return nil
}

func (m *simNamespace) snapshot() simNamespace {
	cp := simNamespace{dentries: map[modelDentry]Ino{}, inodes: map[Ino]*modelInode{}}
	for k, v := range m.dentries {
		cp.dentries[k] = v
	}
	for ino, node := range m.inodes {
		cloned := &modelInode{nlink: node.nlink, attr: node.attr,
			xattrs: map[string]int{}, chunks: map[uint32][]uint64{}}
		for k, v := range node.xattrs {
			cloned.xattrs[k] = v
		}
		for k, v := range node.chunks {
			cloned.chunks[k] = append([]uint64(nil), v...)
		}
		cp.inodes[ino] = cloned
	}
	return cp
}

// ---- permutation enumeration ---------------------------------------------------

func allPermutations(n int) [][]int {
	var out [][]int
	xs := make([]int, n)
	for i := range xs {
		xs[i] = i
	}
	var rec func(k int)
	rec = func(k int) {
		if k == 1 {
			out = append(out, append([]int(nil), xs...))
			return
		}
		for i := 0; i < k; i++ {
			rec(k - 1)
			if k%2 == 0 {
				xs[i], xs[k-1] = xs[k-1], xs[i]
			} else {
				xs[0], xs[k-1] = xs[k-1], xs[0]
			}
		}
	}
	if n > 0 {
		rec(n)
	}
	return out
}

func planSegments(plan *DependencyPlan, n int) [][]int {
	var segs [][]int
	start := 0
	for _, b := range plan.Barriers {
		if b.Index > start {
			seg := make([]int, 0, b.Index-start)
			for i := start; i < b.Index; i++ {
				seg = append(seg, i)
			}
			segs = append(segs, seg)
		}
		start = b.Index
	}
	if start < n {
		seg := make([]int, 0, n-start)
		for i := start; i < n; i++ {
			seg = append(seg, i)
		}
		segs = append(segs, seg)
	}
	return segs
}

func enumerateOrders(segs [][]int) [][]int {
	orders := [][]int{{}}
	for _, seg := range segs {
		var next [][]int
		for _, perm := range allPermutations(len(seg)) {
			mapped := make([]int, len(perm))
			for i, p := range perm {
				mapped[i] = seg[p]
			}
			for _, prefix := range orders {
				order := append(append([]int(nil), prefix...), mapped...)
				next = append(next, order)
			}
		}
		orders = next
	}
	return orders
}

func orderRespectsEdges(order []int, edges []DependencyEdge) bool {
	pos := make(map[int]int, len(order))
	for i, idx := range order {
		pos[idx] = i
	}
	for _, ed := range edges {
		if pos[ed.From] >= pos[ed.To] {
			return false
		}
	}
	return true
}

// checkBatchInvariants verifies batches partition the nodes, hold no internal
// edge, and schedule every edge from an earlier batch to a later batch.
func checkBatchInvariants(t *testing.T, plan *DependencyPlan, n int) {
	t.Helper()
	batchOf := make(map[int]int)
	for bi, batch := range plan.Batches {
		for _, idx := range batch.Indices {
			if _, dup := batchOf[idx]; dup {
				t.Fatalf("node %d appears in multiple batches", idx)
			}
			batchOf[idx] = bi
		}
	}
	for idx := 0; idx < n; idx++ {
		isBarrier := false
		for _, b := range plan.Barriers {
			if b.Index == idx {
				isBarrier = true
			}
		}
		if !isBarrier {
			if _, ok := batchOf[idx]; !ok {
				t.Fatalf("node %d is missing from batches", idx)
			}
		}
	}
	for _, ed := range plan.Edges {
		bf, ok1 := batchOf[ed.From]
		bt, ok2 := batchOf[ed.To]
		if !ok1 || !ok2 {
			t.Fatalf("edge %d->%d touches a missing batch node", ed.From, ed.To)
		}
		if bf >= bt {
			t.Fatalf("edge %d->%d not ordered across batches %d,%d", ed.From, ed.To, bf, bt)
		}
	}
	// No edge may cross a barrier.
	for _, ed := range plan.Edges {
		for _, b := range plan.Barriers {
			lo, hi := ed.From, ed.To
			if lo > hi {
				lo, hi = hi, lo
			}
			if lo < b.Index && hi >= b.Index {
				t.Fatalf("edge %d->%d crosses barrier at %d", ed.From, ed.To, b.Index)
			}
		}
	}
}

// ---- footprint unit tests ------------------------------------------------------

func resourceClasses(rs []DependencyResource) []ResourceClass {
	if len(rs) == 0 {
		return nil
	}
	out := make([]ResourceClass, len(rs))
	for i, r := range rs {
		out[i] = r.Class
	}
	return out
}

func TestAnalyzeFootprints(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantReads  []ResourceClass
		wantWrites []ResourceClass
	}{
		{"create updates parent", createBody(1, "f", 10, true),
			nil, []ResourceClass{ResDentry, ResInode, ResInode}},
		{"create leaves parent untouched", createBody(1, "f", 10, false),
			nil, []ResourceClass{ResDentry, ResInode}},
		{"link", linkBody(10, 1, "g"),
			[]ResourceClass{ResInode}, []ResourceClass{ResInode, ResDentry, ResInode}},
		{"unlink reads old name", unlinkBody(1, "f", 10),
			[]ResourceClass{ResDentry}, []ResourceClass{ResDentry, ResInode, ResInode}},
		{"move dual directory", moveBody(1, "a", 2, "b", 0, 10),
			[]ResourceClass{ResDentry},
			[]ResourceClass{ResDentry, ResDentry, ResInode, ResInode, ResInode}},
		{"setattr", setAttrBody(10),
			[]ResourceClass{ResInode}, []ResourceClass{ResAttrSet}},
		{"write", writeBody(10, 0, 100),
			[]ResourceClass{ResInode}, []ResourceClass{ResChunk}},
		{"truncate", truncateBody(10),
			[]ResourceClass{ResInode}, []ResourceClass{ResChunkAll, ResAttrSet}},
		{"setxattr", setXattrBody(10, "user.foo"),
			[]ResourceClass{ResInode}, []ResourceClass{ResXattr}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := AnalyzeChangelogDependencies(depsStream(t, []string{tc.body}))
			if err != nil {
				t.Fatal(err)
			}
			if len(plan.Nodes) != 1 {
				t.Fatalf("got %d nodes, want 1", len(plan.Nodes))
			}
			node := plan.Nodes[0]
			if len(node.Reads) != len(tc.wantReads) ||
				!reflect.DeepEqual(resourceClasses(node.Reads), tc.wantReads) {
				t.Fatalf("reads = %v, want %v", resourceClasses(node.Reads), tc.wantReads)
			}
			if !reflect.DeepEqual(resourceClasses(node.Writes), tc.wantWrites) {
				t.Fatalf("writes = %v, want %v", resourceClasses(node.Writes), tc.wantWrites)
			}
		})
	}
}

func TestAnalyzeResourceKeys(t *testing.T) {
	entries := depsStream(t, []string{
		moveBody(1, "a", 2, "b", 30, 10),
		setXattrBody(10, "user.foo"),
		writeBody(10, 7, 100),
		truncateBody(10),
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct{}{
		`dentry:1/"a"`:        {},
		`dentry:2/"b"`:        {},
		`inode:10`:            {},
		`inode:30`:            {},
		`inode:1`:             {},
		`inode:2`:             {},
		`xattr:10/"user.foo"`: {},
		`chunk:10#7`:          {},
		`chunkall:10`:         {},
		`attr:10`:             {},
	}
	got := map[string]struct{}{}
	for _, node := range plan.Nodes {
		for _, r := range append(append([]DependencyResource{}, node.Reads...), node.Writes...) {
			got[r.String()] = struct{}{}
		}
	}
	for k := range want {
		if _, ok := got[k]; !ok {
			t.Fatalf("missing resource key %q, got %v", k, got)
		}
	}
}

// ---- edge-level scenario tests -------------------------------------------------

func findEdge(edges []DependencyEdge, from, to int, class ResourceClass) (DependencyEdge, bool) {
	for _, ed := range edges {
		if ed.From == from && ed.To == to && ed.Resource.Class == class {
			return ed, true
		}
	}
	return DependencyEdge{}, false
}

func TestAnalyzeHardlinkEdges(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "f", 10, false), // 0
		linkBody(10, 1, "g"),          // 1 alias of inode 10
		writeBody(10, 0, 100),         // 2 reaches inode 10 through either name
		unlinkBody(1, "g", 10),        // 3 removes one alias
		unlinkBody(1, "f", 10),        // 4 removes the last alias
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	// CREATE returns inode 10, LINK then reads and writes the same inode.
	if ed, ok := findEdge(plan.Edges, 0, 1, ResInode); !ok {
		t.Fatalf("missing inode edge create->link: %+v", plan.Edges)
	} else if ed.FromVer != 1 || ed.ToVer != 2 {
		t.Fatalf("edge versions = %d->%d, want 1->2", ed.FromVer, ed.ToVer)
	}
	// Alias and unlink connect through the shared inode, not only dentries.
	if _, ok := findEdge(plan.Edges, 1, 3, ResInode); !ok {
		t.Fatalf("missing inode edge link->unlink(alias): %+v", plan.Edges)
	}
	if _, ok := findEdge(plan.Edges, 3, 4, ResInode); !ok {
		t.Fatalf("missing inode edge unlink(g)->unlink(f): %+v", plan.Edges)
	}
	// WRITE and unlink both touch inode 10.
	if _, ok := findEdge(plan.Edges, 1, 2, ResInode); !ok {
		t.Fatalf("missing inode edge link->write: %+v", plan.Edges)
	}
	checkBatchInvariants(t, plan, len(entries))
}

func TestAnalyzeMultipleRenames(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "a", 10, false),   // 0
		moveBody(1, "a", 1, "b", 0, 10), // 1
		moveBody(1, "b", 2, "c", 0, 10), // 2 cross-directory
		moveBody(2, "c", 2, "d", 0, 10), // 3
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	// Each rename reads the name written by its predecessor.
	for _, pair := range [][2]int{{0, 1}, {1, 2}, {2, 3}} {
		if _, ok := findEdge(plan.Edges, pair[0], pair[1], ResDentry); !ok {
			t.Fatalf("missing dentry edge %d->%d: %+v", pair[0], pair[1], plan.Edges)
		}
	}
	// Cross-directory move touches both parent inodes.
	if _, ok := findEdge(plan.Edges, 1, 2, ResInode); !ok {
		t.Fatalf("missing inode edge across directories: %+v", plan.Edges)
	}
	checkBatchInvariants(t, plan, len(entries))
}

func TestAnalyzeChunkConflicts(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "f", 10, false), // 0
		writeBody(10, 0, 100),         // 1
		writeBody(10, 0, 200),         // 2 same chunk, overwrite order matters
		writeBody(10, 1, 300),         // 3 different chunk, parallelizable with 2
		truncateBody(10),              // 4 conflicts with every chunk
		writeBody(10, 0, 400),         // 5 must follow truncate
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	ed, ok := findEdge(plan.Edges, 1, 2, ResChunk)
	if !ok || ed.Conflict != ConflictWAW {
		t.Fatalf("same chunk needs WAW edge, got %+v ok=%v", ed, ok)
	}
	// Different chunks have no direct chunk edge.
	if _, ok := findEdge(plan.Edges, 2, 3, ResChunk); ok {
		t.Fatalf("different chunks must not conflict: %+v", plan.Edges)
	}
	if _, ok := findEdge(plan.Edges, 3, 2, ResChunk); ok {
		t.Fatalf("different chunks must not conflict: %+v", plan.Edges)
	}
	// TRUNCATE connects to precise chunks on both sides via chunkAll.
	if _, ok := findEdge(plan.Edges, 3, 4, ResChunkAll); !ok {
		t.Fatalf("missing chunkAll edge write->truncate: %+v", plan.Edges)
	}
	if ed, ok := findEdge(plan.Edges, 4, 5, ResChunkAll); !ok || ed.Conflict != ConflictWAW {
		t.Fatalf("missing chunkAll WAW edge truncate->write: %+v", ed)
	}
	// Writes to different chunks may share a batch: node 1 (chunk 0) and
	// node 3 (chunk 1) only depend on the create, not on each other.
	foundParallelBatch := false
	for _, batch := range plan.Batches {
		has1, has3 := false, false
		for _, idx := range batch.Indices {
			has1 = has1 || idx == 1
			has3 = has3 || idx == 3
		}
		foundParallelBatch = foundParallelBatch || (has1 && has3)
	}
	if !foundParallelBatch {
		t.Fatalf("writes on different chunks were not scheduled together: %+v", plan.Batches)
	}
	checkBatchInvariants(t, plan, len(entries))
}

func TestAnalyzeSameNameRebuild(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "f", 10, false), // 0
		unlinkBody(1, "f", 10),        // 1 old name must be read first
		createBody(1, "f", 20, false), // 2 rebuild under the same name, new inode
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	if ed, ok := findEdge(plan.Edges, 0, 1, ResDentry); !ok || ed.Conflict != ConflictRAW {
		t.Fatalf("unlink must read the dentry written by create (RAW): %+v", ed)
	}
	if _, ok := findEdge(plan.Edges, 1, 2, ResDentry); !ok {
		t.Fatalf("rebuild must wait for unlink on the dentry: %+v", plan.Edges)
	}
	// Inode 10 and inode 20 have no inode-level edge between them.
	if _, ok := findEdge(plan.Edges, 0, 2, ResInode); ok {
		t.Fatalf("rebuilt inode 20 must not inherit inode 10's ordering: %+v", plan.Edges)
	}
	checkBatchInvariants(t, plan, len(entries))
}

func TestAnalyzeMoveOverwrite(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "dst", 20, false),      // 0 victim
		createBody(1, "src", 10, false),      // 1 moved inode
		moveBody(1, "src", 1, "dst", 20, 10), // 2 overwrites inode 20
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := findEdge(plan.Edges, 0, 2, ResInode); !ok {
		t.Fatalf("overwritten inode 20 must connect to the move: %+v", plan.Edges)
	}
	if _, ok := findEdge(plan.Edges, 1, 2, ResInode); !ok {
		t.Fatalf("moved inode 10 must connect to the move: %+v", plan.Edges)
	}
	checkBatchInvariants(t, plan, len(entries))
}

// ---- barriers ------------------------------------------------------------------

func TestAnalyzeBarrierOperations(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "f", 10, false),                // 0
		"SETQUOTA(1,2,3,4,5,6)",                      // 1 global quota side effect
		"UNLINKBATCH(1,a,b,0,true):11,false,12,true", // 2 variadic, unexplainable
		createBody(1, "g", 20, false),                // 3
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Barriers) != 2 {
		t.Fatalf("got %d barriers, want 2: %+v", len(plan.Barriers), plan.Barriers)
	}
	if plan.Barriers[0].Index != 1 || !strings.Contains(plan.Barriers[0].Reason, OpSetQuota) {
		t.Fatalf("first barrier = %+v", plan.Barriers[0])
	}
	if plan.Barriers[1].Index != 2 || !strings.Contains(plan.Barriers[1].Reason, OpUnlinkBatch) {
		t.Fatalf("second barrier = %+v", plan.Barriers[1])
	}
	if len(plan.Nodes) != 2 {
		t.Fatalf("barrier entries must not produce nodes, got %d", len(plan.Nodes))
	}
	checkBatchInvariants(t, plan, len(entries))
}

func TestAnalyzeWindowCut(t *testing.T) {
	bodies := []string{
		createBody(1, "a", 10, false), // 0
		createBody(1, "b", 11, false), // 1
		writeBody(10, 0, 100),         // 2 would depend on 0, but a barrier cuts first
		createBody(1, "c", 12, false), // 3
		writeBody(11, 0, 200),         // 4 depends on 1 across the cut
	}
	plan, err := AnalyzeChangelogDependencies(depsStream(t, bodies), WithMaxWindow(2))
	if err != nil {
		t.Fatal(err)
	}
	var cuts []int
	for _, b := range plan.Barriers {
		if strings.Contains(b.Reason, "max analysis window") {
			cuts = append(cuts, b.Index)
		}
	}
	if !reflect.DeepEqual(cuts, []int{2, 4}) {
		t.Fatalf("window cuts = %v, want [2 4]", cuts)
	}
	// The cross-window dependency is replaced by the barrier, never an edge.
	for _, ed := range plan.Edges {
		if (ed.From < 2 && ed.To >= 2) || (ed.From < 4 && ed.To >= 4 && ed.From >= 2) {
			t.Fatalf("edge %d->%d crosses a window cut", ed.From, ed.To)
		}
	}
	for _, batch := range plan.Batches {
		if len(batch.Indices) > 2 {
			t.Fatalf("batch %v exceeds the max window", batch.Indices)
		}
	}
	checkBatchInvariants(t, plan, len(bodies))
}

// ---- transaction ordering ------------------------------------------------------

func TestAnalyzeTxnOrdering(t *testing.T) {
	// updateParent=false so the entries share no resource: only txn edges apply.
	entries := []*ChangeEntry{
		depsEntry(t, 1, createBody(1, "a", 10, false), 1, 7),
		depsEntry(t, 2, createBody(1, "b", 11, false), 1, 7), // same session + txn
		depsEntry(t, 3, createBody(1, "c", 12, false), 2, 7), // other session, same txnId
		depsEntry(t, 4, createBody(1, "d", 13, false), 2, 7),
		depsEntry(t, 5, createBody(1, "e", 14, false), 1, 8), // same session, new txn
	}
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	var txnPairs [][2]int
	for _, ed := range plan.Edges {
		if ed.Conflict != ConflictTxn {
			t.Fatalf("unexpected non-txn edge: %+v", ed)
		}
		if ed.Resource.Class != ResTransaction || ed.Resource.Sid == 0 {
			t.Fatalf("txn edge carries no transaction resource: %+v", ed)
		}
		txnPairs = append(txnPairs, [2]int{ed.From, ed.To})
	}
	want := [][2]int{{0, 1}, {2, 3}}
	if !reflect.DeepEqual(txnPairs, want) {
		t.Fatalf("txn edges = %v, want %v", txnPairs, want)
	}
}

// ---- version validation --------------------------------------------------------

func TestAnalyzeVersionErrors(t *testing.T) {
	body := createBody(1, "f", 10, false)
	t.Run("gap", func(t *testing.T) {
		entries := []*ChangeEntry{
			depsEntry(t, 1, body, 1, 1),
			depsEntry(t, 2, body, 1, 2),
			depsEntry(t, 4, body, 1, 3),
		}
		if _, err := AnalyzeChangelogDependencies(entries); err == nil ||
			!strings.Contains(err.Error(), "version gap") {
			t.Fatalf("gap error = %v", err)
		}
	})
	t.Run("duplicate", func(t *testing.T) {
		entries := []*ChangeEntry{
			depsEntry(t, 1, body, 1, 1),
			depsEntry(t, 2, body, 1, 2),
			depsEntry(t, 2, body, 1, 3),
		}
		if _, err := AnalyzeChangelogDependencies(entries); err == nil ||
			!strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("duplicate error = %v", err)
		}
	})
	t.Run("out of order", func(t *testing.T) {
		entries := []*ChangeEntry{
			depsEntry(t, 3, body, 1, 1),
			depsEntry(t, 2, body, 1, 2),
			depsEntry(t, 1, body, 1, 3),
		}
		if _, err := AnalyzeChangelogDependencies(entries); err == nil ||
			!strings.Contains(err.Error(), "older") {
			t.Fatalf("order error = %v", err)
		}
	})
	t.Run("invalid window", func(t *testing.T) {
		if _, err := AnalyzeChangelogDependencies(nil, WithMaxWindow(0)); err == nil {
			t.Fatal("zero window should fail")
		}
	})
}

// ---- full topology soundness against the simplified model ----------------------

func TestAnalyzeTopologySoundness(t *testing.T) {
	scenarios := []struct {
		name             string
		bodies           []string
		requireDivergent bool // an edge-violating order must change the final state
		minRespecting    int
	}{
		{
			name: "hardlink lifecycle",
			bodies: []string{
				createBody(1, "f", 10, false),
				linkBody(10, 1, "g"),
				writeBody(10, 0, 100),
				unlinkBody(1, "g", 10),
				unlinkBody(1, "f", 10),
			},
			requireDivergent: true, minRespecting: 1,
		},
		{
			name: "multiple renames",
			bodies: []string{
				createBody(1, "a", 10, false),
				moveBody(1, "a", 1, "b", 0, 10),
				moveBody(1, "b", 2, "c", 0, 10),
				moveBody(2, "c", 2, "d", 0, 10),
			},
			requireDivergent: true, minRespecting: 1,
		},
		{
			name: "same chunk overwrite",
			bodies: []string{
				createBody(1, "f", 10, false),
				writeBody(10, 0, 100),
				writeBody(10, 0, 200),
			},
			requireDivergent: true, minRespecting: 1,
		},
		{
			name: "different chunks",
			bodies: []string{
				createBody(1, "f", 10, false),
				writeBody(10, 0, 100),
				writeBody(10, 1, 200),
			},
			requireDivergent: false, minRespecting: 2,
		},
		{
			name: "same name rebuild",
			bodies: []string{
				createBody(1, "f", 10, false),
				unlinkBody(1, "f", 10),
				createBody(1, "f", 20, false),
			},
			requireDivergent: true, minRespecting: 1,
		},
		{
			name: "unknown op barrier",
			bodies: []string{
				createBody(1, "f", 10, false),
				"SETQUOTA(1,2,3,4,5,6)",
				createBody(1, "g", 20, false),
			},
			requireDivergent: false, minRespecting: 1,
		},
		{
			name: "truncate vs writes",
			bodies: []string{
				createBody(1, "f", 10, false),
				writeBody(10, 0, 100),
				truncateBody(10),
				writeBody(10, 0, 200),
			},
			requireDivergent: true, minRespecting: 1,
		},
		{
			name: "move overwrite",
			bodies: []string{
				createBody(1, "dst", 20, false),
				createBody(1, "src", 10, false),
				moveBody(1, "src", 1, "dst", 20, 10),
			},
			requireDivergent: true, minRespecting: 1,
		},
		{
			name: "independent xattrs",
			bodies: []string{
				createBody(1, "f", 10, false),
				setXattrBody(10, "user.a"),
				setXattrBody(10, "user.b"),
				setAttrBody(10),
			},
			// CREATE must run first; the three attribute operations then have
			// 3! valid topological orders.
			requireDivergent: false, minRespecting: 6,
		},
	}
	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			entries := depsStream(t, sc.bodies)
			plan, err := AnalyzeChangelogDependencies(entries)
			if err != nil {
				t.Fatal(err)
			}
			checkBatchInvariants(t, plan, len(entries))

			want := newSimNamespace()
			for _, e := range entries {
				if err := want.apply(e); err != nil {
					t.Fatalf("sequential model rejected %s: %s", e.Op, err)
				}
			}
			wantState := want.snapshot()

			respecting, divergent := 0, 0
			for _, order := range enumerateOrders(planSegments(plan, len(entries))) {
				okEdges := orderRespectsEdges(order, plan.Edges)
				m := newSimNamespace()
				rejected := false
				for _, idx := range order {
					if err := m.apply(entries[idx]); err != nil {
						rejected = true
						break
					}
				}
				same := !rejected && reflect.DeepEqual(m.snapshot(), wantState)
				if okEdges {
					respecting++
					if !same {
						t.Fatalf("edge-respecting order %v yields a different final state", order)
					}
				} else if !same {
					divergent++
				}
			}
			if respecting < sc.minRespecting {
				t.Fatalf("got %d edge-respecting orders, want at least %d", respecting, sc.minRespecting)
			}
			// For genuine conflict scenarios at least one forbidden ordering
			// must visibly corrupt the model, proving the edge is not spurious.
			if sc.requireDivergent && divergent == 0 {
				t.Fatal("every edge-violating order reached the same state; edges do not capture a real conflict")
			}
		})
	}
}

// TestAnalyzeNoMetadataSideEffect keeps the analyzer honest: it works on bare
// parsed entries without constructing any metadata client.
func TestAnalyzeNoMetadataSideEffect(t *testing.T) {
	entries := depsStream(t, []string{
		createBody(1, "f", 10, false),
		writeBody(10, 0, 100),
	})
	plan, err := AnalyzeChangelogDependencies(entries)
	if err != nil {
		t.Fatal(err)
	}
	if plan.MaxWindow != DefaultAnalyzeMaxWindow {
		t.Fatalf("default window = %d, want %d", plan.MaxWindow, DefaultAnalyzeMaxWindow)
	}
	if len(plan.Batches) == 0 {
		t.Fatal("plan should contain batches")
	}
}
